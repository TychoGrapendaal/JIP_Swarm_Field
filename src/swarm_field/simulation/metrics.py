"""Metrics and benchmarking (B.5) — the passive observer of a run.

``MetricsCollector`` watches the ground truth the simulation already owns and turns it into a
series of ``MetricsSample`` rows: targets remaining over time, detection latency, energy per
unit area, messages sent, coverage, how fresh the swarm keeps the field (idleness and
redundancy), and how the killers respond to pests. It decides nothing, and asks the algorithm
only what ``SwarmAgent.inspect`` reports, so attaching it cannot change how the swarm behaves.

The collector never draws from the simulation's random source. It holds no random source at
all, leaving the single ``SimulationContext.rng`` to the pest dynamics, so a run's world is
identical whether or not metrics are being recorded.

Every pest is followed from the moment it appears to its clearing: ``WorldTick``, an injected
pest and a ``Treat`` report to the collector when infestation changes, robots' sensing reports
its detection, and the killers' inspections report when news of it reached a killer. The sample
period therefore only sets how often the figures are written down, not how exact they are.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path

from swarm_field.contracts import (
    Action,
    AgentInspection,
    AgentObjective,
    CellId,
    InfestationObservation,
    MetricsSample,
    Move,
    Observation,
    RobotId,
    Time,
    TimeUnit,
    Treat,
)
from swarm_field.simulation.world import WorldState

_METRIC_FIELDS = tuple(MetricsSample.__dataclass_fields__)


@dataclass(frozen=True)
class MetricsConfig:
    """How often metrics are recorded and where the thresholds sit.

    Attributes:
        period: Time between samples in simulation time units.
        target_threshold: Infestation severity at or below which a crop cell no longer counts
            as an active target.
        fresh_age: A robot arriving in a crop cell sensed less than this before makes a
            redundant visit.
        measure_from: Idleness is averaged over the samples from this time on, e.g. after a
            warm-up in which the swarm explores the field for the first time.
    """

    period: Time = TimeUnit.HOUR
    target_threshold: float = 0.0
    fresh_age: Time = 30 * TimeUnit.MINUTE
    measure_from: Time = 0

    def __post_init__(self) -> None:
        """Validate the sampling cadence and the target threshold."""
        if self.period <= 0:
            raise ValueError("metric sample period must be positive")
        if not 0.0 <= self.target_threshold <= 1.0:
            raise ValueError("target_threshold must be between 0 and 1")


class KillerActivity(Enum):
    """What a killer spent one decision on, for the killers' time split."""

    IDLE = auto()
    EXPLORING = auto()
    DRIVING = auto()
    TREATING = auto()

    @classmethod
    def from_decision(cls, action: Action, objective: AgentObjective | None) -> KillerActivity:
        """The activity of a decision that took ``action`` for ``objective``."""
        if isinstance(action, Treat):
            return cls.TREATING
        if objective is AgentObjective.TREAT and isinstance(action, Move):
            return cls.DRIVING
        if objective is AgentObjective.EXPLORE:
            return cls.EXPLORING
        return cls.IDLE


@dataclass
class PestRecord:
    """One pest's life, from appearing on a crop cell to being cleared.

    Attributes:
        cell: The infested cell.
        appeared_at: When its infestation rose above the target threshold.
        detected_at: When a robot first sensed it infested, if any has.
        known_at: When a killer first knew of it, if any has.
        cleared_at: When it fell back to the threshold or below, if it has.
    """

    cell: CellId
    appeared_at: Time
    detected_at: Time | None = None
    known_at: Time | None = None
    cleared_at: Time | None = None


@dataclass
class MetricsCollector:
    """Records a run's performance by reading ground truth, and keeps the series.

    Attributes:
        config: The sampling cadence and the target threshold.
        samples: Every sample taken so far, oldest first.
    """

    config: MetricsConfig = field(default_factory=MetricsConfig)
    samples: list[MetricsSample] = field(default_factory=list)
    _crop_cells: frozenset[CellId] = frozenset()
    _observed: set[CellId] = field(default_factory=set)
    _active_pests: dict[CellId, PestRecord] = field(default_factory=dict)
    _cleared_pests: list[PestRecord] = field(default_factory=list)
    _killer_activity: Counter[KillerActivity] = field(default_factory=Counter)
    _killer_targets: dict[RobotId, CellId] = field(default_factory=dict)
    _trips: int = 0
    _duplicate_trips: int = 0
    _start: Time = 0
    _last_sensed: dict[CellId, Time] = field(default_factory=dict)
    _robot_cells: dict[RobotId, CellId] = field(default_factory=dict)
    _visits: int = 0
    _redundant: int = 0
    _idleness_sum: float = 0.0
    _idleness_samples: int = 0
    _worst_idleness: float = 0.0

    def bind(self, world: WorldState, now: Time = 0) -> None:
        """Record which cells count as crop and when they were already infested.

        Called once when the run is built. A cell that is not cropped at that moment is never
        counted as covered, so coverage stays comparable between runs. Cells that already carry
        infestation at ``now`` are dated ``now`` rather than at the first sample, so a target the
        swarm starts on top of is still attributed a detection latency.
        """
        self._start = now
        self._crop_cells = frozenset(
            cell for cell, state in world.cells.items() if state.crop is not None
        )
        self.track_infestation(world, now)

    def observe(self, observations: Iterable[Observation], now: Time) -> None:
        """Record what a robot sensed at ``now``.

        A cell counts as sensed when its infestation is: a robot also sees the terrain of the
        cells around it, which tells nothing about their pests. Sensing a cell for the first time
        settles the coverage figure. Sensing an infested cell detects its infestation, also when
        the cell was sensed before the pest appeared, which settles its detection latency. Every
        sensing keeps the cell fresh for the idleness, and a robot arriving in a cell is a visit,
        redundant if the cell was sensed less than ``fresh_age`` before.
        """
        for observation in observations:
            if not isinstance(observation, InfestationObservation):
                continue
            cell = observation.cell
            if cell not in self._crop_cells:
                self._observed.add(cell)
                continue
            self._record_visit(observation.source, cell, now)
            self._last_sensed[cell] = now
            self._observed.add(cell)
            pest = self._active_pests.get(cell)
            seen_infested = observation.severity > self.config.target_threshold
            if pest is not None and seen_infested and pest.detected_at is None:
                pest.detected_at = now

    def observe_killer(
        self, robot: RobotId, action: Action, inspection: AgentInspection, now: Time
    ) -> None:
        """Record one decision of a killer: what it spent it on, where it was heading, and
        which pests it knew of.

        A pest is known from the first decision at which a killer reports it among its known
        pests. A trip is a duplicate when another killer's latest decision drove to the same pest.
        """
        for cell in inspection.known_pests:
            pest = self._active_pests.get(cell)
            if pest is not None and pest.known_at is None:
                pest.known_at = now
        activity = KillerActivity.from_decision(action, inspection.current_objective)
        self._killer_activity[activity] += 1
        target = inspection.current_target
        if activity is not KillerActivity.DRIVING or target is None:
            self._killer_targets.pop(robot, None)
            return
        self._killer_targets[robot] = target
        self._trips += 1
        for other, other_target in self._killer_targets.items():
            if other != robot and other_target == target:
                self._duplicate_trips += 1
                break

    def sample(
        self, world: WorldState, now: Time, messages_sent: int, messages_received: int = 0
    ) -> MetricsSample:
        """Read the world at ``now``, record one row of the series, and return it."""
        world.advance_to(now)
        self.track_infestation(world, now)
        self._track_idleness(now)

        taken = MetricsSample(
            time=now,
            targets_remaining=self._targets_remaining(world),
            detection_latency_ms=self._mean_detection_latency(),
            energy_per_area=self._energy_per_area(world),
            messages_sent=messages_sent,
            messages_received=messages_received,
            coverage=self._coverage(),
            active_robots=len(world.live_robots()),
            mean_idleness_ms=self._mean_idleness(),
            worst_idleness_ms=self._worst_idleness,
            redundancy=self._redundancy(),
            pests_appeared=len(self._active_pests) + len(self._cleared_pests),
            pests_cleared=len(self._cleared_pests),
            knowledge_latency_ms=self._mean_knowledge_latency(),
            response_time_ms=self._mean_response_time(),
            killer_idle_share=self._killer_share(KillerActivity.IDLE),
            killer_exploring_share=self._killer_share(KillerActivity.EXPLORING),
            killer_driving_share=self._killer_share(KillerActivity.DRIVING),
            killer_treating_share=self._killer_share(KillerActivity.TREATING),
            duplicate_trip_share=self._duplicate_trip_share(),
        )
        self.samples.append(taken)
        return taken

    def series(self) -> tuple[MetricsSample, ...]:
        """Every sample taken so far, oldest first."""
        return tuple(self.samples)

    def latest(self) -> MetricsSample | None:
        """The most recent sample, or ``None`` before the first one is taken."""
        return self.samples[-1] if self.samples else None

    def last_sensed_times(self) -> dict[CellId, Time]:
        """When each crop cell was last sensed; a cell never sensed is left out."""
        return dict(self._last_sensed)

    def _record_visit(self, robot: RobotId, cell: CellId, now: Time) -> None:
        """Count a visit when ``robot`` senses a cell other than the one it sensed last."""
        if self._robot_cells.get(robot) == cell:
            return
        self._robot_cells[robot] = cell
        self._visits += 1
        last = self._last_sensed.get(cell)
        if last is not None and now - last < self.config.fresh_age:
            self._redundant += 1

    def _track_idleness(self, now: Time) -> None:
        """Add the crop cells' ages at ``now`` to the idleness, from ``measure_from`` on."""
        if now < self.config.measure_from or not self._crop_cells:
            return
        total = 0
        for cell in self._crop_cells:
            age = now - self._last_sensed.get(cell, self._start)
            total += age
            self._worst_idleness = max(self._worst_idleness, float(age))
        self._idleness_sum += total / len(self._crop_cells)
        self._idleness_samples += 1

    def _mean_idleness(self) -> float:
        """The crop cells' mean age, averaged over the samples since ``measure_from``."""
        if self._idleness_samples == 0:
            return 0.0
        return self._idleness_sum / self._idleness_samples

    def _redundancy(self) -> float:
        """The share of visits to a cell sensed less than ``fresh_age`` before."""
        if self._visits == 0:
            return 0.0
        return self._redundant / self._visits

    def track_infestation(self, world: WorldState, now: Time) -> None:
        """Start a record for every crop cell that turned infested, and close the cleared ones.

        Called wherever infestation can change, ``WorldTick``, an injected pest and a ``Treat``,
        so a pest is dated when it appears and when it is cleared rather than at the next
        sample; the sample calls it too.
        """
        threshold = self.config.target_threshold
        for cell in self._crop_cells:
            infested = world.cells[cell].infestation > threshold
            pest = self._active_pests.get(cell)
            if infested and pest is None:
                self._active_pests[cell] = PestRecord(cell, now)
            elif not infested and pest is not None:
                pest.cleared_at = now
                self._cleared_pests.append(self._active_pests.pop(cell))

    def pests(self) -> tuple[PestRecord, ...]:
        """Every pest so far: the cleared ones in the order they were cleared, then the rest."""
        return (*self._cleared_pests, *self._active_pests.values())

    def _targets_remaining(self, world: WorldState) -> int:
        """How many crop cells are still infested above the threshold."""
        return sum(
            1
            for cell in self._crop_cells
            if world.cells[cell].infestation > self.config.target_threshold
        )

    def _mean_detection_latency(self) -> float | None:
        """Mean delay from a pest appearing to its first sighting, or ``None`` before any."""
        latencies: list[Time] = []
        for pest in self.pests():
            if pest.detected_at is not None:
                latencies.append(pest.detected_at - pest.appeared_at)
        return _mean_or_none(latencies)

    def _mean_knowledge_latency(self) -> float | None:
        """Mean delay from a pest's detection to a killer knowing of it, or ``None`` before any."""
        latencies: list[Time] = []
        for pest in self.pests():
            if pest.detected_at is not None and pest.known_at is not None:
                latencies.append(max(0, pest.known_at - pest.detected_at))
        return _mean_or_none(latencies)

    def _mean_response_time(self) -> float | None:
        """Mean delay from a pest's detection to its clearing, or ``None`` before any."""
        times: list[Time] = []
        for pest in self._cleared_pests:
            if pest.detected_at is not None and pest.cleared_at is not None:
                times.append(pest.cleared_at - pest.detected_at)
        return _mean_or_none(times)

    def _killer_share(self, activity: KillerActivity) -> float:
        """The share of the killers' decisions so far spent on ``activity``."""
        total = sum(self._killer_activity.values())
        if total == 0:
            return 0.0
        return self._killer_activity[activity] / total

    def _duplicate_trip_share(self) -> float:
        """The share of the killers' trip decisions that another killer duplicated."""
        if self._trips == 0:
            return 0.0
        return self._duplicate_trips / self._trips

    def _energy_per_area(self, world: WorldState) -> float:
        """Battery fraction spent per square metre of the whole field."""
        area_m2 = world.field.width * world.field.height * world.field.cell_size**2
        if area_m2 <= 0:
            return 0.0
        return world.energy_spent() / area_m2

    def _coverage(self) -> float:
        """Fraction of the crop cells at least one robot has sensed."""
        if not self._crop_cells:
            return 0.0
        covered = sum(1 for cell in self._crop_cells if cell in self._observed)
        return covered / len(self._crop_cells)


def _mean_or_none(values: Sequence[Time]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def write_metrics_csv(samples: Sequence[MetricsSample], path: Path) -> None:
    """Write a metrics series as CSV: one row per sample, columns in field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_METRIC_FIELDS))
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def write_metrics_json(samples: Sequence[MetricsSample], path: Path) -> None:
    """Write a metrics series as a JSON array holding one object per sample."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(s) for s in samples], indent=2), encoding="utf-8")
