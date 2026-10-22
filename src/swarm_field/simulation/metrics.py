"""Metrics and benchmarking (B.5) — the passive observer of a run.

``MetricsCollector`` watches the ground truth the simulation already owns and turns it into a
series of ``MetricsSample`` rows: targets remaining over time, detection latency, energy per
unit area, messages sent, coverage, and how fresh the swarm keeps the field (idleness and
redundancy). It decides nothing and asks the algorithm nothing, so
attaching it cannot change how the swarm behaves.

The collector never draws from the simulation's random source. It holds no random source at
all, leaving the single ``SimulationContext.rng`` to the pest dynamics, so a run's world is
identical whether or not metrics are being recorded.

Detection latency and target counts are read at the sample period rather than at the instant
they happen: a cell that turns infested between two samples is first seen at the next one.
Both figures are therefore quantised to the period. That is a deliberate trade for keeping
the collector out of the dynamics, which would otherwise have to report to it.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from swarm_field.contracts import (
    CellId,
    InfestationObservation,
    MetricsSample,
    Observation,
    RobotId,
    Time,
    TimeUnit,
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
    _infested_at: dict[CellId, Time] = field(default_factory=dict)
    _detected_at: dict[CellId, Time] = field(default_factory=dict)
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
        for cell in self._crop_cells:
            if world.cells[cell].infestation > self.config.target_threshold:
                self._infested_at[cell] = now

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
            infested_at = self._infested_at.get(cell)
            seen_infested = observation.severity > self.config.target_threshold
            if infested_at is not None and seen_infested and cell not in self._detected_at:
                self._detected_at[cell] = now

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
            battery_failures=world.battery_failure_count(),
            mean_idleness_ms=self._mean_idleness(),
            worst_idleness_ms=self._worst_idleness,
            redundancy=self._redundancy(),
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
        """Remember when each crop cell first became an active target.

        Called wherever infestation can change, ``WorldTick`` and an injected pest, so a pest is
        dated when it appears rather than at the next sample; the sample calls it too.
        """
        for cell in self._crop_cells:
            if cell in self._infested_at:
                continue
            if world.cells[cell].infestation > self.config.target_threshold:
                self._infested_at[cell] = now

    def _targets_remaining(self, world: WorldState) -> int:
        """How many crop cells are still infested above the threshold."""
        return sum(
            1
            for cell in self._crop_cells
            if world.cells[cell].infestation > self.config.target_threshold
        )

    def _mean_detection_latency(self) -> float | None:
        """Mean delay from infestation to first sighting, or ``None`` if nothing is detected."""
        latencies = [
            self._detected_at[cell] - self._infested_at[cell]
            for cell in self._detected_at
            if cell in self._infested_at
        ]
        if not latencies:
            return None
        return sum(latencies) / len(latencies)

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
