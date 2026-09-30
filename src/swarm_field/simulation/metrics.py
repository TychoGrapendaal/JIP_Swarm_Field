"""Metrics and benchmarking (B.5) — the passive observer of a run.

``MetricsCollector`` watches the ground truth the simulation already owns and turns it into a
series of ``MetricsSample`` rows: targets remaining over time, detection latency, energy per
unit area, messages sent, and coverage. It decides nothing and asks the algorithm nothing, so
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

from swarm_field.contracts import CellId, MetricsSample, Observation, Time
from swarm_field.simulation.world import WorldState

_METRIC_FIELDS = tuple(MetricsSample.__dataclass_fields__)


@dataclass(frozen=True)
class MetricsConfig:
    """How often metrics are recorded and where the thresholds sit.

    Attributes:
        period: Time between samples in simulation time units.
        target_threshold: Infestation severity at or below which a crop cell no longer counts
            as an active target.
    """

    period: Time = 3_600_000
    target_threshold: float = 0.0

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

    def bind(self, world: WorldState, now: Time = 0) -> None:
        """Record which cells count as crop and when they were already infested.

        Called once when the run is built. A cell that is not cropped at that moment is never
        counted as covered, so coverage stays comparable between runs. Cells that already carry
        infestation at ``now`` are dated ``now`` rather than at the first sample, so a target the
        swarm starts on top of is still attributed a detection latency.
        """
        self._crop_cells = frozenset(
            cell for cell, state in world.cells.items() if state.crop is not None
        )
        for cell in self._crop_cells:
            if world.cells[cell].infestation > self.config.target_threshold:
                self._infested_at[cell] = now

    def observe(self, observations: Iterable[Observation], now: Time) -> None:
        """Record what a robot sensed at ``now``.

        Sensing a cell for the first time settles both the coverage figure and, for a cell
        that was already infested, its detection latency.
        """
        for observation in observations:
            cell = observation.cell
            if cell in self._observed:
                continue
            self._observed.add(cell)
            if cell not in self._crop_cells:
                continue
            infested_at = self._infested_at.get(cell)
            if infested_at is not None and cell not in self._detected_at:
                self._detected_at[cell] = now

    def sample(self, world: WorldState, now: Time, messages_sent: int) -> MetricsSample:
        """Read the world at ``now``, record one row of the series, and return it."""
        world.advance_to(now)
        self._track_infestation(world, now)

        taken = MetricsSample(
            time=now,
            targets_remaining=self._targets_remaining(world),
            detection_latency_ms=self._mean_detection_latency(),
            energy_per_area=self._energy_per_area(world),
            messages_sent=messages_sent,
            coverage=self._coverage(),
            active_robots=len(world.live_robots()),
        )
        self.samples.append(taken)
        return taken

    def series(self) -> tuple[MetricsSample, ...]:
        """Every sample taken so far, oldest first."""
        return tuple(self.samples)

    def latest(self) -> MetricsSample | None:
        """The most recent sample, or ``None`` before the first one is taken."""
        return self.samples[-1] if self.samples else None

    def _track_infestation(self, world: WorldState, now: Time) -> None:
        """Remember when each crop cell first became an active target."""
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
