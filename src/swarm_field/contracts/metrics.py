"""Metrics and benchmarking types (B.5) — how swarm performance is measured and shared.

A ``MetricsSample`` is what the simulation hands to visualization in a ``WorldSnapshot`` and
what a run writes out for offline analysis. It is pure data: the metrics module decides
nothing, it only records what the ground truth already says.
"""

from dataclasses import asdict, dataclass
from typing import Any

from swarm_field.contracts.ids import Time


@dataclass(frozen=True)
class MetricsSample:
    """Swarm performance at one instant, sampled periodically during a run.

    Every figure is derived from ground truth the simulation owns, so two runs with the same
    seed and the same configuration produce the same series.

    Attributes:
        time: The simulation time this sample was taken at.
        targets_remaining: Crop cells whose infestation is still above the target threshold.
        detection_latency_ms: Mean time, over the cells detected so far, between a cell
            becoming infested and a robot first sensing it infested. ``None`` while nothing has
            been detected. A pest is dated when it appears, not at the next sample.
        energy_per_area: Battery fraction the swarm has spent per square metre of field.
        messages_sent: Messages handed to the channel since the run began.
        messages_received: Times a message reached a robot in range since the run began.
        coverage: Fraction of the crop cells that at least one robot has sensed, 0..1.
        active_robots: Robots still taking part in the run. This drops when a robot fails, so a
            robustness sweep can read the swarm's remaining capacity off the series.
        mean_idleness_ms: The age of a crop cell, the time since a robot last sensed it,
            averaged over the cells and the samples so far (from ``MetricsConfig.measure_from``).
            For pests appearing at uniformly random times and cells it is the expected time
            until the next visit, i.e. the mean detection latency. A cell never sensed is as old
            as the run.
        worst_idleness_ms: The largest age of a crop cell in those samples.
        redundancy: The share of the robots' visits to a crop cell sensed less than
            ``MetricsConfig.fresh_age`` before, 0..1.
        pests_appeared: Pests that have appeared on crop cells so far, including those present
            at the start; a cell infested again after it was cleared counts again.
        pests_cleared: Of those, the pests cleared.
        knowledge_latency_ms: Mean time from a pest's first detection to the first killer
            knowing of it, over the pests a killer has heard of; ``None`` before any.
        response_time_ms: Mean time from a pest's first detection to its clearing, over the
            pests cleared; ``None`` before any.
        killer_idle_share: The share of the killers' decisions so far with nothing to do.
        killer_exploring_share: The share spent exploring.
        killer_driving_share: The share spent driving to a pest.
        killer_treating_share: The share spent treating.
        duplicate_trip_share: Of the killers' decisions driving to a pest, the share heading for
            a pest another killer was heading for as well.
        battery_failures: Number of robots that have failed from battery depletion by this sample.
    """

    time: Time
    targets_remaining: int
    detection_latency_ms: float | None
    energy_per_area: float
    messages_sent: int
    messages_received: int
    coverage: float
    active_robots: int
    mean_idleness_ms: float = 0.0
    worst_idleness_ms: float = 0.0
    redundancy: float = 0.0
    pests_appeared: int = 0
    pests_cleared: int = 0
    knowledge_latency_ms: float | None = None
    response_time_ms: float | None = None
    killer_idle_share: float = 0.0
    killer_exploring_share: float = 0.0
    killer_driving_share: float = 0.0
    killer_treating_share: float = 0.0
    duplicate_trip_share: float = 0.0
    battery_failures: int = 0

    def to_row(self) -> dict[str, Any]:
        """This sample as a flat ``dict`` of field name to value, for CSV and JSON export."""
        return asdict(self)
