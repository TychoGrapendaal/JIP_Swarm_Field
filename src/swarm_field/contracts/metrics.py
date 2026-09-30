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
            becoming infested and a robot first sensing it. ``None`` while nothing has been
            detected, and quantised to the sample period.
        energy_per_area: Battery fraction the swarm has spent per square metre of field.
        messages_sent: Messages handed to the channel since the run began.
        coverage: Fraction of the crop cells that at least one robot has sensed, 0..1.
    """

    time: Time
    targets_remaining: int
    detection_latency_ms: float | None
    energy_per_area: float
    messages_sent: int
    coverage: float

    def to_row(self) -> dict[str, Any]:
        """This sample as a flat ``dict`` of field name to value, for CSV and JSON export."""
        return asdict(self)
