"""Configurable battery-return thresholds used by the existing behaviour/arbiter path."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RechargeConfig:
    """Thresholds and bids for returning to a charging dock.

    Scores use the existing UtilityArbiter scale. The low-priority bid is just above the
    arbiter's default override threshold; the critical bid is the strongest possible bid.
    ``low_battery_threshold`` initiates a dock return, while ``critical_battery_threshold``
    ensures the robot abandons normal work before it is stranded.
    """

    low_battery_threshold: float = 0.25
    critical_battery_threshold: float = 0.10
    low_priority: float = 0.91
    critical_priority: float = 1.0
    no_spares_priority: float = 0.95
    dock_priority: float = 1.0

    def __post_init__(self) -> None:
        """Validate thresholds and scores against the shared [0, 1] scale."""
        if not 0.0 < self.critical_battery_threshold < self.low_battery_threshold <= 1.0:
            raise ValueError("battery thresholds must satisfy 0 < critical < low <= 1")
        for name, score in (
            ("low_priority", self.low_priority),
            ("critical_priority", self.critical_priority),
            ("no_spares_priority", self.no_spares_priority),
            ("dock_priority", self.dock_priority),
        ):
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.critical_priority < self.low_priority:
            raise ValueError("critical_priority must be at least low_priority")
