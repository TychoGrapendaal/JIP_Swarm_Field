"""Configurable battery-return thresholds used by the existing behaviour/arbiter path."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RechargeConfig:
    """Thresholds and bids for returning to a charging dock.

    Scores use the existing UtilityArbiter scale. The low-priority bid is just above the
    arbiter's default override threshold; the critical bid is the strongest possible bid.
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


@dataclass(frozen=True)
class ResupplyConfig:
    """When a killer returns to a dock to restock its doses, and how eagerly.

    The threshold is a number of doses, not a share of the tank, so that it means the same with
    any tank size: a share of 0.2 would be 200 doses of a tank of 1000.

    Attributes:
        low_doses: At or below this many doses left the killer bids to restock. 2, the default,
            lets a seeded killer restock before it runs dry; 0 restocks only an empty tank.
        low_priority: The bid at ``low_doses``: below a seeded treat bid (0.8 to 0.89), so a
            seeded killer with doses left treats the pests it knows of before it restocks.
        empty_priority: The bid with no dose left: above the arbiter's override threshold (0.9),
            since an empty killer can do nothing else of use.
        dock_priority: The bid to restock on a dock, which the dock serves at once.
    """

    low_doses: int = 2
    low_priority: float = 0.6
    empty_priority: float = 0.95
    dock_priority: float = 1.0

    def __post_init__(self) -> None:
        """Validate the threshold and the bids against the shared [0, 1] scale."""
        if self.low_doses < 0:
            raise ValueError("low_doses must not be negative")
        if not 0.0 <= self.low_priority <= self.empty_priority <= 1.0:
            raise ValueError("bids must satisfy 0 <= low_priority <= empty_priority <= 1")
        if not 0.0 <= self.dock_priority <= 1.0:
            raise ValueError("dock_priority must be between 0 and 1")
