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
    """When a killer returns to a dock to restock its treatment consumable, and how eagerly.

    Attributes:
        low_consumable: At or below this share of a full consumable the killer bids to restock.
            0.2 is two doses of the default ten, so a killer restocks before it runs dry.
        low_priority: The bid at ``low_consumable``: below a treat bid (0.8 to 0.89), so a killer
            with doses left treats the pests it knows of before it restocks.
        empty_priority: The bid with nothing left: above the arbiter's override threshold (0.9),
            since an empty killer can do nothing else of use.
        dock_priority: The bid to restock on a dock, which the dock serves at once.
    """

    low_consumable: float = 0.2
    low_priority: float = 0.6
    empty_priority: float = 0.95
    dock_priority: float = 1.0

    def __post_init__(self) -> None:
        """Validate the threshold and the bids against the shared [0, 1] scale."""
        if not 0.0 < self.low_consumable < 1.0:
            raise ValueError("low_consumable must be between 0 and 1")
        if not 0.0 <= self.low_priority <= self.empty_priority <= 1.0:
            raise ValueError("bids must satisfy 0 <= low_priority <= empty_priority <= 1")
        if not 0.0 <= self.dock_priority <= 1.0:
            raise ValueError("dock_priority must be between 0 and 1")
