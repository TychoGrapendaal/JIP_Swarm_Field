"""Observations — a tagged union of beliefs about cells.

An observation is a *belief*, not a truth. Variants share ``ObservationBase`` (which cell,
how reliable, when, by whom) and are matched with ``match``/``case``, like ``Action``.
"""

from dataclasses import dataclass

from swarm_field.contracts.ids import CellId, CropType, PestType, RobotId, Tick


@dataclass(frozen=True)
class ObservationBase:
    """Fields common to every observation.

    Attributes:
        cell: The cell the observation is about.
        confidence: How reliable this reading is, 0..1.
        observed_at: Tick the observation was made (drives decay / staleness).
        source: The robot that made it.
    """

    cell: CellId
    confidence: float
    observed_at: Tick
    source: RobotId


@dataclass(frozen=True)
class CropObservation(ObservationBase):
    """A reading of what crop is in a cell and how mature it is.

    Attributes:
        crop: Estimated crop species, or ``None`` if none / unresolved.
        growth_stage: Estimated maturity, 0..1.
    """

    crop: CropType | None
    growth_stage: float


@dataclass(frozen=True)
class InfestationObservation(ObservationBase):
    """A reading of pest presence in a cell.

    Attributes:
        severity: ESTIMATED severity, 0..1 (not the true value).
        pest: Estimated pest type, or ``None`` when unresolved at low confidence.
    """

    severity: float
    pest: PestType | None


@dataclass(frozen=True)
class ObstacleObservation(ObservationBase):
    """A reading of whether a cell blocks movement.

    Attributes:
        is_obstacle: Whether the cell was seen to be an obstacle.
    """

    is_obstacle: bool


@dataclass(frozen=True)
class PeerObservation(ObservationBase):
    """A sighting of another robot at ``cell``.

    Attributes:
        peer: The peer's identity if resolvable, else ``None`` for an anonymous sighting.
    """

    peer: RobotId | None


Observation = CropObservation | InfestationObservation | ObstacleObservation | PeerObservation
