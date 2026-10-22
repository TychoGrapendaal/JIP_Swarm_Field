"""Observations — a tagged union of beliefs about cells.

An observation is a *belief*, not a truth. Variants share ``ObservationBase`` (which cell,
how reliable, when, by whom) and are matched with ``match``/``case``, like ``Action``.
"""

from dataclasses import dataclass

from swarm_field.contracts.ids import (
    AgentObjective,
    BodyType,
    CellId,
    CropType,
    PestType,
    RobotId,
    Terrain,
    Time,
)


@dataclass(frozen=True)
class ObservationBase:
    """Fields common to every observation.

    Attributes:
        cell: The cell the observation is about.
        confidence: How reliable this reading is, 0..1.
        observed_at: Time the observation was made (drives decay / staleness).
        source: The robot that made it.
    """

    cell: CellId
    confidence: float
    observed_at: Time
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
class TerrainObservation(ObservationBase):
    """A reading of what a cell is made of: free, an obstacle, or outside the field.

    Attributes:
        terrain: The observed terrain.
    """

    terrain: Terrain


@dataclass(frozen=True)
class PeerObservation(ObservationBase):
    """A sighting of another robot at ``cell``, or a robot's report of itself.

    Attributes:
        peer: The peer's identity if resolvable, else ``None`` for an anonymous sighting.
        body: The peer's body type.
        battery: Reported battery fraction in [0, 1]. An unavailable report uses 1.0 conservatively.
        target: The cell the peer is heading for, if it reported one. A status report can include
            it; a local sighting may also include the current move target when exposed by the world.
        objective: The objective the peer is pursuing, if it reported one, so that a seeder
            can tell a peer treating a cell from one exploring it.
    """

    peer: RobotId | None
    body: BodyType
    target: CellId | None = None
    objective: AgentObjective | None = None
    battery: float = 1.0


Observation = CropObservation | InfestationObservation | TerrainObservation | PeerObservation
