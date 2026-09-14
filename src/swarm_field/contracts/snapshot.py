"""Read-only view of the world handed from the simulation (B) to visualization (C)."""

from dataclasses import dataclass

from swarm_field.contracts.agent import RobotPublicState
from swarm_field.contracts.ids import CellId, Tick
from swarm_field.contracts.world import CellState, FieldGeometry


@dataclass
class WorldSnapshot:
    """A full, read-only picture of the world at one tick, for rendering.

    Attributes:
        tick: The tick this snapshot was taken at.
        field: The field geometry.
        cells: Ground-truth state of every cell.
        robots: Ground-truth public state of every robot.
    """

    tick: Tick
    field: FieldGeometry
    cells: dict[CellId, CellState]
    robots: list[RobotPublicState]
