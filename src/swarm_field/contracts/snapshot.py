"""Read-only view of the world handed from the simulation (B) to visualization (C)."""

from dataclasses import dataclass
from typing import Protocol

from swarm_field.contracts.agent import RobotPublicState
from swarm_field.contracts.ids import CellId, Time
from swarm_field.contracts.metrics import MetricsSample
from swarm_field.contracts.world import CellState, FieldGeometry


@dataclass
class WorldSnapshot:
    """A full, read-only picture of the world at one instant, for rendering.

    Attributes:
        time: The simulation time this snapshot was taken at.
        field: The field geometry.
        cells: Ground-truth state of every cell.
        robots: Ground-truth public state of every robot.
        metrics: The metrics sample for ``time``, or ``None`` when the run records none or no
            sample has been taken yet.
    """

    time: Time
    field: FieldGeometry
    cells: dict[CellId, CellState]
    robots: list[RobotPublicState]
    metrics: MetricsSample | None = None


class Renderer(Protocol):
    """Draws a snapshot of the world. One implementation per rendering technology.

    The port lives here rather than in ``visualization`` because the simulation offers render
    ticks to a renderer it never constructs: a run takes a ``Renderer`` the same way it takes a
    ``SwarmAgent``, through a shared boundary type.
    """

    def render(self, snapshot: WorldSnapshot) -> None: ...
