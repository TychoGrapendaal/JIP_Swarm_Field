"""The per-robot exchange between the simulation (B) and the algorithm (A)."""

from dataclasses import dataclass, field
from typing import Protocol

from swarm_field.contracts.actions import Action
from swarm_field.contracts.ids import AgentObjective, BodyType, CellId, RobotId, Time
from swarm_field.contracts.messages import Message
from swarm_field.contracts.observations import Observation


@dataclass(frozen=True)
class RobotSelfState:
    """A robot's current self-perception.

    Attributes:
        robot_id: This robot's identifier.
        position: Perceived field cell, not necessarily ground truth in future sensor models.
        battery: Remaining energy, 0..1.
        body: Robot body type.
        speed_mps: Current maximum travel speed in metres per second.
        consumable: Remaining treatment consumable, 0..1: the share of a full tank of doses left.
        health: Condition, 0..1.
        spare_batteries: Number of charged spare batteries carried by a support robot.
        doses_left: Treatments left in the tank; 10, a full tank of the default size, unless
            given.
    """

    robot_id: RobotId
    position: CellId
    battery: float
    body: BodyType
    speed_mps: float
    consumable: float
    health: float
    spare_batteries: int = 0
    doses_left: int = 10


@dataclass(frozen=True)
class Perception:
    """Everything a robot receives at one decision point."""

    time: Time
    self_state: RobotSelfState
    sensed: list[Observation]
    inbox: list[Message]


@dataclass(frozen=True)
class Decision:
    """A robot's output for a single decision point."""

    action: Action
    outbox: list[Message] = field(default_factory=list)


@dataclass(frozen=True)
class RobotPublicState:
    """Ground-truth robot state exposed to visualization.

    Attributes:
        robot_id: Robot identifier.
        position: Current discrete field cell.
        body: Robot body type.
        speed_mps: Maximum travel speed in metres per second.
        battery: Remaining energy.
        health: Condition, 0..1.
        target: Current direct movement target, if the robot is travelling.
        current_action: The currently executing action, if any.
        failed: Whether the robot has stopped taking part in the run, so a renderer can tell a
            lost robot from a live one.
        point: Where the robot is, in cells: between two cells while it moves, so a renderer can
            draw it gliding. ``None`` places it at the centre of ``position``.
    """

    robot_id: RobotId
    position: CellId
    body: BodyType
    speed_mps: float
    battery: float
    health: float
    target: CellId | None
    current_action: Action | None
    failed: bool = False
    point: tuple[float, float] | None = None

    def drawn_at(self) -> tuple[float, float]:
        """Where to draw the robot, in cells: ``point``, else the centre of its cell."""
        if self.point is not None:
            return self.point
        return (self.position[0] + 0.5, self.position[1] + 0.5)


@dataclass
class AgentInspection:
    """Read-only algorithm state, for the visualization layer and the metrics.

    Attributes:
        robot_id: The robot inspected.
        current_goal: The name of the behaviour that won the last decision, if any.
        belief: The robot's beliefs, in a shape the algorithm chooses.
        current_target: The cell the last winning candidate committed the robot to, if any.
        current_objective: The objective the last winning candidate served, if any.
        known_pests: The cells the robot believes are infested, so the metrics can tell when
            news of a pest reached it without reading ``belief``.
    """

    robot_id: RobotId
    current_goal: str | None
    belief: object
    current_target: CellId | None = None
    current_objective: AgentObjective | None = None
    known_pests: tuple[CellId, ...] = ()


class SwarmAgent(Protocol):
    """Implemented by the algorithm (A), called by the simulation (B)."""

    def decide(self, perception: Perception) -> Decision: ...

    def inspect(self) -> AgentInspection: ...
