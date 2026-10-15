"""The per-robot exchange between the simulation (B) and the algorithm (A)."""

from dataclasses import dataclass, field
from typing import Protocol

from swarm_field.contracts.actions import Action
from swarm_field.contracts.ids import BodyType, CellId, RobotId, Time
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
        consumable: Remaining treatment consumable, 0..1.
        health: Condition, 0..1.
    """

    robot_id: RobotId
    position: CellId
    battery: float
    body: BodyType
    speed_mps: float
    consumable: float
    health: float


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


@dataclass
class AgentInspection:
    """Read-only algorithm state for the visualization layer."""

    robot_id: RobotId
    current_goal: str | None
    belief: object


class SwarmAgent(Protocol):
    """Implemented by the algorithm (A), called by the simulation (B)."""

    def decide(self, perception: Perception) -> Decision: ...

    def inspect(self) -> AgentInspection: ...
