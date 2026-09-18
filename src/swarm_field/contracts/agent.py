"""The per-robot exchange between the simulation (B) and the algorithm (A).

``SwarmAgent`` is the single critical boundary: the simulation calls ``decide`` each tick
and ``inspect`` for visualization, and never reaches into an algorithm's internals.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol

from swarm_field.contracts.actions import Action
from swarm_field.contracts.ids import CellId, Direction, RobotId, Time
from swarm_field.contracts.messages import Message
from swarm_field.contracts.observations import Observation


class BodyType(Enum):
    """The kind of robot body, which determines its capabilities.

    Members:
        SCOUT: Light and fast; senses the field.
        KILLER: Treats pests.
        BATTERY: Carries energy to other robots.
    """

    SCOUT = auto()
    KILLER = auto()
    BATTERY = auto()


@dataclass(frozen=True)
class RobotSelfState:
    """A robot's view of itself, handed to it each tick inside :class:`Perception`.

    Attributes:
        robot_id: This robot's identifier.
        position: PERCEIVED position (subject to error), not ground truth.
        facing: Current heading.
        battery: Remaining energy.
        body: The robot's body type.
        consumable: Level of the body's consumable (e.g. a spray tank), 0..1.
        health: Condition, 0..1 (1 = healthy, lower = degraded).
    """

    robot_id: RobotId
    position: CellId
    facing: Direction
    battery: float
    body: BodyType
    consumable: float
    health: float


@dataclass(frozen=True)
class Perception:
    """Everything a robot is given to decide at one decision point.

    Attributes:
        time: The current simulation time.
        self_state: The robot's view of itself.
        sensed: What the robot's sensors picked up at this decision point.
        inbox: Messages that reached the robot (already range-filtered by the channel).
    """

    time: Time
    self_state: RobotSelfState
    sensed: list[Observation]
    inbox: list[Message]


@dataclass(frozen=True)
class Decision:
    """A robot's output for a single tick.

    Attributes:
        action: The one physical action to take this tick.
        outbox: Messages to broadcast this tick, if any.
    """

    action: Action
    outbox: list[Message] = field(default_factory=list)


@dataclass(frozen=True)
class RobotPublicState:
    """Ground-truth per-robot state the simulation exposes to visualization for rendering.

    This is the god's-eye view for the frontend. It is NOT broadcast on the radio and NOT
    what other robots perceive of this one (that is a :class:`Observation` /
    :class:`Message`). It exists only so the visualization can draw the true world.

    Attributes:
        robot_id: This robot's identifier.
        position: GROUND-TRUTH position.
        facing: Current heading.
        body: The robot's body type.
        battery: Remaining energy.
        health: Condition, 0..1.
        current_action: The action being taken this tick, if any.
    """

    robot_id: RobotId
    position: CellId
    facing: Direction
    body: BodyType
    battery: float
    health: float
    current_action: Action | None


@dataclass
class AgentInspection:
    """What the algorithm exposes read-only for the "click a robot" feature.

    Attributes:
        robot_id: The inspected robot's identifier.
        current_goal: A short human-readable description of what it is doing, if any.
        belief: Algorithm-owned memory snapshot; opaque to visualization except for display.
    """

    robot_id: RobotId
    current_goal: str | None
    belief: object


class SwarmAgent(Protocol):
    """Implemented by the algorithm (A), called by the simulation (B)."""

    def decide(self, perception: Perception) -> Decision: ...

    def inspect(self) -> AgentInspection: ...
