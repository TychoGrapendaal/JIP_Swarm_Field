"""Composable robot behaviours for the high-level cell model."""

from dataclasses import dataclass, field
from typing import Protocol

from swarm_field.contracts import (
    CellId,
    Charge,
    Idle,
    Message,
    Observation,
    RobotSelfState,
    Time,
    Treat,
)
from swarm_field.swarm.memory import MemoryView
from swarm_field.swarm.seeder import TreatSeeder


@dataclass(frozen=True)
class FollowField:
    """Take one step along the grid potential field of a registered seeder.

    Attributes:
        seeder: The name of the seeder whose field to follow.
    """

    seeder: str


Intent = FollowField | Treat | Charge | Idle
"""What a behaviour wants the robot to do if its candidate wins.

The brain's own counterpart of a contract ``Action``; it never crosses into the simulation.
``Treat``, ``Charge`` and ``Idle`` pass through unchanged. ``Move`` is deliberately absent: a
behaviour names the field to follow with ``FollowField``, and only the navigator decides the
step, for the winning candidate.
"""


@dataclass(frozen=True)
class ActionCandidate:
    """A behaviour's bid at a decision point: what it wants done, and how much.

    Attributes:
        action: What the robot should do if this candidate wins.
        score: How much the behaviour wants it; the highest score wins. The scale shared by all
            behaviours is settled together with the arbiter (#39).
        source: The name of the proposing behaviour.
        target: The cell this candidate commits the robot to, if the behaviour knows it from
            memory (such as a known pest's cell), or ``None``. The arbiter uses it to tell a new
            target from the one the robot is already heading for. For a winning ``FollowField``
            the brain reports the field's target instead.
    """

    action: Intent
    score: float
    source: str
    target: CellId | None


@dataclass(frozen=True)
class DecisionContext:
    """Read-only view handed to every behaviour at a decision point.

    Attributes:
        time: The time of the decision.
        self_state: The robot's perception of itself.
        sensed: What the robot sensed at this decision point (already merged into memory).
        memory: The robot's facts about the field and its peers.
        inbox: The messages received at this decision point.
    """

    time: Time
    self_state: RobotSelfState
    sensed: list[Observation]
    memory: MemoryView
    inbox: list[Message] = field(default_factory=list)


class Behavior(Protocol):
    """A single composable piece of decision logic, bidding for what the robot does next.

    At each decision point a behaviour reads the context and returns its candidates, or an
    empty tuple when it has nothing to propose. It decides whether it wants the robot, how much
    (``score``), and what to do if it wins (``action``), from the robot's own state and its
    memory alone. It never reads a potential field: building one is expensive, so only the
    winner's field is built.

    A behaviour never moves the robot itself. To move, it proposes ``FollowField`` with a
    seeder's name; only if the candidate wins does the brain build that field, which picks the
    target, and the navigator step along it. A behaviour whose memory says the robot stands on
    its target, such as a known pest at the robot's cell, proposes its on-the-spot action
    instead, such as ``Treat`` or ``Charge``.
    """

    name: str

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]: ...


class TreatInfestation:
    """Skeleton of the pest-response behaviour: follow the field of the TREAT seeder."""

    name: str = "treat-infestation"

    def __init__(self, seeder: str = TreatSeeder.name) -> None:
        self._seeder: str = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        # TODO (#40): bid only while memory knows a pest, and propose Treat(cell=position) when
        # one is known at the robot's cell.
        # TODO (#39): a score from the robot's state and memory, on the scale shared by all
        # behaviours.
        candidate = ActionCandidate(
            action=FollowField(self._seeder), score=0.5, source=self.name, target=None
        )
        return (candidate,)
