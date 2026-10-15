"""Composable robot behaviours for the high-level cell model."""

import math
from dataclasses import dataclass, field
from typing import Protocol

from swarm_field.contracts import (
    AgentObjective,
    BodyType,
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
        score: How much the behaviour wants it, on the ``[0, 1]`` scale shared by all
            behaviours; the arbiter picks the winner.
        source: The name of the proposing behaviour.
        target: The cell this candidate commits the robot to, if the behaviour knows it from
            memory (such as a known pest's cell), or ``None``. The arbiter uses it to tell a new
            target from the one the robot is already heading for. For a winning ``FollowField``
            the brain reports the field's target instead.
        objective: The objective this candidate serves, reported to peers with the target, or
            ``None`` if it serves none of them.
    """

    action: Intent
    score: float
    source: str
    target: CellId | None
    objective: AgentObjective | None = None


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


_TREAT = AgentObjective.TREAT

_TREAT_SCORE = 0.99
"""The score of treating the pest the robot stands on.

Just below 1, so that the battery behaviour can still outbid it to save the robot from dying.
"""


class TreatInfestation:
    """The pest-response behaviour: follow the field of the TREAT seeder and treat on arrival."""

    name: str = "treat-infestation"

    def __init__(self, seeder: str = TreatSeeder.name) -> None:
        self._seeder: str = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        position = ctx.self_state.position
        here = ctx.memory.infestation_at(position)
        if here is not None and here.severity > 0:
            treat = Treat(position)
            return (ActionCandidate(treat, _TREAT_SCORE, self.name, position, _TREAT),)

        num_pests = 0
        for pest in ctx.memory.infestations():
            if pest.severity > 0:
                num_pests += 1
        if num_pests == 0:
            return ()
        num_killers = 0
        for peer in ctx.memory.peers():
            if peer.body is BodyType.KILLER:
                num_killers += 1
        # The more pests are left per killer already known, the more this robot is needed.
        score = 1 / (1 + math.exp(-(num_pests - num_killers)))
        return (ActionCandidate(FollowField(self._seeder), score, self.name, None, _TREAT),)
