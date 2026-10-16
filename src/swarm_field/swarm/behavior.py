"""Composable robot behaviours for the high-level cell model."""

import logging
import math
from dataclasses import dataclass, field
from typing import Protocol

from swarm_field.contracts import (
    BodyType,
    CellId,
    Charge,
    Idle,
    Message,
    Observation,
    RobotSelfState,
    Time,
    TimeUnit,
    Treat,
)
from swarm_field.swarm.memory import MemoryView
from swarm_field.swarm.provider import FieldProvider
from swarm_field.swarm.seeder import RegroupSeeder, TreatSeeder

logger = logging.getLogger(__name__)


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
        target: The cell this candidate commits the robot to, such as the field's target at the
            robot's cell, or ``None``. The arbiter uses it to tell a new target from the one
            the robot is already heading for.
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
        fields: Hands out the grid potential fields of the registered seeders by name, through
            ``field_for``. Behaviours read fields; they never move the robot themselves.
        inbox: The messages received at this decision point.
    """

    time: Time
    self_state: RobotSelfState
    sensed: list[Observation]
    memory: MemoryView
    fields: FieldProvider
    inbox: list[Message] = field(default_factory=list)


class Behavior(Protocol):
    """A single composable piece of decision logic, bidding for what the robot does next.

    At each decision point a behaviour reads the context and returns its candidates, or an
    empty tuple when it has nothing to propose. It decides whether it wants the robot, how much
    (``score``), and what to do if it wins (``action``). It may read the fields of seeders
    through ``ctx.fields``, e.g. to score from the potential at the robot's cell.

    A behaviour never moves the robot itself. To move, it proposes
    ``FollowField`` with a seeder's name, and the navigator steps along that
    field only if the candidate wins. A behaviour standing on a target proposes
    its on-the-spot action instead, such as ``Treat`` or ``Charge``.
    """

    name: str

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]: ...


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
        seeded = ctx.fields.field_for(self._seeder, position, ctx.time)
        if seeded is None:
            return ()
        for seed in seeded.seeds:
            if seed.cell == position:
                return (ActionCandidate(Treat(position), _TREAT_SCORE, self.name, position),)

        num_killers = 0
        for peer in ctx.memory.peers():
            if peer.body is BodyType.KILLER:
                num_killers += 1
        # The more pests are left per killer already known, the more this robot is needed.
        score = 1 / (1 + math.exp(-(len(seeded.seeds) - num_killers)))
        return (ActionCandidate(FollowField(self._seeder), score, self.name, None),)


_REGROUP_SCORE = 0.98
"""The score of walking back to the swarm after a long silence.

Just below ``_TREAT_SCORE``, so that the robot still treats the pest it stands on first.
"""


class Regroup:
    """Returns a robot to the swarm once it has received no message for ``silence_timeout``.

    It follows the field of its ``RegroupSeeder`` to where the peers were last known to be.
    Standing on such a cell and still hearing no one, it tells the seeder the cell was searched
    and moves on to the next; the seeder's rally point is where it finally waits. The first
    message received ends the search.
    """

    name: str = "regroup"

    def __init__(self, seeder: RegroupSeeder, silence_timeout: Time = 2 * TimeUnit.MINUTE) -> None:
        self._seeder: RegroupSeeder = seeder
        self._silence_timeout: Time = silence_timeout
        self._last_contact: Time | None = None
        self._warned_nowhere_to_go: bool = False

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        if ctx.inbox or self._last_contact is None:
            self._last_contact = ctx.time
            self._warned_nowhere_to_go = False
            self._seeder.forget_searched()
            return ()
        if ctx.time - self._last_contact < self._silence_timeout:
            return ()

        position = ctx.self_state.position
        seeded = ctx.fields.field_for(self._seeder.name, position, ctx.time)
        if seeded is not None and position != self._seeder.rally_point:
            for seed in seeded.seeds:
                if seed.cell == position:
                    self._seeder.mark_searched(position)
                    seeded = ctx.fields.field_for(self._seeder.name, position, ctx.time)
                    break
        if seeded is None:
            if not self._warned_nowhere_to_go:
                logger.warning(
                    "Robot %s heard no one since %s and knows no place to look for peers",
                    ctx.self_state.robot_id,
                    self._last_contact,
                )
                self._warned_nowhere_to_go = True
            return ()
        #target = seeded.field.target_for(position)
        return (ActionCandidate(FollowField(self._seeder.name), _REGROUP_SCORE, self.name, None),)
