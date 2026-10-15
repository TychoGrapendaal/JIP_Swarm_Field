"""Composable robot behaviours for the high-level cell model."""

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from swarm_field.contracts import (
    AgentObjective,
    BodyType,
    CellId,
    Charge,
    Idle,
    Message,
    Observation,
    PeerObservation,
    RobotSelfState,
    Time,
    Treat,
)
from swarm_field.swarm.explore import ExploreConfig, ExploreSeeder, revisit_worths
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
        current_target: The target committed to by the preceding decision, if any.
    """

    time: Time
    self_state: RobotSelfState
    sensed: list[Observation]
    memory: MemoryView
    inbox: list[Message] = field(default_factory=list)
    current_target: CellId | None = None


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


class ExploreBehaviour:
    """Score exploration using peer composition and the available exploration field.

    The behaviour combines its criteria with persistent weights, follows the configured
    seeder's field when a target is available, and proposes ``Idle`` otherwise.
    """

    name: str = "explore"

    def __init__(self, seeder: str = ExploreSeeder.name) -> None:
        self._seeder: str = seeder
        self._weights: list[float] = [1 / 3, 1 / 3, 1 / 3]
        self._previous_time: Time | None = None

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        """Build this robot's exploration candidate for the current decision.

        The score combines the nearby non-scout fraction, the revisit worth of the selected
        target cell, and the unexplored-cell fraction. If no usable field target exists, the
        scored candidate proposes ``Idle`` instead of following a field.
        """
        position = ctx.self_state.position
        inbox = ctx.inbox
        memory = ctx.memory

        criteria_scores = [
            self._body_type_criteria(inbox),
            self._explore_score_criteria(memory, ctx.time),
            self._unexplore_criteria(memory),
        ]
        previous_time = self._previous_time
        if previous_time is None:
            previous_time = ctx.time
        weights = self._update_weight(ctx, self._weights, previous_time)
        self._weights = weights
        self._previous_time = ctx.time
        score = sum(w * s for w, s in zip(weights, criteria_scores, strict=True))
        action = FollowField(self._seeder)
        target = None
        if ctx.current_target == position:
            action = Idle()
            target = position
        candidate = ActionCandidate(action=action, score=score, source=self.name, target=target)
        return (candidate,)

    def _body_type_criteria(self, inbox: list[Message]) -> float:
        """Return the 1 - (fraction of nearby robots that are scouts)."""
        peer_observations = [
            message.payload for message in inbox if isinstance(message.payload, PeerObservation)
        ]
        if not peer_observations:
            return 1.0
        matching_robots = sum(
            observation.body is BodyType.SCOUT for observation in peer_observations
        )
        return 1 - (matching_robots / len(peer_observations))

    def _explore_score_criteria(
        self,
        memory: MemoryView,
        time: Time,
    ) -> float:
        """Return the revisit worth of the chosen exploration target."""
        revisit_age = ExploreConfig.revisit_age
        last_seen = np.asarray([memory.last_observed_times()], dtype=np.float64)
        monitorable = np.asarray([memory.monitorable_cells()], dtype=np.bool_)
        worths = revisit_worths(last_seen, monitorable, time, revisit_age)
        avg_worth = np.mean(worths)
        return avg_worth

    def _unexplore_criteria(self, memory: MemoryView) -> float:
        """Return the 1 - (fraction of unexplored cells in the provided exploration field)."""
        last_seen = memory.last_observed_times()
        if last_seen is None:
            return 1.0
        total_cells = last_seen.size
        if total_cells == 0:
            return 1.0
        unexplored_cells = np.count_nonzero(np.isneginf(last_seen))
        return 1 - (unexplored_cells / total_cells)

    def _update_weight(
        self,
        ctx: DecisionContext,
        old_weights: list[float],
        previous_time: Time,
        alpha: float = 0.2,
    ) -> list[float]:
        """Blend prior weights toward equal criterion weights after time advances.

        Args:
            ctx: The current decision context, including the current timestamp.
            old_weights: The weights retained from the previous decision.
            previous_time: The timestamp saved from the previous weight update.
            alpha: The fraction of the way to move toward equal weights.

        Returns:
            Updated weights. If the timestamp has not advanced, the old weights are retained.
        """
        if ctx.time <= previous_time:
            return old_weights
        importance = [
            1.0,  # Weight for body type criteria
            1.0,  # Weight for explore score criteria
            1.0,  # Weight for unexplored criteria
        ]
        total_weight = sum(importance)
        normalized_weights = [w / total_weight for w in importance]
        weights = [
            alpha * nw + (1 - alpha) * ow
            for nw, ow in zip(normalized_weights, old_weights, strict=True)
        ]
        return weights
