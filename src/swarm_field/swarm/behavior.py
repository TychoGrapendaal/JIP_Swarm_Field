"""Composable robot behaviours for the high-level cell model."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol

from swarm_field.contracts import (
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
from swarm_field.swarm.memory import MemoryView
from swarm_field.swarm.provider import FieldProvider
from swarm_field.swarm.seeder import SeededField, TreatSeeder
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

    A behaviour never moves the robot itself. To move, it proposes ``FollowField`` with a
    seeder's name, and the navigator steps along that field only if the candidate wins. A
    behaviour standing on its target (``field.target_for(position) == position``) proposes its
    on-the-spot action instead, such as ``Treat`` or ``Charge``.
    """

    name: str

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]: ...


class TreatInfestation:
    """Skeleton of the pest-response behaviour: follow the field of the TREAT seeder."""

    name: str = "treat-infestation"

    def __init__(self, seeder: str = TreatSeeder.name) -> None:
        self._seeder: str = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        position = ctx.self_state.position
        seeded = ctx.fields.field_for(self._seeder, position, ctx.time)
        if seeded is None:
            return ()
        target = seeded.field.target_for(position)
        if target == position:
            # TODO (#40): standing on a pest: propose Treat(cell=position).
            return ()
        # TODO (#39): a score on the scale shared by all behaviours. Lower potential is better,
        # so negate it for now.
        score = -seeded.field.value_at(position)
        candidate = ActionCandidate(
            action=FollowField(self._seeder), score=score, source=self.name, target=target
        )
        return (candidate,)


class ExploreBehaviour:
    """Score exploration using peer composition and the available exploration field.

    The behaviour combines its criteria with persistent weights, follows the configured
    seeder's field when a target is available, and proposes ``Idle`` otherwise.
    """

    name: str = "explore"

    def __init__(self, seeder: str) -> None:
        # TODO: include the seeder name once the EXPLORE seeder is implemented (#41).
        self._seeder: str = seeder
        self._weights: list[float] = [1 / 3, 1 / 3, 1 / 3]
        self._previous_time: Time | None = None

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        """Build this robot's exploration candidate for the current decision.

        The score combines the nearby non-scout fraction, average finite field score, and
        explored-cell fraction. If no usable field target exists, the scored candidate proposes
        ``Idle`` instead of following a field.
        """
        position = ctx.self_state.position
        inbox = ctx.inbox
        seeded_field = ctx.fields.field_for(self._seeder, position, ctx.time)
        target = None
        if seeded_field is not None:
            target = seeded_field.field.target_for(position)
        criteria_scores = [
            self._body_type_criteria(inbox),
            self._explore_score_criteria(seeded_field),
            self._unexplore_criteria(seeded_field),
        ]
        previous_time = self._previous_time
        if previous_time is None:
            previous_time = ctx.time
        weights = self._update_weight(ctx, self._weights, previous_time)
        self._weights = weights
        self._previous_time = ctx.time
        score = sum(w * s for w, s in zip(weights, criteria_scores, strict=True))
        action = FollowField(self._seeder)
        if seeded_field is None or target is None or target == position:
            action = Idle()
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

    def _explore_score_criteria(self, seeded_field: SeededField | None) -> float:
        """Return the average finite score in the provided exploration field."""
        if seeded_field is None:
            return 1.0
        scores = [
            result.value for result in seeded_field.field.settled.values() if isfinite(result.value)
        ]
        if not scores:
            return 1.0
        return sum(scores) / len(scores)

    def _unexplore_criteria(self, seeded_field: SeededField | None) -> float:
        """Return the 1 - (fraction of unexplored cells in the provided exploration field)."""
        if seeded_field is None:
            return 1.0
        total_cells = len(seeded_field.field.settled)
        if total_cells == 0:
            return 1.0
        unexplored_cells = sum(
            1 for result in seeded_field.field.settled.values() if result.value == 1.0
        )
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
