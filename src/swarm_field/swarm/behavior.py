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
    ReplaceBattery,
    RobotSelfState,
    Time,
    Treat,
)
from swarm_field.swarm.capabilities import CapabilitySet, capabilities_for
from swarm_field.swarm.explore import ExploreConfig, ExploreSeeder, revisit_worths
from swarm_field.swarm.memory import MemoryView
from swarm_field.swarm.seeder import BatteryRequestSeeder, TreatSeeder


@dataclass(frozen=True)
class FollowField:
    """Take one step along the grid potential field of a registered seeder.

    Attributes:
        seeder: The name of the seeder whose field to follow.
    """

    seeder: str


Intent = FollowField | Treat | Charge | ReplaceBattery | Idle
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
        capabilities: The body's allowed actions and physical capability profile, if provided.
    """

    time: Time
    self_state: RobotSelfState
    sensed: list[Observation]
    memory: MemoryView
    inbox: list[Message] = field(default_factory=list)
    capabilities: CapabilitySet | None = None


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
    """The pest-response behaviour: follow the TREAT field and treat on arrival."""

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
    """Score exploration from remembered terrain and the composition of nearby peers.

    The behaviour proposes the configured exploration field as an intent. It never reads the
    field itself; the brain builds it only after this candidate wins arbitration.
    """

    name: str = "explore"

    def __init__(self, seeder: str = ExploreSeeder.name) -> None:
        self._seeder: str = seeder
        self._weights: list[float] = [1 / 3, 1 / 3, 1 / 3]
        self._previous_time: Time | None = None

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        """Build an exploration candidate scored from memory, without inspecting a field.

        The score combines the nearby non-scout fraction, revisit worth across monitorable cells,
        and the fraction of cells already observed. The field provider resolves the target only
        if this candidate wins arbitration.
        """
        memory = ctx.memory

        criteria_scores = [
            self._body_type_criteria(memory),
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
        candidate = ActionCandidate(
            action=action,
            score=score,
            source=self.name,
            target=None,
            objective=AgentObjective.EXPLORE,
        )
        return (candidate,)

    def _body_type_criteria(self, memory: MemoryView) -> float:
        """Return the 1 - (fraction of nearby robots that are scouts)."""
        num_explore = 0
        num_peers = 0
        for peer in memory.peers():
            num_peers += 1
            if peer.body is BodyType.SCOUT:
                num_explore += 1
        if num_explore == 0:
            return 0.99
        return 1 - (num_explore / num_peers)

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
        avg_worth = min(float(np.mean(worths)), 0.99)
        return avg_worth

    def _unexplore_criteria(self, memory: MemoryView) -> float:
        """Return the 1 - (fraction of unexplored cells in the provided exploration field)."""
        last_seen = memory.last_observed_times()
        if last_seen is None:
            return 0.99
        total_cells = last_seen.size
        if total_cells == 0:
            return 0.99
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


class FollowSeederField:
    """A mock behaviour: always bid 0.5 to follow one seeder's field, for the objective EXPLORE.

    As the only behaviour of a robot it always wins. Next to other behaviours it is a mock
    explore behaviour with an even bid on the arbiter's scale.
    """

    name: str = "follow-field"

    def __init__(self, seeder: str) -> None:
        self._seeder: str = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        candidate = ActionCandidate(
            FollowField(self._seeder), 0.5, self.name, target=None, objective=AgentObjective.EXPLORE
        )
        return (candidate,)


class ExploreField(ExploreBehaviour):
    """Bid for the EXPLORE field using memory-based exploration scoring.

    The legacy class name is retained for configuration and registry compatibility. Target
    selection remains the responsibility of the winning field and navigator.
    """

    name: str = "explore-field"

    def __init__(self, seeder: str = ExploreSeeder.name) -> None:
        super().__init__(seeder)


class SupplyBattery:
    """Visit recent low-battery peer sightings and replace a battery when co-located.

    Eligibility and urgency are calculated from memory alone. The battery-support field chooses
    which eligible peer to visit only after this behaviour's movement intent wins arbitration.
    """

    name: str = "supply-battery"
    low_battery_threshold: float = BatteryRequestSeeder.low_battery_threshold
    observation_ttl: Time = BatteryRequestSeeder.observation_ttl
    max_travel_score: float = 0.89

    def __init__(self, seeder: str = BatteryRequestSeeder.name) -> None:
        self._seeder = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        if ctx.self_state.spare_batteries <= 0:
            return ()

        eligible = [
            peer
            for peer in ctx.memory.peers()
            if peer.peer is not None
            and peer.battery is not None
            and peer.body in (BodyType.SCOUT, BodyType.KILLER)
            and peer.battery <= self.low_battery_threshold
            and 0 <= ctx.time - peer.observed_at <= self.observation_ttl
            and ctx.memory.confidence(peer, ctx.time) >= 0.5
        ]
        if not eligible:
            return ()

        position = ctx.self_state.position
        eligible_here = [peer for peer in eligible if peer.cell == position]
        if eligible_here:
            recipient = min(
                eligible_here,
                key=lambda peer: (peer.battery or 0.0, peer.peer or -1),
            )
            if recipient.peer is None:
                return ()
            return (
                ActionCandidate(
                    action=ReplaceBattery(robot_id=recipient.peer),
                    score=self.max_travel_score,
                    source=self.name,
                    target=position,
                ),
            )

        lowest_battery = min(peer.battery or 0.0 for peer in eligible)
        urgency = 1.0 - lowest_battery / self.low_battery_threshold
        score = 0.5 + (self.max_travel_score - 0.5) * max(0.0, min(1.0, urgency))
        return (
            ActionCandidate(
                action=FollowField(self._seeder),
                score=score,
                source=self.name,
                target=None,
            ),
        )


class ReturnToDock:
    """Return to a known dock when energy is low or a battery robot is out of spares.

    Dock membership is checked from the configured dock cells. The dock field is not queried by
    this behaviour; it is built only when the follow-field intent wins arbitration.
    """

    name: str = "return-to-dock"

    def __init__(self, seeder: str = "dock", *, docks: tuple[CellId, ...] = ()) -> None:
        self._seeder = seeder
        self._docks = frozenset(docks)

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        profile = ctx.capabilities
        if profile is None:
            profile = capabilities_for(ctx.self_state.body)

        position = ctx.self_state.position
        at_dock = position in self._docks
        if at_dock and (
            ctx.self_state.battery < 1.0
            or (
                ctx.self_state.body is BodyType.BATTERY
                and ctx.self_state.spare_batteries < profile.spare_battery_capacity
            )
        ):
            return (
                ActionCandidate(
                    action=Charge(),
                    score=0.99,
                    source=self.name,
                    target=position,
                    objective=AgentObjective.RECHARGE,
                ),
            )

        if not self._docks:
            return ()

        if ctx.self_state.body is BodyType.BATTERY:
            needs_dock = ctx.self_state.battery <= 0.25 or ctx.self_state.spare_batteries <= 0
        else:
            # A scout/killer stays available to the swarm while a support robot may still help;
            # it heads to the dock only when the battery is nearly exhausted.
            needs_dock = ctx.self_state.battery <= 0.10
        if not needs_dock:
            return ()

        return (
            ActionCandidate(
                action=FollowField(self._seeder),
                score=0.95,
                source=self.name,
                target=None,
                objective=AgentObjective.RECHARGE,
            ),
        )
