"""Composable robot behaviours for the high-level cell model."""

import logging
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
    TimeUnit,
    Treat,
)
from swarm_field.swarm.capabilities import CapabilitySet, capabilities_for
from swarm_field.swarm.explore import ExploreConfig, ExploreSeeder, revisit_worths
from swarm_field.swarm.memory import MemoryView
from swarm_field.swarm.recharge import RechargeConfig
from swarm_field.swarm.seeder import (
    BatteryRequestSeeder,
    RegroupSeeder,
    TreatSeeder,
    infested,
    pests_left_to_robot,
    unsearched_peer_locations,
)

logger = logging.getLogger(__name__)


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

_TREAT_BID_LOW = 0.8
"""The bid for driving to the least severe pest left to the robot.

A killer's job is treating, so even a faint pest outbids exploring, whose bid stays below about
0.7 in practice, by more than the arbiter's margin for switching objectives.
"""

_TREAT_BID_HIGH = 0.89
"""The bid for driving to a pest of full severity at full confidence.

Below the arbiter's override threshold of 0.9, so treating stays subject to its hysteresis and
never outbids a robot saving its battery.
"""

TREAT_SCORE = 0.99
"""The score of treating the pest the robot stands on.

Just below 1, so that the battery behaviour can still outbid it to save the robot from dying.
"""


class TreatInfestation:
    """The pest-response behaviour: follow the TREAT field and treat on arrival.

    It bids by the worst pest it may go for, its severity times memory's confidence in it,
    between ``_TREAT_BID_LOW`` and ``_TREAT_BID_HIGH``, and not at all without one.

    With ``yield_to_treating_peer`` the killer claims pests: it may go only for the pests left to
    it (``pests_left_to_robot``), and standing on a pest that a peer with a lower id reports,
    within ``claim_fresh_for``, to be treating on that same cell, it proposes nothing until the
    pest is cleared, so two killers do not spend two doses on one pest. Without it, every known
    pest counts.

    Attributes:
        yield_to_treating_peer: Whether the killer claims pests and leaves them to its peers.
        claim_fresh_for: How old a peer's report of treating a cell may be to count.
    """

    name: str = "treat-infestation"

    def __init__(
        self,
        seeder: str = TreatSeeder.name,
        yield_to_treating_peer: bool = False,
        claim_fresh_for: Time = 30 * TimeUnit.SECOND,
    ) -> None:
        self._seeder: str = seeder
        self.yield_to_treating_peer: bool = yield_to_treating_peer
        self.claim_fresh_for: Time = claim_fresh_for

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        position = ctx.self_state.position
        here = ctx.memory.infestation_at(position)
        if here is not None and here.severity > 0:
            if self.yield_to_treating_peer and self._peer_treats(ctx, position):
                return ()
            treat = Treat(position)
            return (ActionCandidate(treat, TREAT_SCORE, self.name, position, _TREAT),)

        # With claims, a pest a closer killer is heading for is left to it, so the bid, like the
        # TREAT field, counts only the pests left to this robot, and with none it leaves the robot
        # to explore.
        pests = infested(ctx.memory)
        if self.yield_to_treating_peer:
            pests = pests_left_to_robot(ctx.memory, position, ctx.time)
        worst = 0.0
        for pest in pests:
            worst = max(worst, pest.severity * ctx.memory.confidence(pest, ctx.time))
        if worst <= 0.0:
            return ()
        score = _TREAT_BID_LOW + (_TREAT_BID_HIGH - _TREAT_BID_LOW) * min(1.0, worst)
        return (ActionCandidate(FollowField(self._seeder), score, self.name, None, _TREAT),)

    def _peer_treats(self, ctx: DecisionContext, cell: CellId) -> bool:
        """Whether a peer with a lower id freshly reports treating ``cell`` from that cell."""
        for peer in ctx.memory.peers():
            if peer.peer is None or peer.peer >= ctx.self_state.robot_id:
                continue
            if ctx.time - peer.observed_at > self.claim_fresh_for:
                continue
            if peer.cell == cell and peer.target == cell and peer.objective is _TREAT:
                return True
        return False


_REGROUP_SCORE = 0.85
"""The score of walking back to the swarm after a long silence.

Below the arbiter's override threshold, so regrouping goes through the ordinary arbitration
and never outbids an override such as ``ReturnToDock`` saving a robot whose battery runs out.
"""


class Regroup:
    """Returns a robot to the swarm once it has received no message for ``silence_timeout``.

    It follows the field of the REGROUP seeder (``RegroupSeeder``) to where the peers were last
    known to be; a location the robot has observed since without finding the peer no longer
    counts (``unsearched_peer_locations``), and the rally point is where it finally waits. The
    first message received ends the search. With no location left and no rally point it does
    not bid, since there is nowhere to go.
    """

    name: str = "regroup"

    def __init__(
        self,
        seeder: str = RegroupSeeder.name,
        silence_timeout: Time = 2 * TimeUnit.MINUTE,
        rally_point: CellId | None = None,
    ) -> None:
        self._seeder: str = seeder
        self._silence_timeout: Time = silence_timeout
        self._rally_point: CellId | None = rally_point
        self._last_contact: Time | None = None

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        if ctx.inbox or self._last_contact is None:
            self._last_contact = ctx.time
            return ()
        if ctx.time - self._last_contact < self._silence_timeout:
            return ()
        if self._rally_point is None and not unsearched_peer_locations(ctx.memory):
            return ()
        return (ActionCandidate(FollowField(self._seeder), _REGROUP_SCORE, self.name, None),)


class ExploreBehaviour:
    """Bid for the EXPLORE field, scored from the robot's memory.

    The bid is the mean of three criteria, each from 0 to 0.99:

    - the share of the peers the robot remembers that are not scouts, so a robot among few
      explorers explores more eagerly;
    - the mean revisit worth of the cells it remembers, by the field's ``revisit_age``, so a
      field gone stale draws it out;
    - the share of the field it has observed at least once.

    It never reads the field itself; the brain builds it only after this candidate wins.

    Attributes:
        config: The parameters of the EXPLORE field, whose ``revisit_age`` the bid uses.
    """

    name: str = "explore"

    def __init__(
        self, seeder: str = ExploreSeeder.name, config: ExploreConfig | None = None
    ) -> None:
        self._seeder: str = seeder
        self.config: ExploreConfig = ExploreConfig()
        if config is not None:
            self.config = config

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        memory = ctx.memory
        criteria = [
            self._non_scout_share(memory),
            self._mean_revisit_worth(memory, ctx.time),
            self._explored_share(memory),
        ]
        candidate = ActionCandidate(
            action=FollowField(self._seeder),
            score=sum(criteria) / len(criteria),
            source=self.name,
            target=None,
            objective=AgentObjective.EXPLORE,
        )
        return (candidate,)

    @staticmethod
    def _non_scout_share(memory: MemoryView) -> float:
        """The share of the remembered peers that are not scouts; 0.99 with no scout among them."""
        scouts = 0
        peers = 0
        for peer in memory.peers():
            peers += 1
            if peer.body is BodyType.SCOUT:
                scouts += 1
        if scouts == 0:
            return 0.99
        return 1 - scouts / peers

    def _mean_revisit_worth(self, memory: MemoryView, time: Time) -> float:
        """The mean revisit worth of the field's cells at ``time``, at most 0.99."""
        worths = revisit_worths(
            memory.last_observed_times(), memory.monitorable_cells(), time, self.config.revisit_age
        )
        return min(float(np.mean(worths)), 0.99)

    @staticmethod
    def _explored_share(memory: MemoryView) -> float:
        """The share of the field's cells observed at least once; 0.99 for an empty field."""
        last_seen = memory.last_observed_times()
        if last_seen.size == 0:
            return 0.99
        unexplored = np.count_nonzero(np.isneginf(last_seen))
        return 1 - unexplored / last_seen.size


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
                key=lambda peer: (peer.battery, peer.peer or -1),
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

        lowest_battery = min(peer.battery for peer in eligible)
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

    def __init__(
        self,
        seeder: str = "dock",
        *,
        docks: tuple[CellId, ...] = (),
        recharge_config: RechargeConfig | None = None,
    ) -> None:
        self._seeder = seeder
        self._docks = frozenset(docks)
        self._recharge_config = recharge_config or RechargeConfig()

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        profile = ctx.capabilities
        if profile is None:
            profile = capabilities_for(ctx.self_state.body)

        position = ctx.self_state.position
        at_dock = position in self._docks
        needs_spares = (
            ctx.self_state.body is BodyType.BATTERY
            and ctx.self_state.spare_batteries < profile.spare_battery_capacity
        )
        if at_dock and (ctx.self_state.battery < 1.0 or needs_spares):
            return (
                ActionCandidate(
                    action=Charge(),
                    score=self._recharge_config.dock_priority,
                    source=self.name,
                    target=position,
                    objective=AgentObjective.RECHARGE,
                ),
            )

        if not self._docks:
            return ()

        battery = ctx.self_state.battery
        if battery <= self._recharge_config.critical_battery_threshold:
            priority = self._recharge_config.critical_priority
        elif battery <= self._recharge_config.low_battery_threshold:
            span = (
                self._recharge_config.low_battery_threshold
                - self._recharge_config.critical_battery_threshold
            )
            urgency = (self._recharge_config.low_battery_threshold - battery) / span
            priority = self._recharge_config.low_priority + urgency * (
                self._recharge_config.critical_priority - self._recharge_config.low_priority
            )
        elif ctx.self_state.body is BodyType.BATTERY and ctx.self_state.spare_batteries <= 0:
            priority = self._recharge_config.no_spares_priority
        else:
            return ()

        return (
            ActionCandidate(
                action=FollowField(self._seeder),
                score=priority,
                source=self.name,
                target=None,
                objective=AgentObjective.RECHARGE,
            ),
        )
