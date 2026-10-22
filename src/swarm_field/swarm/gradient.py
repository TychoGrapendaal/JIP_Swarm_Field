"""Hop-gradient pest navigation: killers find a pest by hop counts alone, with no positions sent.

A robot that senses a pest on its cell becomes its anchor. A scout anchor stays on the pest
(``AnchorPest``) for up to ``GradientConfig.anchor_time``, and at every decision sends a
``HopGradient`` at hop 0 with a new round. Every robot that hears one keeps the newest round
and, within a round, the fewest hops, remembers the neighbour it heard that count from (its
parent), and passes on its own count, one more. The counts form a gradient over the swarm that
points down to the anchor, and it follows the swarm as robots move: a count is refreshed every
round and dropped when no refresh has arrived for ``max_age``, so a broken chain fades away.

A killer follows the gradient down: it drives to its parent's current cell, which it knows
because neighbours in radio range report their cells, and as it closes in it hears a robot one
hop nearer, which becomes its new parent. Once next to the anchor its parent is the anchor
itself, standing on the pest, so the killer arrives on the pest, senses it and treats it. No
message carries the pest's position, and no robot needs to know where the pest is but the
anchor. The potential field is used only to drive to the parent around obstacles.

The robot that finds the pest gone, by sensing its cell clean, sends a cleared gradient,
which wins wherever it arrives. A killer claims the pest it follows; the claim travels with
the gradients, so other killers follow other pests.
"""

import logging
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import cast, override

from swarm_field.contracts import (
    AgentObjective,
    BodyType,
    CellId,
    Decision,
    HopClaim,
    HopGradient,
    Idle,
    InfestationObservation,
    Message,
    MessageKind,
    Perception,
    PestId,
    RobotId,
    Time,
    TimeUnit,
    Treat,
)
from swarm_field.swarm.arbiter import Arbiter
from swarm_field.swarm.behavior import ActionCandidate, Behavior, DecisionContext, FollowField
from swarm_field.swarm.brain import ComposedAgent
from swarm_field.swarm.field import Seed
from swarm_field.swarm.memory import AgentMemory, MemoryView
from swarm_field.swarm.navigator import Navigator
from swarm_field.swarm.priority import ranks_before
from swarm_field.swarm.provider import FieldProvider
from swarm_field.swarm.sharing import SharingConfig
from swarm_field.swarm.tether import TetherConfig

logger = logging.getLogger(__name__)

_TREAT = AgentObjective.TREAT
_TREAT_HERE_SCORE = 0.99
"""Treating the pest underfoot, as ``TreatInfestation`` bids it."""


@dataclass(frozen=True)
class GradientConfig:
    """How hop counts spread and age, how long anchors stay, and how much killers want to follow.

    Attributes:
        max_hops: Counts beyond this many hops are not kept or passed on.
        max_age: A count not refreshed for this long is dropped, as is a claim this old. Long
            enough that a killer whose chain breaks keeps driving to where its parent was, in
            the pest's general direction, until a new chain reaches it; short enough that a
            gradient whose anchor left fades within minutes.
        cleared_lifetime: How long a robot remembers, and passes on, that a pest is cleared.
        anchor_time: How long a scout stays on a pest it found, waiting for a killer.
        claimed_anchor_time: How long it stays at most while a killer has claimed the pest,
            so a killer already on its way does not arrive to find its anchor gone.
        anchor_score: The anchor's bid to stay. Above exploring's bids plus the arbiter's
            objective floor, so a scout stays on the pest it finds, but below the override
            threshold, so a need that must not be traded off, such as a battery about to run
            out, still takes the robot away.
        follow_score: The bid to follow a gradient at hop 1. Above exploring's bids plus the
            arbiter's objective floor, so a killer leaves exploring when it hears of a pest.
        score_per_hop: How much less the bid is per hop, so a far pest matters less.
        min_follow_score: The bid never drops below this while a count is live.
        pests_in_knowledge: Whether pests in peers' memory snapshots are believed too. Off,
            hop counts are the only way a robot learns of a pest it has not sensed itself.
    """

    max_hops: int = 20
    max_age: Time = 2 * TimeUnit.MINUTE
    cleared_lifetime: Time = 30 * TimeUnit.MINUTE
    anchor_time: Time = 15 * TimeUnit.MINUTE
    claimed_anchor_time: Time = 45 * TimeUnit.MINUTE
    anchor_score: float = 0.85
    follow_score: float = 0.8
    score_per_hop: float = 0.01
    min_follow_score: float = 0.65
    pests_in_knowledge: bool = False


@dataclass(frozen=True)
class GradientEntry:
    """What one robot knows of the gradient to one pest.

    Attributes:
        pest: The pest.
        round: The newest round heard.
        hops: This robot's own count: 0 at the anchor.
        parent: The neighbour the count was heard from, ``None`` at the anchor.
        heard_at: When the count was last refreshed.
    """

    pest: PestId
    round: int
    hops: int
    parent: RobotId | None
    heard_at: Time


class GradientTable:
    """One robot's hop counts, the pests it anchors, the pests known cleared, and the claims.

    Attributes:
        robot_id: The robot that keeps this table, which numbers the pests it anchors.
        config: Hop limit and ages.
    """

    def __init__(self, robot_id: RobotId, config: GradientConfig) -> None:
        self.robot_id: RobotId = robot_id
        self.config: GradientConfig = config
        self._entries: dict[PestId, GradientEntry] = {}
        self._cleared: dict[PestId, Time] = {}
        self._claims: dict[tuple[PestId, RobotId], HopClaim] = {}
        self._anchored: dict[CellId, tuple[PestId, Time]] = {}
        self._numbered: int = 0
        self._last_round: dict[PestId, int] = {}
        self._anchors_heard: dict[RobotId, Time] = {}

    def receive(self, gradient: HopGradient, sender: RobotId, now: Time) -> None:
        """Take in a neighbour's count, one hop more than the neighbour's own."""
        if gradient.cleared:
            self._mark_cleared(gradient.pest, now)
            return
        if gradient.pest in self._cleared:
            return
        if gradient.hops == 0:
            self._anchors_heard[sender] = now
        hops = gradient.hops + 1
        if hops > self.config.max_hops:
            return
        known = self._entries.get(gradient.pest)
        if known is not None and known.hops == 0:
            return
        # Only a round newer than any heard before refreshes a count. Without this, robots
        # whose counts expired after the anchor left would learn them back from each other,
        # one hop more each time, and keep a dead gradient alive ("count to infinity").
        newer = gradient.round > self._last_round.get(gradient.pest, -1)
        shorter = known is not None and gradient.round == known.round and hops < known.hops
        if newer or shorter:
            entry = GradientEntry(gradient.pest, gradient.round, hops, sender, now)
            self._entries[gradient.pest] = entry
            self._last_round[gradient.pest] = gradient.round

    def receive_claim(self, claim: HopClaim, now: Time) -> None:
        """Take in a killer's claim, keeping the newest per pest and killer."""
        if now - claim.sent_at > self.config.max_age or claim.killer == self.robot_id:
            return
        key = (claim.pest, claim.killer)
        known = self._claims.get(key)
        if known is None or claim.sent_at > known.sent_at:
            self._claims[key] = claim

    def anchor(self, cell: CellId, now: Time, known_as: PestId | None = None) -> PestId:
        """Send hop 0 for the pest this robot senses at ``cell``, in a new round.

        ``known_as`` is the id the swarm already knows this pest by, when this robot's parent
        for it stands on the same cell: this robot then anchors that pest rather than a new
        one, so that clearing it reaches everyone following it.
        """
        anchored = self._anchored.get(cell)
        if anchored is None:
            pest = known_as
            if pest is None or pest in self._cleared:
                self._numbered += 1
                pest = PestId(finder=self.robot_id, number=self._numbered)
            anchored = (pest, now)
            self._anchored[cell] = anchored
        pest = anchored[0]
        round_number = self._last_round.get(pest, -1) + 1
        self._entries[pest] = GradientEntry(pest, round_number, 0, None, now)
        self._last_round[pest] = round_number
        return pest

    def anchors_heard(self, now: Time) -> set[RobotId]:
        """The robots heard sending hop 0, as anchors, within ``max_age``."""
        return {
            robot
            for robot, heard_at in self._anchors_heard.items()
            if now - heard_at <= self.config.max_age
        }

    def anchored_at(self, cell: CellId) -> PestId | None:
        """The pest this robot anchors at ``cell``, or ``None``."""
        anchored = self._anchored.get(cell)
        if anchored is None:
            return None
        return anchored[0]

    def anchored_since(self, cell: CellId) -> Time | None:
        """When this robot started anchoring the pest at ``cell``, or ``None``."""
        anchored = self._anchored.get(cell)
        if anchored is None:
            return None
        return anchored[1]

    def clear(self, cell: CellId, now: Time) -> None:
        """Mark the pest this robot anchored at ``cell`` cleared: it senses the cell clean."""
        anchored = self._anchored.pop(cell, None)
        if anchored is not None:
            self._mark_cleared(anchored[0], now)

    def claim(self, pest: PestId, now: Time) -> None:
        """Claim ``pest`` for this robot, a killer following its gradient."""
        known = self._entries.get(pest)
        hops = 0
        if known is not None:
            hops = known.hops
        self._claims[(pest, self.robot_id)] = HopClaim(pest, self.robot_id, hops, now)

    def live(self, now: Time) -> list[GradientEntry]:
        """The counts refreshed within ``max_age``, of pests not known cleared."""
        self._prune(now)
        return list(self._entries.values())

    def cleared(self, now: Time) -> list[PestId]:
        """The pests known cleared, still remembered."""
        self._prune(now)
        return list(self._cleared)

    def claimed_by_others(self, now: Time) -> set[PestId]:
        """The pests another killer claimed within ``max_age``."""
        self._prune(now)
        claimed: set[PestId] = set()
        for (pest, killer), _ in self._claims.items():
            if killer != self.robot_id:
                claimed.add(pest)
        return claimed

    def taken_by_others(self, now: Time) -> set[PestId]:
        """The pests another killer has a better claim to than this robot would have.

        A claim is better with fewer hops to the pest than this robot's own count, or as many
        and a killer first in the order of ``swarm.priority``. Every killer applies the same
        rule to the same claims, so of two killers on one pest exactly one gives way.
        """
        self._prune(now)
        taken: set[PestId] = set()
        for (pest, killer), claim in self._claims.items():
            if killer == self.robot_id:
                continue
            mine = self._entries.get(pest)
            if mine is None:
                continue
            nearer = claim.hops < mine.hops
            as_near = claim.hops == mine.hops and ranks_before(claim.killer, self.robot_id, now)
            if nearer or as_near:
                taken.add(pest)
        return taken

    def to_send(self, now: Time) -> tuple[HopGradient | HopClaim, ...]:
        """This robot's counts, the pests it knows cleared, and the claims it knows."""
        self._prune(now)
        sent: list[HopGradient | HopClaim] = []
        for pest in sorted(self._cleared, key=_pest_order):
            sent.append(HopGradient(pest, 0, 0, cleared=True))
        for entry in sorted(self._entries.values(), key=lambda e: (e.hops, _pest_order(e.pest))):
            sent.append(HopGradient(entry.pest, entry.round, entry.hops))
        sent.extend(self._claims.values())
        return tuple(sent)

    def _mark_cleared(self, pest: PestId, now: Time) -> None:
        if pest not in self._cleared:
            self._cleared[pest] = now
        self._entries.pop(pest, None)
        for key in [key for key in self._claims if key[0] == pest]:
            del self._claims[key]
        for cell, (anchored, _) in list(self._anchored.items()):
            if anchored == pest:
                del self._anchored[cell]

    def _prune(self, now: Time) -> None:
        max_age = self.config.max_age
        for pest, entry in list(self._entries.items()):
            if now - entry.heard_at > max_age:
                del self._entries[pest]
        for key, claim in list(self._claims.items()):
            if now - claim.sent_at > max_age or claim.pest in self._cleared:
                del self._claims[key]
        for pest, cleared_at in list(self._cleared.items()):
            if now - cleared_at > self.config.cleared_lifetime:
                del self._cleared[pest]


def _pest_order(pest: PestId) -> tuple[RobotId, int]:
    return (pest.finder, pest.number)


def chosen_gradient(table: GradientTable, now: Time) -> GradientEntry | None:
    """The gradient a killer follows: not taken by a nearer killer first, then fewest hops.

    Ties go to the lowest pest id, so a killer does not flip between equal counts.
    """
    live = table.live(now)
    if not live:
        return None
    taken = table.taken_by_others(now)
    return min(live, key=lambda e: (e.pest in taken, e.hops, _pest_order(e.pest)))


def parent_cell(memory: MemoryView, parent: RobotId | None) -> CellId | None:
    """Where the parent was last reported, from its status messages; ``None`` if never."""
    if parent is None:
        return None
    for peer in memory.peers():
        if peer.peer == parent:
            return peer.cell
    return None


class FollowGradient:
    """The killer's hop-gradient behaviour: drive to its parent, treat the pest on arrival.

    It treats the pest it stands on. Otherwise it bids to follow the gradient field towards
    its parent for the chosen pest, with the pest's ``PestId`` as its target and a score that
    drops with the hops.
    """

    name: str = "follow-gradient"

    def __init__(self, table: GradientTable, seeder: str = "gradient") -> None:
        self._table: GradientTable = table
        self._seeder: str = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        position = ctx.self_state.position
        here = ctx.memory.infestation_at(position)
        if here is not None and here.severity > 0:
            treat = Treat(position)
            return (ActionCandidate(treat, _TREAT_HERE_SCORE, self.name, position, _TREAT),)
        entry = chosen_gradient(self._table, ctx.time)
        if entry is None or entry.hops == 0:
            return ()
        if parent_cell(ctx.memory, entry.parent) is None:
            return ()
        config = self._table.config
        score = config.follow_score - config.score_per_hop * (entry.hops - 1)
        score = max(config.min_follow_score, score)
        follow = FollowField(self._seeder)
        return (ActionCandidate(follow, score, self.name, entry.pest, _TREAT),)


class GradientSeeder:
    """Seeds the gradient field at the parent's cell for the chosen pest, and nowhere else."""

    name: str = "gradient"
    window_blocks: int = 2

    def __init__(self, table: GradientTable) -> None:
        self._table: GradientTable = table

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        entry = chosen_gradient(self._table, now)
        if entry is None:
            return None
        return (entry.pest, entry.parent, parent_cell(memory, entry.parent))

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        entry = chosen_gradient(self._table, now)
        if entry is None:
            return ()
        cell = parent_cell(memory, entry.parent)
        if cell is None:
            return ()
        return (Seed(cell, -1.0),)


class AnchorPest:
    """A scout's behaviour: stay on a pest it found, as the gradient's anchor, for a while.

    It bids to stay idle while the robot stands on a pest it anchors and has anchored it for
    less than ``GradientConfig.anchor_time``, or for less than ``claimed_anchor_time`` while a
    killer has claimed the pest; the robot then sends hop 0 at every decision. It leaves the
    pest to another anchor with a lower id on the same cell, so each pest has one anchor.
    """

    name: str = "anchor"

    def __init__(self, table: GradientTable) -> None:
        self._table: GradientTable = table

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        position = ctx.self_state.position
        here = ctx.memory.infestation_at(position)
        if here is None or here.severity <= 0:
            return ()
        since = self._table.anchored_since(position)
        pest = self._table.anchored_at(position)
        if since is None or pest is None:
            return ()
        if self._other_anchor_here(ctx):
            return ()
        config = self._table.config
        waited = ctx.time - since
        claimed = pest in self._table.claimed_by_others(ctx.time)
        if waited >= config.claimed_anchor_time:
            return ()
        if waited >= config.anchor_time and not claimed:
            return ()
        score = self._table.config.anchor_score
        return (ActionCandidate(Idle(), score, self.name, position, AgentObjective.VERIFY),)

    def _other_anchor_here(self, ctx: DecisionContext) -> bool:
        """Whether another robot on this cell keeps the pest, so this one leaves it.

        A robot already anchoring keeps it against one that has just arrived. Between two
        that arrived together, or two anchoring already, the one first in the order of
        ``swarm.priority`` keeps it. Every robot applies the same rule, so one robot waits on
        each pest. Robots on the same cell are sensed at once, so robots arriving together
        settle it at their first decision.
        """
        position = ctx.self_state.position
        me = ctx.self_state.robot_id
        since = self._table.anchored_since(position)
        anchoring = since is not None and since < ctx.time
        oldest = ctx.time - self._table.config.max_age
        for peer in ctx.memory.peers():
            if peer.peer is None or peer.cell != position or peer.observed_at < oldest:
                continue
            other_anchoring = peer.objective is AgentObjective.VERIFY
            if other_anchoring and not anchoring:
                return True
            if other_anchoring == anchoring and ranks_before(peer.peer, me, ctx.time):
                return True
        return False


class GradientAgent(ComposedAgent):
    """A ``ComposedAgent`` that also anchors pests, keeps hop counts and passes them on.

    At each decision it reads the counts and claims in its inbox, anchors the pest on its own
    cell or clears it when the cell is clean, decides as usual, claims the pest it follows if
    it is a killer, and adds its counts to the messages it sends.

    Attributes:
        gradients: The robot's hop counts, shared with its gradient behaviours and seeder.
    """

    def __init__(
        self,
        robot_id: RobotId,
        behaviors: list[Behavior],
        memory: AgentMemory,
        fields: FieldProvider,
        navigator: Navigator,
        arbiter: Arbiter,
        gradients: GradientTable,
        sharing: SharingConfig | None = None,
        tether: TetherConfig | None = None,
    ) -> None:
        super().__init__(
            robot_id, behaviors, memory, fields, navigator, arbiter, sharing=sharing, tether=tether
        )
        self.gradients: GradientTable = gradients
        self.exchange.keep_pests = gradients.config.pests_in_knowledge

    @override
    def decide(self, perception: Perception) -> Decision:
        now = perception.time
        for sender, item in _gradients_in(perception.inbox):
            if isinstance(item, HopGradient):
                self.gradients.receive(item, sender, now)
            else:
                self.gradients.receive_claim(item, now)
        for fact in perception.sensed:
            if not isinstance(fact, InfestationObservation):
                continue
            if fact.severity > 0:
                self.gradients.anchor(fact.cell, now, self._pest_anchored_next_to(fact.cell, now))
            else:
                self.gradients.clear(fact.cell, now)
        decision = super().decide(perception)
        if perception.self_state.body is BodyType.KILLER:
            followed = chosen_gradient(self.gradients, now)
            if followed is not None and self.inspect().current_goal == FollowGradient.name:
                self.gradients.claim(followed.pest, now)
        message = Message(
            sender=self.robot_id,
            sent_at=now,
            kind=MessageKind.GRADIENT,
            payload=self.gradients.to_send(now),
        )
        return Decision(action=decision.action, outbox=[*decision.outbox, message])

    def _pest_anchored_next_to(self, cell: CellId, now: Time) -> PestId | None:
        """The pest whose anchor, this robot's parent one hop away, stands on ``cell``."""
        for entry in self.gradients.live(now):
            if entry.hops == 1 and parent_cell(self.memory, entry.parent) == cell:
                return entry.pest
        return None


def _gradients_in(inbox: Iterable[Message]) -> list[tuple[RobotId, HopGradient | HopClaim]]:
    """The counts and claims carried by the received messages, with their senders."""
    found: list[tuple[RobotId, HopGradient | HopClaim]] = []
    for message in inbox:
        if message.kind is not MessageKind.GRADIENT:
            continue
        if not isinstance(message.payload, (tuple, list)):
            logger.warning(
                "Skipping a gradient message from %s with payload %r",
                message.sender,
                message.payload,
            )
            continue
        for item in cast(tuple[object, ...], tuple(message.payload)):
            if isinstance(item, (HopGradient, HopClaim)):
                found.append((message.sender, item))
    return found
