"""Hop-based pest navigation: pest news spreads robot to robot, killers follow the nearest chain.

Instead of every robot sharing its whole memory, news of a pest travels as a ``PestBeacon``:
the robot that senses a pest starts a beacon at hop 0, and every robot that hears one passes
it on one hop further, up to ``HopConfig.max_hops`` and for ``HopConfig.lifetime``. A robot
that finds the pest gone passes on a cleared beacon, which wins over the live one wherever it
arrives, so the news dies out instead of circulating.

A killer picks the live beacon with the fewest hops, that is the pest nearest to it through
the swarm, and goes there along the usual grid potential field, seeded at that pest's cell
only. The hop count is the swarm's gradient; the potential field is how the killer drives
around obstacles on the way. The killer bids with the pest's ``PestId`` as its target, so the
arbiter keeps the effort of a trip even while the beacon it follows is refreshed.

``HopAgent`` is a ``ComposedAgent`` that also reads, starts, clears and passes on beacons.
With ``HopConfig.pests_in_knowledge`` off it ignores the pests in its peers' memory
snapshots, so beacons are the only way it learns of a pest it has not sensed itself.
"""

import logging
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, replace
from typing import cast, override

from swarm_field.contracts import (
    AgentObjective,
    CellId,
    Decision,
    InfestationObservation,
    Message,
    MessageKind,
    Perception,
    PestBeacon,
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
from swarm_field.swarm.provider import FieldProvider
from swarm_field.swarm.sharing import SharingConfig
from swarm_field.swarm.tether import TetherConfig

logger = logging.getLogger(__name__)

_TREAT = AgentObjective.TREAT
_TREAT_HERE_SCORE = 0.99
"""Treating the pest underfoot, as ``TreatInfestation`` bids it."""


@dataclass(frozen=True)
class HopConfig:
    """How far and how long pest news travels, and how much a killer wants to follow it.

    Attributes:
        max_hops: Beacons are not passed on beyond this many hops from their finder.
        lifetime: A beacon expires this long after its pest was found.
        max_beacons: At most this many beacons are passed on per message, fewest hops first.
        follow_score: The bid to follow the beacon at hop 0. Above exploring's 0.5 plus the
            arbiter's 0.125 objective floor, so a killer leaves exploring when it hears of a pest.
        score_per_hop: How much less the bid is per hop, so news from far away matters less.
        min_follow_score: The bid never drops below this while a beacon is live.
        claim_freshness: A peer's report of heading for a pest's cell counts this long.
        pests_in_knowledge: Whether pests in peers' memory snapshots are believed too. Off,
            beacons are the only way a robot learns of a pest it has not sensed itself.
    """

    max_hops: int = 20
    lifetime: Time = 30 * TimeUnit.MINUTE
    max_beacons: int = 12
    follow_score: float = 0.8
    score_per_hop: float = 0.01
    min_follow_score: float = 0.65
    claim_freshness: Time = 2 * TimeUnit.MINUTE
    pests_in_knowledge: bool = False


class BeaconTable:
    """The beacons one robot knows of: the best news per pest, and the pests known cleared.

    Attributes:
        config: Hop limit, lifetime and message size.
        robot_id: The robot that keeps this table, which numbers the pests it finds.
    """

    def __init__(self, robot_id: RobotId, config: HopConfig) -> None:
        self.robot_id: RobotId = robot_id
        self.config: HopConfig = config
        self._beacons: dict[PestId, PestBeacon] = {}
        self._found: int = 0

    def receive(self, beacon: PestBeacon, now: Time) -> None:
        """Take in a peer's beacon, one hop further from its finder than the peer was."""
        if self._expired(beacon, now):
            return
        heard = replace(beacon, hops=beacon.hops + 1)
        if not heard.cleared and heard.hops > self.config.max_hops:
            return
        self._keep(heard)

    def found(self, cell: CellId, now: Time) -> None:
        """Start a beacon for a pest this robot senses, unless one for its cell is live."""
        for beacon in self.live(now):
            if beacon.cell == cell:
                return
        self._found += 1
        pest = PestId(finder=self.robot_id, number=self._found)
        self._keep(PestBeacon(pest=pest, cell=cell, hops=0, found_at=now))

    def clear(self, cell: CellId, now: Time) -> None:
        """Mark every live beacon at ``cell`` as cleared: the pest is treated or not there."""
        for beacon in self.live(now):
            if beacon.cell == cell:
                self._beacons[beacon.pest] = replace(beacon, hops=0, cleared=True)

    def live(self, now: Time) -> list[PestBeacon]:
        """The beacons of pests not known cleared and not expired."""
        return [b for b in self._beacons.values() if not b.cleared and not self._expired(b, now)]

    def known(self, now: Time) -> list[PestBeacon]:
        """Every beacon not expired, live or cleared, fewest hops first."""
        fresh = [b for b in self._beacons.values() if not self._expired(b, now)]
        fresh.sort(key=lambda b: (b.hops, b.pest.finder, b.pest.number))
        return fresh

    def to_send(self, now: Time) -> tuple[PestBeacon, ...]:
        """The beacons to pass on: cleared ones first, then live ones by hops, capped."""
        fresh = [b for b in self._beacons.values() if not self._expired(b, now)]
        fresh.sort(key=lambda b: (not b.cleared, b.hops, b.pest.finder, b.pest.number))
        self._beacons = {b.pest: b for b in fresh}
        return tuple(fresh[: self.config.max_beacons])

    def _keep(self, beacon: PestBeacon) -> None:
        """Keep ``beacon`` if it is better news than what is known of its pest.

        A cleared beacon always wins and is never undone. Between live ones, fewer hops win.
        """
        known = self._beacons.get(beacon.pest)
        if known is not None and known.cleared:
            return
        if known is None or beacon.cleared or beacon.hops < known.hops:
            self._beacons[beacon.pest] = beacon

    def _expired(self, beacon: PestBeacon, now: Time) -> bool:
        return now - beacon.found_at > self.config.lifetime


def chosen_beacon(
    table: BeaconTable, memory: MemoryView, robot: RobotId, now: Time
) -> PestBeacon | None:
    """The live beacon a killer follows: unclaimed first, then fewest hops.

    A pest whose cell a peer recently reported heading for, to treat it, comes after every
    unclaimed one, so killers spread over the pests they hear of. Ties go to the oldest news,
    then to the finder's id, so a killer does not flip between equal beacons.
    """
    claimed: set[CellId] = set()
    oldest = now - table.config.claim_freshness
    for peer in memory.peers():
        fresh = peer.observed_at >= oldest and peer.peer != robot
        if fresh and peer.objective is _TREAT and peer.target is not None:
            claimed.add(peer.target)
    live = table.live(now)
    if not live:
        return None
    return min(
        live,
        key=lambda b: (b.cell in claimed, b.hops, b.found_at, b.pest.finder, b.pest.number),
    )


class FollowBeacon:
    """The killer's hop-based pest behaviour: go to the pest of the chosen beacon, and treat it.

    It treats the pest it stands on, as ``TreatInfestation`` does. Otherwise it bids to follow
    the beacon field, with the chosen pest's ``PestId`` as its target and a score that drops
    with the hops the news has travelled.
    """

    name: str = "follow-beacon"

    def __init__(self, table: BeaconTable, seeder: str = "beacon") -> None:
        self._table: BeaconTable = table
        self._seeder: str = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        position = ctx.self_state.position
        here = ctx.memory.infestation_at(position)
        if here is not None and here.severity > 0:
            treat = Treat(position)
            return (ActionCandidate(treat, _TREAT_HERE_SCORE, self.name, position, _TREAT),)
        beacon = chosen_beacon(self._table, ctx.memory, ctx.self_state.robot_id, ctx.time)
        if beacon is None:
            return ()
        config = self._table.config
        score = config.follow_score - config.score_per_hop * beacon.hops
        score = max(config.min_follow_score, score)
        follow = FollowField(self._seeder)
        return (ActionCandidate(follow, score, self.name, beacon.pest, _TREAT),)


class BeaconSeeder:
    """Seeds the beacon field at the chosen beacon's pest, and nowhere else.

    The killer and its field agree on the pest because both ask ``chosen_beacon``. The field is
    rebuilt when the chosen pest changes.
    """

    name: str = "beacon"
    window_blocks: int = 2

    def __init__(self, table: BeaconTable, robot: RobotId) -> None:
        self._table: BeaconTable = table
        self._robot: RobotId = robot

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        beacon = chosen_beacon(self._table, memory, self._robot, now)
        if beacon is None:
            return None
        return (beacon.pest, beacon.cell)

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        beacon = chosen_beacon(self._table, memory, self._robot, now)
        if beacon is None:
            return ()
        return (Seed(beacon.cell, -1.0),)


class HopAgent(ComposedAgent):
    """A ``ComposedAgent`` that also starts, clears and passes on pest beacons.

    At each decision it reads the beacons in its inbox, starts one for every pest it senses,
    clears the beacons of every cell it senses clean, decides as usual, and adds its beacons to
    the messages it sends.

    Attributes:
        beacons: The beacons this robot knows of, shared with its ``FollowBeacon`` and
            ``BeaconSeeder``.
    """

    def __init__(
        self,
        robot_id: RobotId,
        behaviors: list[Behavior],
        memory: AgentMemory,
        fields: FieldProvider,
        navigator: Navigator,
        arbiter: Arbiter,
        beacons: BeaconTable,
        sharing: SharingConfig | None = None,
        tether: TetherConfig | None = None,
    ) -> None:
        super().__init__(
            robot_id, behaviors, memory, fields, navigator, arbiter, sharing=sharing, tether=tether
        )
        self.beacons: BeaconTable = beacons
        self.exchange.keep_pests = beacons.config.pests_in_knowledge

    @override
    def decide(self, perception: Perception) -> Decision:
        now = perception.time
        for beacon in _beacons_in(perception.inbox):
            self.beacons.receive(beacon, now)
        for fact in perception.sensed:
            if not isinstance(fact, InfestationObservation):
                continue
            if fact.severity > 0:
                self.beacons.found(fact.cell, now)
            else:
                self.beacons.clear(fact.cell, now)
        decision = super().decide(perception)
        beacon_message = Message(
            sender=self.robot_id,
            sent_at=now,
            kind=MessageKind.BEACON,
            payload=self.beacons.to_send(now),
        )
        return Decision(action=decision.action, outbox=[*decision.outbox, beacon_message])


def _beacons_in(inbox: Iterable[Message]) -> list[PestBeacon]:
    """The beacons carried by the received messages."""
    beacons: list[PestBeacon] = []
    for message in inbox:
        if message.kind is not MessageKind.BEACON:
            continue
        if not isinstance(message.payload, (tuple, list)):
            logger.warning(
                "Skipping a beacon message from %s with payload %r", message.sender, message.payload
            )
            continue
        for item in cast(tuple[object, ...], tuple(message.payload)):
            if isinstance(item, PestBeacon):
                beacons.append(item)
    return beacons
