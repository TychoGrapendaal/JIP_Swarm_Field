"""Tests for the composed brain and high-level movement action."""

from collections.abc import Hashable, Sequence
from dataclasses import replace

import pytest

from swarm_field.contracts import (
    AgentObjective,
    BodyType,
    CellId,
    Charge,
    CropObservation,
    FieldGeometry,
    Idle,
    InfestationObservation,
    Message,
    MessageKind,
    Move,
    Observation,
    PeerObservation,
    Perception,
    RobotSelfState,
    Terrain,
    TerrainObservation,
    Time,
    TimeUnit,
    Treat,
)
from swarm_field.swarm import (
    ActionCandidate,
    AgentMemory,
    ArbiterChoice,
    Behavior,
    ComposedAgent,
    DecisionContext,
    FieldProvider,
    FieldSeeder,
    FollowField,
    Intent,
    KnowledgeSharing,
    KnowledgeSnapshot,
    MemoryView,
    Navigator,
    Regroup,
    RegroupSeeder,
    Resupply,
    SharingConfig,
    TreatInfestation,
    TreatSeeder,
)
from swarm_field.swarm.field import Seed

_FIELD = FieldGeometry(width=10, height=10, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)


def _perception(
    sensed: list[Observation] | None = None, body: BodyType = BodyType.SCOUT
) -> Perception:
    time = 0
    if sensed is None:
        sensed = _sensed(time)
    return Perception(
        time=time,
        self_state=RobotSelfState(
            robot_id=1,
            position=(0, 0),
            battery=1.0,
            body=body,
            speed_mps=2.0,
            consumable=1.0,
            health=1.0,
        ),
        sensed=sensed,
        inbox=[],
    )


def _sensed(time: Time) -> list[Observation]:
    return [
        InfestationObservation(
            cell=(3, 3),
            confidence=1.0,
            observed_at=time,
            source=1,
            severity=0.8,
            pest=None,
        ),
        TerrainObservation(
            cell=(2, 2),
            confidence=1.0,
            observed_at=time,
            source=1,
            terrain=Terrain.OBSTACLE,
        ),
        PeerObservation(
            cell=(2, 0),
            confidence=1.0,
            observed_at=time,
            source=1,
            peer=2,
            body=BodyType.KILLER,
        ),
    ]


class _MaxArbiter:
    """A mock arbiter: the highest score wins, without hysteresis."""

    def choose(self, candidates: Sequence[ActionCandidate], now: Time) -> ArbiterChoice | None:
        if not candidates:
            return None
        return ArbiterChoice(max(candidates, key=lambda c: c.score), switched=False)


def _agent(behaviors: list[Behavior], seeders: list[FieldSeeder] | None = None) -> ComposedAgent:
    memory = AgentMemory(1, _FIELD)
    fields = FieldProvider(memory, flies=False, seeders=seeders or [])
    return ComposedAgent(
        robot_id=1,
        behaviors=behaviors,
        memory=memory,
        fields=fields,
        navigator=Navigator(fields.quadtree),
        arbiter=_MaxArbiter(),
    )


class _FixedBehavior:
    def __init__(self, name: str, action: Intent, score: float) -> None:
        self.name: str = name
        self._action: Intent = action
        self._score: float = score

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        return (
            ActionCandidate(action=self._action, score=self._score, source=self.name, target=None),
        )


class _MultiBehavior:
    def __init__(self, name: str, candidates: tuple[ActionCandidate, ...]) -> None:
        self.name: str = name
        self._candidates: tuple[ActionCandidate, ...] = candidates

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        return self._candidates


class _FixedTargetSeeder:
    name: str = "fixed-target"
    window_blocks: int = 2

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        return None

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        return (Seed((5, 5), 0.0),)


class _CountingSeeder:
    """A seeder with one fixed target, counting how often it is asked for seeds."""

    def __init__(self, name: str, target: CellId) -> None:
        self.name: str = name
        self.window_blocks: int = 2
        self.seeds_asked: int = 0
        self._target: CellId = target

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        return None

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        self.seeds_asked += 1
        return (Seed(self._target, 0.0),)


class _FieldFollower:
    """Bids a fixed score to follow its seeder's field."""

    def __init__(self, seeder: str, score: float, objective: AgentObjective | None = None) -> None:
        self.name: str = seeder
        self._score: float = score
        self._objective: AgentObjective | None = objective

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        follow = FollowField(self.name)
        return (ActionCandidate(follow, self._score, self.name, None, self._objective),)


def test_agent_picks_highest_scoring_candidate() -> None:
    agent = _agent(
        [
            _FixedBehavior("low", Idle(), 0.1),
            _MultiBehavior(
                "multi",
                (
                    ActionCandidate(action=Treat((1, 0)), score=0.9, source="multi", target=None),
                    ActionCandidate(action=Treat((0, 1)), score=0.5, source="multi", target=None),
                ),
            ),
        ]
    )

    decision = agent.decide(_perception(body=BodyType.KILLER))

    assert decision.action == Treat((1, 0))
    assert agent.inspect().current_goal == "multi"


def test_decision_context_does_not_expose_field_provider() -> None:
    assert "fields" not in DecisionContext.__dataclass_fields__


def test_agent_idles_without_candidates() -> None:
    agent = _agent([])

    decision = agent.decide(_perception())

    assert decision.action == Idle()
    assert agent.inspect().current_goal is None


def test_sensed_terrain_reaches_memory_and_the_quadtree() -> None:
    agent = _agent([])
    wall = TerrainObservation(
        cell=(1, 1), confidence=1.0, observed_at=0, source=1, terrain=Terrain.OBSTACLE
    )

    agent.decide(_perception(sensed=[wall]))

    assert wall in list(agent.memory.records())
    assert agent.fields.quadtree is not None
    assert not agent.fields.quadtree.is_traversable((1, 1))


def test_treat_infestation_follows_the_field_to_a_known_pest_and_treats_it_on_the_spot() -> None:
    memory = AgentMemory(1, _FIELD)
    memory.merge(_sensed(0))
    behavior = TreatInfestation()

    def proposed_at(position: CellId) -> list[Intent]:
        self_state = replace(_perception().self_state, position=position)
        ctx = DecisionContext(time=0, self_state=self_state, sensed=[], memory=memory)
        return [c.action for c in behavior.propose(ctx)]

    assert proposed_at((0, 0)) == [FollowField(TreatSeeder.name)]
    assert proposed_at((3, 3)) == [Treat((3, 3))]


def test_a_yielding_killer_waits_on_a_pest_a_lower_id_peer_already_treats() -> None:
    behavior = TreatInfestation(yield_to_treating_peer=True)

    def proposed_with_peer(peer: int) -> list[Intent]:
        memory = AgentMemory(5, _FIELD)
        memory.merge([*_sensed(0), _treating_peer(peer, (3, 3), (3, 3), observed_at=1)])
        self_state = replace(_perception().self_state, robot_id=5, position=(3, 3))
        ctx = DecisionContext(time=1, self_state=self_state, sensed=[], memory=memory)
        return [c.action for c in behavior.propose(ctx)]

    assert proposed_with_peer(2) == []
    assert proposed_with_peer(9) == [Treat((3, 3))]


def test_a_winning_follow_field_becomes_a_step_along_the_field() -> None:
    agent = _agent([_FieldFollower(_FixedTargetSeeder.name, 0.5)], [_FixedTargetSeeder()])

    decision = agent.decide(_perception(sensed=[]))

    assert decision.action == Move((1, 1))
    assert agent.inspect().current_goal == _FixedTargetSeeder.name


def test_agent_broadcasts_its_memory_and_peer_learns_it() -> None:
    sender = _agent([])
    receiver_memory = AgentMemory(2, _FIELD)
    receiver_fields = FieldProvider(receiver_memory, flies=False, seeders=[])
    receiver = ComposedAgent(
        robot_id=2,
        behaviors=[],
        memory=receiver_memory,
        fields=receiver_fields,
        navigator=Navigator(receiver_fields.quadtree),
        arbiter=_MaxArbiter(),
    )
    fact = InfestationObservation(
        cell=(6, 6),
        confidence=1.0,
        observed_at=0,
        source=1,
        severity=0.8,
        pest=None,
    )
    crop_fact = CropObservation(
        cell=(7, 6),
        confidence=0.9,
        observed_at=0,
        source=1,
        crop=None,
        growth_stage=0.65,
    )

    first = sender.decide(
        Perception(
            time=0,
            self_state=RobotSelfState(
                robot_id=1,
                position=(0, 0),
                battery=1.0,
                body=BodyType.SCOUT,
                speed_mps=1.0,
                consumable=1.0,
                health=1.0,
            ),
            sensed=[fact, crop_fact],
            inbox=[],
        )
    )

    status, knowledge = first.outbox
    assert status.kind is MessageKind.STATUS
    assert isinstance(status.payload, PeerObservation)
    assert status.payload.battery == 1.0
    assert knowledge.kind is MessageKind.KNOWLEDGE
    assert isinstance(knowledge.payload, KnowledgeSnapshot)
    assert set(knowledge.payload.facts()) == {fact, crop_fact}

    receiver.decide(
        Perception(
            time=5_000,
            self_state=RobotSelfState(
                robot_id=2,
                position=(0, 0),
                battery=1.0,
                body=BodyType.SCOUT,
                speed_mps=1.0,
                consumable=1.0,
                health=1.0,
            ),
            sensed=[],
            inbox=first.outbox,
        )
    )

    assert receiver_memory.infestation_at((6, 6)) == fact
    assert crop_fact in set(receiver_memory.records())


def test_only_the_winners_field_is_built_and_it_is_not_rebuilt() -> None:
    winner = _CountingSeeder("winner", (5, 5))
    loser = _CountingSeeder("loser", (0, 9))
    agent = _agent([_FieldFollower("winner", 2.0), _FieldFollower("loser", 1.0)], [winner, loser])
    resensed = TerrainObservation(
        cell=(1, 0), confidence=1.0, observed_at=0, source=1, terrain=Terrain.FREE
    )

    for _ in range(3):
        decision = agent.decide(_perception(sensed=[resensed]))
        assert decision.action == Move((1, 1))

    assert winner.seeds_asked == 1
    assert loser.seeds_asked == 0


_SILENCE_TIMEOUT = 2 * TimeUnit.MINUTE


def _peer_at(peer: int, cell: CellId) -> PeerObservation:
    return PeerObservation(
        cell=cell, confidence=1.0, observed_at=0, source=peer, peer=peer, body=BodyType.KILLER
    )


def _regrouping_agent(peers: list[PeerObservation], rally_point: CellId | None) -> ComposedAgent:
    """An agent that knows of ``peers`` and was last in contact at time 0."""
    seeder = RegroupSeeder(rally_point=rally_point)
    agent = _agent([Regroup(seeder.name, _SILENCE_TIMEOUT, rally_point)], seeders=[seeder])
    agent.decide(_perception_at(0, (0, 0), sensed=list(peers)))
    return agent


def _perception_at(
    time: Time,
    position: CellId,
    sensed: list[Observation] | None = None,
    inbox: list[Message] | None = None,
) -> Perception:
    """A killer at ``position`` that senses its own cell, as in the simulation, and ``sensed``."""
    own_cell = InfestationObservation(
        cell=position, confidence=1.0, observed_at=time, source=1, severity=0.0, pest=None
    )
    return Perception(
        time=time,
        self_state=RobotSelfState(
            robot_id=1,
            position=position,
            battery=1.0,
            body=BodyType.KILLER,
            speed_mps=2.0,
            consumable=1.0,
            health=1.0,
        ),
        sensed=[own_cell, *(sensed or [])],
        inbox=inbox or [],
    )


def test_regroup_does_not_bid_before_the_silence_timeout() -> None:
    agent = _regrouping_agent([_peer_at(2, (5, 0))], rally_point=None)

    decision = agent.decide(_perception_at(_SILENCE_TIMEOUT - 1, (0, 0)))

    assert decision.action == Idle()
    assert agent.inspect().current_goal is None


def test_regroup_heads_for_a_peers_last_known_cell_after_a_long_silence() -> None:
    agent = _regrouping_agent([_peer_at(2, (5, 0))], rally_point=None)

    decision = agent.decide(_perception_at(_SILENCE_TIMEOUT, (0, 0)))

    assert decision.action == Move((1, 0))
    assert agent.inspect().current_goal == Regroup.name
    status = decision.outbox[0].payload
    assert isinstance(status, PeerObservation)
    assert status.target == (5, 0)


def test_a_received_message_ends_the_silence() -> None:
    agent = _regrouping_agent([_peer_at(2, (5, 0))], rally_point=None)
    hello = Message(sender=2, sent_at=_SILENCE_TIMEOUT, kind=MessageKind.HELLO, payload=None)

    heard = agent.decide(_perception_at(_SILENCE_TIMEOUT, (0, 0), inbox=[hello]))
    soon_after = agent.decide(_perception_at(_SILENCE_TIMEOUT + 5_000, (0, 0)))

    assert heard.action == Idle()
    assert soon_after.action == Idle()


def test_regroup_moves_on_from_an_empty_peer_cell_to_the_rally_point() -> None:
    agent = _regrouping_agent([_peer_at(2, (5, 0))], rally_point=(5, 5))

    on_peer_cell = agent.decide(_perception_at(_SILENCE_TIMEOUT, (5, 0)))
    at_rally_point = agent.decide(_perception_at(_SILENCE_TIMEOUT + 5_000, (5, 5)))

    assert on_peer_cell.action == Move((5, 1))
    assert at_rally_point.action == Idle()
    assert agent.inspect().current_goal == Regroup.name


def test_the_agent_broadcasts_its_cell_and_the_winners_target_and_objective() -> None:
    explore = _FieldFollower(_FixedTargetSeeder.name, 0.5, AgentObjective.EXPLORE)
    agent = _agent([explore], [_FixedTargetSeeder()])

    decision = agent.decide(_perception(sensed=[]))

    message, _knowledge = decision.outbox
    assert message.kind is MessageKind.STATUS
    assert isinstance(message.payload, PeerObservation)
    assert (message.payload.peer, message.payload.cell) == (1, (0, 0))
    assert message.payload.target == (5, 5)
    assert message.payload.objective is AgentObjective.EXPLORE
    assert message.payload.battery == 1.0
    assert agent.inspect().current_target == (5, 5)


@pytest.mark.parametrize(
    ("sharing", "shared_kinds"),
    [
        (
            KnowledgeSharing.EVERYTHING,
            {InfestationObservation, TerrainObservation, PeerObservation},
        ),
        (KnowledgeSharing.INFESTATIONS, {InfestationObservation}),
        (KnowledgeSharing.INFESTATIONS_AND_TARGETS, {InfestationObservation}),
        (KnowledgeSharing.NOTHING, None),
    ],
)
def test_the_sharing_level_selects_the_broadcast_facts(
    sharing: KnowledgeSharing, shared_kinds: set[type] | None
) -> None:
    agent = _agent([])
    agent.exchange.config = SharingConfig(facts=sharing)

    decision = agent.decide(_perception())

    knowledge = [m for m in decision.outbox if m.kind is MessageKind.KNOWLEDGE]
    assert decision.outbox[0].kind is MessageKind.STATUS
    if shared_kinds is None:
        assert knowledge == []
    else:
        (message,) = knowledge
        assert isinstance(message.payload, KnowledgeSnapshot)
        assert {type(fact) for fact in message.payload.facts()} == shared_kinds


def test_battery_body_status_broadcast_includes_its_current_battery_level() -> None:
    agent = _agent([])
    base_state = _perception(sensed=[]).self_state
    battery_state = replace(base_state, body=BodyType.BATTERY, battery=0.17)

    decision = agent.decide(replace(_perception(sensed=[]), self_state=battery_state))

    status, _knowledge = decision.outbox
    assert status.kind is MessageKind.STATUS
    assert isinstance(status.payload, PeerObservation)
    assert status.payload.body is BodyType.BATTERY
    assert status.payload.battery == 0.17


def test_a_peers_status_is_stored_as_a_peer_fact() -> None:
    agent = _agent([])
    status = PeerObservation(
        cell=(4, 4),
        confidence=1.0,
        observed_at=0,
        source=2,
        peer=2,
        body=BodyType.SCOUT,
        target=(7, 7),
    )
    message = Message(sender=2, sent_at=0, kind=MessageKind.STATUS, payload=status)

    agent.decide(replace(_perception(sensed=[]), inbox=[message]))

    assert list(agent.memory.peers()) == [status]


def _memory_with_a_pest_and_a_killer_heading_for_it(
    killer_cell: CellId, objective: AgentObjective
) -> AgentMemory:
    memory = AgentMemory(1, _FIELD)
    pest = InfestationObservation(
        cell=(5, 5), confidence=1.0, observed_at=0, source=1, severity=1.0, pest=None
    )
    peer = PeerObservation(
        cell=killer_cell,
        confidence=1.0,
        observed_at=0,
        source=2,
        peer=2,
        body=BodyType.KILLER,
        target=(5, 5),
        objective=objective,
    )
    memory.merge([pest, peer])
    return memory


def test_the_treat_field_leaves_a_pest_to_a_closer_killer_heading_for_it() -> None:
    def seeds_from_next_to_it(killer_cell: CellId, objective: AgentObjective) -> int:
        memory = _memory_with_a_pest_and_a_killer_heading_for_it(killer_cell, objective)
        return len(TreatSeeder(yield_to_closer=True).seeds(memory, (6, 6), 0))

    assert seeds_from_next_to_it((5, 4), AgentObjective.TREAT) == 0
    assert seeds_from_next_to_it((0, 0), AgentObjective.TREAT) == 1
    assert seeds_from_next_to_it((5, 4), AgentObjective.EXPLORE) == 1


def test_of_two_killers_on_one_cell_only_the_lower_id_keeps_their_pest() -> None:
    def seeds_of(robot: int, peer: int) -> int:
        memory = AgentMemory(robot, _FIELD)
        pest = InfestationObservation(
            cell=(5, 5), confidence=1.0, observed_at=0, source=robot, severity=1.0, pest=None
        )
        heading = PeerObservation(
            cell=(6, 6),
            confidence=1.0,
            observed_at=0,
            source=peer,
            peer=peer,
            body=BodyType.KILLER,
            target=(5, 5),
            objective=AgentObjective.TREAT,
        )
        memory.merge([pest, heading])
        return len(TreatSeeder(yield_to_closer=True).seeds(memory, (6, 6), 0))

    assert seeds_of(robot=1, peer=2) == 1
    assert seeds_of(robot=2, peer=1) == 0


def test_treat_infestation_bids_above_exploring_for_a_pest_left_to_it_and_not_otherwise() -> None:
    behavior = TreatInfestation(yield_to_treating_peer=True)

    def bids_from_next_to_it(killer_cell: CellId) -> list[float]:
        memory = _memory_with_a_pest_and_a_killer_heading_for_it(killer_cell, AgentObjective.TREAT)
        self_state = replace(_perception().self_state, position=(6, 6))
        ctx = DecisionContext(time=0, self_state=self_state, sensed=[], memory=memory)
        return [candidate.score for candidate in behavior.propose(ctx)]

    assert bids_from_next_to_it((5, 4)) == []
    (bid,) = bids_from_next_to_it((0, 0))
    assert 0.8 <= bid < 0.9


def _pest(cell: CellId) -> InfestationObservation:
    return InfestationObservation(
        cell=cell, confidence=1.0, observed_at=0, source=1, severity=1.0, pest=None
    )


def _treating_peer(peer: int, cell: CellId, target: CellId, observed_at: int) -> PeerObservation:
    return PeerObservation(
        cell=cell,
        confidence=1.0,
        observed_at=observed_at,
        source=peer,
        peer=peer,
        body=BodyType.KILLER,
        target=target,
        objective=AgentObjective.TREAT,
    )


def test_a_yielding_treat_seeder_leaves_a_pest_to_a_closer_fresh_claimant() -> None:
    memory = AgentMemory(1, _FIELD)
    memory.merge([_pest((5, 5)), _pest((9, 0)), _treating_peer(2, (6, 6), (5, 5), observed_at=0)])
    seeder = TreatSeeder(yield_to_closer=True)

    near_the_claimant = {seed.cell for seed in seeder.seeds(memory, (0, 0), 0)}
    closer_than_it = {seed.cell for seed in seeder.seeds(memory, (5, 4), 0)}
    after_the_claim_aged = {seed.cell for seed in seeder.seeds(memory, (0, 0), TimeUnit.HOUR)}

    assert near_the_claimant == {(9, 0)}
    assert closer_than_it == {(5, 5), (9, 0)}
    assert after_the_claim_aged == {(5, 5), (9, 0)}


def test_an_empty_killer_does_not_bid_to_treat_even_on_a_pest() -> None:
    memory = AgentMemory(1, _FIELD)
    memory.merge([_pest((0, 0))])
    self_state = replace(_perception(body=BodyType.KILLER).self_state, consumable=0.0)
    ctx = DecisionContext(time=0, self_state=self_state, sensed=[], memory=memory)

    assert TreatInfestation().propose(ctx) == ()


def test_resupply_bids_more_as_the_consumable_runs_out_and_restocks_on_a_dock() -> None:
    behavior = Resupply(docks=((9, 9),))
    memory = AgentMemory(1, _FIELD)
    killer = _perception(body=BodyType.KILLER).self_state

    def bids(consumable: float, position: CellId = (0, 0)) -> list[ActionCandidate]:
        self_state = replace(killer, consumable=consumable, position=position)
        ctx = DecisionContext(time=0, self_state=self_state, sensed=[], memory=memory)
        return list(behavior.propose(ctx))

    (half_low,) = bids(0.1)
    (empty,) = bids(0.0)
    (on_dock,) = bids(0.1, position=(9, 9))

    assert bids(0.5) == []
    assert half_low.score < 0.8 < 0.9 < empty.score
    assert isinstance(empty.action, FollowField)
    assert isinstance(on_dock.action, Charge)
