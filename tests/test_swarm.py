"""Tests for the composed brain and high-level movement action."""

from collections.abc import Hashable, Sequence
from dataclasses import replace

import pytest

from swarm_field.contracts import (
    AgentObjective,
    BodyType,
    CellId,
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
    MemoryView,
    Navigator,
    TreatInfestation,
    TreatSeeder,
)
from swarm_field.swarm.field import Seed

_FIELD = FieldGeometry(width=10, height=10, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)


def _perception(sensed: list[Observation] | None = None) -> Perception:
    time = 0
    if sensed is None:
        sensed = _sensed(time)
    return Perception(
        time=time,
        self_state=RobotSelfState(
            robot_id=1,
            position=(0, 0),
            battery=1.0,
            body=BodyType.SCOUT,
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

    decision = agent.decide(_perception())

    assert decision.action == Treat((1, 0))
    assert agent.inspect().current_goal == "multi"


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
            sensed=[fact],
            inbox=[],
        )
    )

    status, knowledge = first.outbox
    assert status.kind is MessageKind.STATUS
    assert knowledge.kind is MessageKind.KNOWLEDGE
    assert knowledge.payload == (fact,)

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
    assert agent.inspect().current_target == (5, 5)


@pytest.mark.parametrize(
    ("sharing", "shared_kinds"),
    [
        (
            KnowledgeSharing.EVERYTHING,
            {InfestationObservation, TerrainObservation, PeerObservation},
        ),
        (KnowledgeSharing.INFESTATIONS, {InfestationObservation}),
        (KnowledgeSharing.NOTHING, None),
    ],
)
def test_the_sharing_level_selects_the_broadcast_facts(
    sharing: KnowledgeSharing, shared_kinds: set[type] | None
) -> None:
    agent = _agent([])
    agent.sharing = sharing

    decision = agent.decide(_perception())

    knowledge = [m for m in decision.outbox if m.kind is MessageKind.KNOWLEDGE]
    assert decision.outbox[0].kind is MessageKind.STATUS
    if shared_kinds is None:
        assert knowledge == []
    else:
        (message,) = knowledge
        assert isinstance(message.payload, tuple)
        assert {type(fact) for fact in message.payload} == shared_kinds


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


def test_treat_seeder_discounts_only_a_pest_a_peer_is_heading_for_to_treat() -> None:
    def pest_seed_value(peer_objective: AgentObjective) -> float:
        memory = AgentMemory(1, _FIELD)
        pest = InfestationObservation(
            cell=(5, 5), confidence=1.0, observed_at=0, source=1, severity=1.0, pest=None
        )
        peer = PeerObservation(
            cell=(0, 9),
            confidence=1.0,
            observed_at=0,
            source=2,
            peer=2,
            body=BodyType.KILLER,
            target=(5, 5),
            objective=peer_objective,
        )
        memory.merge([pest, peer])
        (seed,) = TreatSeeder().seeds(memory, (0, 0), 0)
        return seed.value

    assert pest_seed_value(AgentObjective.TREAT) == pytest.approx(-0.2)
    assert pest_seed_value(AgentObjective.EXPLORE) == pytest.approx(-1.0)
