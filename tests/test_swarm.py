"""Tests for the composed brain and high-level movement action."""

from collections.abc import Hashable

from swarm_field.contracts import (
    BodyType,
    CellId,
    FieldGeometry,
    Idle,
    InfestationObservation,
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
    Behavior,
    ComposedAgent,
    DecisionContext,
    FieldProvider,
    FieldSeeder,
    FollowField,
    Intent,
    MemoryView,
    Navigator,
    TreatInfestation,
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


def _agent(behaviors: list[Behavior], seeders: list[FieldSeeder] | None = None) -> ComposedAgent:
    memory = AgentMemory(1, _FIELD)
    fields = FieldProvider(memory, flies=False, seeders=seeders or [])
    return ComposedAgent(
        robot_id=1,
        behaviors=behaviors,
        memory=memory,
        fields=fields,
        navigator=Navigator(fields.quadtree),
    )


def _context(
    sensed: list[Observation] | None = None, seeders: list[FieldSeeder] | None = None
) -> DecisionContext:
    memory = AgentMemory(1, _FIELD)
    perception = _perception(sensed)
    return DecisionContext(
        time=perception.time,
        self_state=perception.self_state,
        sensed=perception.sensed,
        memory=memory,
        fields=FieldProvider(memory, flies=False, seeders=seeders or []),
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

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        return None

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        return (Seed((5, 5), 0.0),)


class _CountingSeeder:
    """A seeder with one fixed target, counting how often it is asked for seeds."""

    def __init__(self, name: str, target: CellId) -> None:
        self.name: str = name
        self.seeds_asked: int = 0
        self._target: CellId = target

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        return None

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        self.seeds_asked += 1
        return (Seed(self._target, 0.0),)


class _FieldFollower:
    """Reads its seeder's field at every decision point and bids a fixed score to follow it."""

    def __init__(self, seeder: str, score: float) -> None:
        self.name: str = seeder
        self._score: float = score

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        position = ctx.self_state.position
        seeded = ctx.fields.field_for(self.name, position, ctx.time)
        assert seeded is not None
        target = seeded.field.target_for(position)
        return (ActionCandidate(FollowField(self.name), self._score, self.name, target),)


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


def test_treat_infestation_bids_to_follow_its_seeders_field() -> None:
    behavior = TreatInfestation(_FixedTargetSeeder.name)

    candidates = behavior.propose(_context(sensed=[], seeders=[_FixedTargetSeeder()]))

    assert [(c.action, c.target) for c in candidates] == [(FollowField("fixed-target"), None)]


def test_a_winning_follow_field_becomes_a_step_along_the_field() -> None:
    agent = _agent([TreatInfestation(_FixedTargetSeeder.name)], seeders=[_FixedTargetSeeder()])

    decision = agent.decide(_perception(sensed=[]))

    assert decision.action == Move((1, 1))
    assert agent.inspect().current_goal == TreatInfestation.name


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


def test_the_winners_field_is_followed_without_rebuilding_either_field() -> None:
    """Two behaviours read their fields at every decision point; each field is built once."""
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
    assert loser.seeds_asked == 1
