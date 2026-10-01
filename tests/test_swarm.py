"""Tests for the composed brain and high-level movement action."""

from swarm_field.contracts import (
    Action,
    BodyType,
    FieldGeometry,
    Idle,
    InfestationObservation,
    Move,
    Observation,
    PeerObservation,
    Perception,
    RobotSelfState,
    Terrain,
    TerrainObservation,
    Time,
)
from swarm_field.swarm import (
    ActionCandidate,
    AgentMemory,
    Behavior,
    ComposedAgent,
    DecisionContext,
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


def _agent(behaviors: list[Behavior]) -> ComposedAgent:
    memory = AgentMemory(1, _FIELD)
    return ComposedAgent(
        robot_id=1,
        behaviors=behaviors,
        memory=memory,
        navigator=Navigator(memory, flies=False),
    )


def _context(sensed: list[Observation] | None = None) -> DecisionContext:
    memory = AgentMemory(1, _FIELD)
    perception = _perception(sensed)
    return DecisionContext(
        time=perception.time,
        self_state=perception.self_state,
        sensed=perception.sensed,
        memory=memory,
        navigator=Navigator(memory, flies=False),
    )


class _FixedBehavior:
    def __init__(self, name: str, action: Action, score: float) -> None:
        self.name: str = name
        self._action: Action = action
        self._score: float = score

    def propose(self, ctx: DecisionContext) -> ActionCandidate:
        return ActionCandidate(action=self._action, score=self._score, source=self.name)


class _MultiBehavior:
    def __init__(self, name: str, candidates: tuple[ActionCandidate, ...]) -> None:
        self.name: str = name
        self._candidates: tuple[ActionCandidate, ...] = candidates

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        return self._candidates


class _FixedTargetObjective:
    name: str = "fixed-target"

    def seeds(self, memory: MemoryView, now: Time) -> tuple[Seed, ...]:
        return (Seed((5, 5), 0.0),)


def test_agent_picks_highest_scoring_candidate() -> None:
    agent = _agent(
        [
            _FixedBehavior("low", Idle(), 0.1),
            _MultiBehavior(
                "multi",
                (
                    ActionCandidate(action=Move((1, 0)), score=0.9, source="multi"),
                    ActionCandidate(action=Move((0, 1)), score=0.5, source="multi"),
                ),
            ),
        ]
    )

    decision = agent.decide(_perception())

    assert decision.action == Move((1, 0))
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
    assert not agent.navigator.quadtree.is_traversable((1, 1))


def test_treat_infestation_follows_its_objectives_field() -> None:
    behavior = TreatInfestation(_FixedTargetObjective())

    candidates = behavior.propose(_context(sensed=[]))

    assert [candidate.action for candidate in candidates] == [Move((1, 1))]
