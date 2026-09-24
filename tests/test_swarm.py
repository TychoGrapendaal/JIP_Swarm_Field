"""Tests for the composed brain and high-level movement action."""

from random import Random

from swarm_field.contracts import (
    Action,
    BodyType,
    FieldGeometry,
    Idle,
    Move,
    Observation,
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
    Explore,
    MemoryView,
    Navigator,
    TreatInfestation,
)
from swarm_field.swarm.field import Seed

_FIELD = FieldGeometry(width=10, height=10, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)


def _perception(sensed: list[Observation] | None = None) -> Perception:
    return Perception(
        time=0,
        self_state=RobotSelfState(
            robot_id=1,
            position=(0, 0),
            battery=1.0,
            body=BodyType.SCOUT,
            speed_mps=2.0,
            consumable=1.0,
            health=1.0,
        ),
        sensed=sensed or [],
        inbox=[],
    )


def _agent(behaviors: list[Behavior]) -> ComposedAgent:
    memory = AgentMemory(1, _FIELD)
    return ComposedAgent(
        robot_id=1,
        behaviors=behaviors,
        memory=memory,
        navigator=Navigator(memory, flies=False),
    )


def _context() -> DecisionContext:
    memory = AgentMemory(1, _FIELD)
    return DecisionContext(
        time=0,
        self_state=_perception().self_state,
        sensed=[],
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


def test_explore_mock_proposes_direct_target_moves() -> None:
    explore = Explore(Random(0))
    ctx = _context()

    actions = {explore.propose(ctx)[0].action for _ in range(20)}

    assert actions <= {Move((1, 0)), Move((0, 1)), Move((-1, 0)), Move((0, -1))}
    assert len(actions) > 1


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

    candidates = behavior.propose(_context())

    assert [candidate.action for candidate in candidates] == [Move((1, 1))]
