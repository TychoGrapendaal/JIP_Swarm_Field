"""Tests for the composed brain: candidate gathering and selection."""

from random import Random

from swarm_field.contracts import (
    Action,
    BodyType,
    Direction,
    Idle,
    Move,
    Perception,
    RobotSelfState,
    Turn,
)
from swarm_field.swarm import ActionCandidate, ComposedAgent, DecisionContext, Explore


def _perception() -> Perception:
    return Perception(
        time=0,
        self_state=RobotSelfState(
            robot_id=1,
            position=(0, 0),
            facing=Direction.N,
            battery=1.0,
            body=BodyType.SCOUT,
            consumable=1.0,
            health=1.0,
        ),
        sensed=[],
        inbox=[],
    )


class _FixedBehavior:
    """Returns a single candidate."""

    def __init__(self, name: str, action: Action, score: float) -> None:
        self.name: str = name
        self._action: Action = action
        self._score: float = score

    def propose(self, ctx: DecisionContext) -> ActionCandidate:
        return ActionCandidate(action=self._action, score=self._score, source=self.name)


class _MultiBehavior:
    """Returns a tuple of candidates."""

    def __init__(self, name: str, candidates: tuple[ActionCandidate, ...]) -> None:
        self.name: str = name
        self._candidates: tuple[ActionCandidate, ...] = candidates

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        return self._candidates


def test_agent_picks_highest_scoring_candidate() -> None:
    agent = ComposedAgent(
        robot_id=1,
        behaviors=[
            _FixedBehavior("low", Idle(), 0.1),
            _MultiBehavior(
                "multi",
                (
                    ActionCandidate(action=Turn(Direction.N), score=0.9, source="multi"),
                    ActionCandidate(action=Turn(Direction.S), score=0.5, source="multi"),
                ),
            ),
        ],
    )

    decision = agent.decide(_perception())

    assert decision.action == Turn(Direction.N)
    assert agent.inspect().current_goal == "multi"


def test_explore_mock_goes_straight_or_turns_left() -> None:
    explore = Explore(Random(0))
    ctx = DecisionContext(
        time=0,
        self_state=RobotSelfState(
            robot_id=1,
            position=(0, 0),
            facing=Direction.N,
            battery=1.0,
            body=BodyType.SCOUT,
            consumable=1.0,
            health=1.0,
        ),
        sensed=[],
    )

    actions: set[Action] = set()
    for _ in range(20):
        candidates = explore.propose(ctx)
        assert len(candidates) == 1
        actions.add(candidates[0].action)

    # Straight ahead is Move(direction=facing); 45 degrees left of North is North-West.
    assert actions <= {Move(direction=Direction.N), Turn(Direction.NW)}
    assert Move(direction=Direction.N) in actions
    assert Turn(Direction.NW) in actions


def test_agent_idles_without_candidates() -> None:
    agent = ComposedAgent(robot_id=1, behaviors=[])

    decision = agent.decide(_perception())

    assert decision.action == Idle()
    assert agent.inspect().current_goal is None
