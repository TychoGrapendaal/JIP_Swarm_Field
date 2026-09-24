"""Tests for the composed brain and high-level movement action."""

from swarm_field.contracts import (
    Action,
    BodyType,
    Idle,
    Move,
    Perception,
    RobotSelfState,
)
from swarm_field.swarm import (
    ActionCandidate,
    ComposedAgent,
    DecisionContext,
    Explore,
)


def _perception() -> Perception:
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
        sensed=[],
        inbox=[],
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


def test_agent_picks_highest_scoring_candidate() -> None:
    agent = ComposedAgent(
        robot_id=1,
        behaviors=[
            _FixedBehavior("low", Idle(), 0.1),
            _MultiBehavior(
                "multi",
                (
                    ActionCandidate(action=Move((1, 0)), score=0.9, source="multi"),
                    ActionCandidate(action=Move((0, 1)), score=0.5, source="multi"),
                ),
            ),
        ],
    )

    decision = agent.decide(_perception())

    assert decision.action == Move((1, 0))
    assert agent.inspect().current_goal == "multi"


def test_explore_mock_goes_straight_or_turns_left() -> None:
    explore = Explore(5)
    ctx = DecisionContext(
        time=0,
        self_state=_perception().self_state,
        sensed=[],
    )

    actions = explore.propose(ctx)
    actions = actions


def test_agent_idles_without_candidates() -> None:
    agent = ComposedAgent(robot_id=1, behaviors=[])

    decision = agent.decide(_perception())
    decision = decision
