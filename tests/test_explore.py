import numpy as np
import pytest

from swarm_field.contracts import (
    BodyType,
    FieldGeometry,
    InfestationObservation,
    Message,
    PeerObservation,
    RobotSelfState,
    Time,
)
from swarm_field.swarm.behavior import DecisionContext, ExploreBehaviour, FollowField
from swarm_field.swarm.explore import ExploreConfig, revisit_worths
from swarm_field.swarm.memory import AgentMemory

_FIELD = FieldGeometry(width=10, height=10, cell_size=1.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)


def _context(
    time: Time = 0,
    inbox: list[Message] | None = None,
) -> DecisionContext:
    memory = AgentMemory(1, _FIELD)
    return DecisionContext(
        time=time,
        self_state=RobotSelfState(
            robot_id=1,
            position=(0, 0),
            battery=1.0,
            body=BodyType.SCOUT,
            speed_mps=1.0,
            consumable=1.0,
            health=1.0,
        ),
        sensed=[],
        memory=memory,
        inbox=inbox or [],
    )


def test_propose_follows_exploration_field() -> None:
    behavior = ExploreBehaviour("explore")

    (candidate,) = behavior.propose(_context())

    assert candidate.action == FollowField("explore")
    assert candidate.target is None


def test_propose_scores_initial_exploration() -> None:
    behavior = ExploreBehaviour("explore")

    (candidate,) = behavior.propose(_context())

    assert candidate.action == FollowField("explore")
    assert candidate.target is None
    assert candidate.score == pytest.approx(0.66)


def test_body_type_criterion_returns_non_scout_fraction() -> None:
    behavior = ExploreBehaviour("explore")
    memory = AgentMemory(1, _FIELD)
    memory.merge(
        [
            PeerObservation(
                cell=(1, 0), confidence=1.0, observed_at=0, source=2, peer=2, body=BodyType.SCOUT
            ),
            PeerObservation(
                cell=(2, 0), confidence=1.0, observed_at=0, source=3, peer=3, body=BodyType.KILLER
            ),
        ]
    )

    assert behavior._body_type_criteria(memory) == 0.5


def test_update_weight_blends_toward_equal_weights_after_time_advances() -> None:
    behavior = ExploreBehaviour("explore")
    ctx = _context(time=10)

    updated_weights = behavior._update_weight(ctx, [1.0, 0.0, 0.0], previous_time=0)

    assert updated_weights == pytest.approx([13 / 15, 1 / 15, 1 / 15])


def test_explore_score_criterion_uses_target_worth() -> None:
    behavior = ExploreBehaviour("explore")
    memory: AgentMemory = AgentMemory(1, _FIELD)
    memory.merge(
        [InfestationObservation((1, 1), 1.0, observed_at=0, source=2, severity=0.0, pest=None)]
    )

    expected = np.mean(
        revisit_worths(
            np.asarray([memory.last_observed_times()], dtype=np.float64),
            np.asarray([memory.monitorable_cells()], dtype=np.bool_),
            80,
            ExploreConfig.revisit_age,
        )
    )

    assert behavior._explore_score_criteria(memory, 80) == pytest.approx(expected)
