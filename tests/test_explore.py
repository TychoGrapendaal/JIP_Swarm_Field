import numpy as np
import pytest

from swarm_field.contracts import (
    BodyType,
    CellId,
    FieldGeometry,
    Idle,
    Message,
    MessageKind,
    PeerObservation,
    RobotSelfState,
    Time,
)
from swarm_field.swarm.behavior import DecisionContext, ExploreBehaviour, FollowField
from swarm_field.swarm.explore import ExploreConfig, revisit_worths
from swarm_field.swarm.memory import AgentMemory

_FIELD = FieldGeometry(width=10, height=10, cell_size=1.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)


def _context(
    current_target: CellId | None,
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
        current_target=current_target,
    )


def test_propose_follows_field_while_previous_target_is_ahead() -> None:
    behavior = ExploreBehaviour("explore")

    (candidate,) = behavior.propose(_context(current_target=(5, 5)))

    assert candidate.action == FollowField("explore")
    assert candidate.target is None


def test_propose_idles_when_previous_target_is_missing() -> None:
    behavior = ExploreBehaviour("explore")

    (candidate,) = behavior.propose(_context(current_target=None))

    assert candidate.action == FollowField("explore")
    assert candidate.target is None
    assert candidate.score == pytest.approx(2 / 3)


def test_propose_idles_when_robot_is_at_previous_target() -> None:
    behavior = ExploreBehaviour("explore")

    (candidate,) = behavior.propose(_context(current_target=(0, 0)))

    assert candidate.action == Idle()
    assert candidate.target == (0, 0)
    assert candidate.score == pytest.approx(2 / 3)


def test_body_type_criterion_returns_non_scout_fraction() -> None:
    behavior = ExploreBehaviour("explore")
    inbox = [
        Message(
            sender=2,
            sent_at=0,
            kind=MessageKind.HELLO,
            payload=PeerObservation(
                cell=(1, 0), confidence=1.0, observed_at=0, source=2, peer=2, body=BodyType.SCOUT
            ),
        ),
        Message(
            sender=3,
            sent_at=0,
            kind=MessageKind.HELLO,
            payload=PeerObservation(
                cell=(2, 0), confidence=1.0, observed_at=0, source=3, peer=3, body=BodyType.KILLER
            ),
        ),
    ]

    assert behavior._body_type_criteria(inbox) == 0.5


def test_update_weight_blends_toward_equal_weights_after_time_advances() -> None:
    behavior = ExploreBehaviour("explore")
    ctx = _context(current_target=None, time=10)

    updated_weights = behavior._update_weight(ctx, [1.0, 0.0, 0.0], previous_time=0)

    assert updated_weights == pytest.approx([13 / 15, 1 / 15, 1 / 15])


def test_explore_score_criterion_uses_target_worth() -> None:
    behavior = ExploreBehaviour("explore")
    memory: AgentMemory = AgentMemory(1, _FIELD)
    memory._last_seen[1, 1] = 0.0
    memory._monitorable[1, 1] = True

    expected = np.mean(
        revisit_worths(
            np.asarray([memory.last_observed_times()], dtype=np.float64),
            np.asarray([memory.monitorable_cells()], dtype=np.bool_),
            80,
            ExploreConfig.revisit_age,
        )
    )

    assert behavior._explore_score_criteria(memory, 80) == pytest.approx(expected)
