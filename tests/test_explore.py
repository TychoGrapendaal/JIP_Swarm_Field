from collections.abc import Hashable

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
from swarm_field.swarm.field import Seed
from swarm_field.swarm.memory import AgentMemory, MemoryView
from swarm_field.swarm.provider import FieldProvider

_FIELD = FieldGeometry(width=10, height=10, cell_size=1.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)


class _ExploreSeeder:
    name: str = "explore"
    window_blocks: int = 1

    def __init__(self, target: CellId | None) -> None:
        self._target: CellId | None = target

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        return None

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        if self._target is None:
            return ()
        return (Seed(self._target, 0.0),)


def _context(
    target: CellId | None,
    time: Time = 0,
    inbox: list[Message] | None = None,
) -> DecisionContext:
    memory = AgentMemory(1, _FIELD)
    fields = FieldProvider(memory, flies=False, seeders=[_ExploreSeeder(target)])
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
        fields=fields,
        inbox=inbox or [],
    )


def test_propose_follows_available_exploration_field() -> None:
    behavior = ExploreBehaviour("explore")

    (candidate,) = behavior.propose(_context(target=(5, 5)))

    assert candidate.action == FollowField("explore")
    assert candidate.target == (5, 5)


def test_propose_scores_and_idles_when_field_is_missing() -> None:
    behavior = ExploreBehaviour("explore")

    (candidate,) = behavior.propose(_context(target=None))

    assert candidate.action == Idle()
    assert candidate.target is None
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
    ctx = _context(target=None, time=10)

    updated_weights = behavior._update_weight(ctx, [1.0, 0.0, 0.0], previous_time=0)

    assert updated_weights == pytest.approx([13 / 15, 1 / 15, 1 / 15])


def test_explore_score_criterion_uses_target_worth() -> None:
    behavior = ExploreBehaviour("explore")
    memory: AgentMemory = AgentMemory(1, _FIELD)
    memory._last_seen[1, 1] = 0.0
    memory._monitorable[1, 1] = True

    expected = revisit_worths(
        np.asarray([[memory.last_observed_times()]], dtype=np.float64),
        np.asarray([[memory.monitorable_cells()]], dtype=np.bool_),
        80,
        ExploreConfig.revisit_age,
    )[0, 0]

    assert behavior._explore_score_criteria(memory, 80) == pytest.approx(expected)
