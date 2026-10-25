import pytest

from swarm_field.contracts import (
    BodyType,
    FieldGeometry,
    InfestationObservation,
    Message,
    Observation,
    PeerObservation,
    RobotSelfState,
    Time,
)
from swarm_field.swarm.behavior import DecisionContext, ExploreBehaviour, FollowField
from swarm_field.swarm.explore import ExploreConfig
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


def test_the_non_scout_criterion_is_the_share_of_peers_that_are_not_scouts() -> None:
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

    assert ExploreBehaviour._non_scout_share(memory) == 0.5


def test_the_revisit_criterion_is_the_mean_worth_at_the_configured_revisit_age() -> None:
    memory: AgentMemory = AgentMemory(1, _FIELD)
    seen: list[Observation] = []
    for x in range(_FIELD.width):
        for y in range(_FIELD.height):
            seen.append(InfestationObservation((x, y), 1.0, 0, source=2, severity=0.0, pest=None))
    memory.merge(seen)

    stale_after_40 = ExploreBehaviour("explore", ExploreConfig(revisit_age=40))

    assert stale_after_40._mean_revisit_worth(memory, 80) == pytest.approx(0.75)
    assert ExploreBehaviour("explore")._mean_revisit_worth(memory, 80) == 0.0
