"""Tests for hop-gradient navigation: counts, rounds, parents, anchors and the killer."""

from dataclasses import replace

from swarm_field.contracts import (
    AgentObjective,
    BodyType,
    FieldGeometry,
    HopClaim,
    HopGradient,
    Idle,
    InfestationObservation,
    PeerObservation,
    PestId,
    RobotSelfState,
    TimeUnit,
    Treat,
)
from swarm_field.experiments import HopPestScenario, LiveRun
from swarm_field.swarm import (
    AgentMemory,
    AnchorPest,
    DecisionContext,
    FollowField,
    FollowGradient,
    GradientAgent,
    GradientConfig,
    GradientSeeder,
    GradientTable,
    chosen_gradient,
)
from swarm_field.swarm.priority import ranks_before

_FIELD = FieldGeometry(width=10, height=10, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)
_PEST = PestId(finder=5, number=1)
_SECOND = TimeUnit.SECOND


def _self_state(
    position: tuple[int, int] = (0, 0), body: BodyType = BodyType.KILLER
) -> RobotSelfState:
    return RobotSelfState(
        robot_id=1,
        position=position,
        battery=1.0,
        body=body,
        speed_mps=0.5,
        consumable=1.0,
        health=1.0,
    )


def _peer_at(peer: int, cell: tuple[int, int]) -> PeerObservation:
    return PeerObservation(
        cell=cell, confidence=1.0, observed_at=0, source=peer, peer=peer, body=BodyType.SCOUT
    )


def test_a_count_is_one_more_than_the_senders_and_remembers_the_sender_as_parent() -> None:
    table = GradientTable(1, GradientConfig())

    table.receive(HopGradient(_PEST, round=0, hops=3), sender=4, now=0)
    table.receive(HopGradient(_PEST, round=0, hops=1), sender=2, now=0)
    table.receive(HopGradient(_PEST, round=0, hops=5), sender=6, now=0)

    (entry,) = table.live(0)
    assert (entry.hops, entry.parent) == (2, 2)


def test_a_newer_round_wins_even_with_more_hops_so_the_gradient_follows_the_swarm() -> None:
    table = GradientTable(1, GradientConfig())
    table.receive(HopGradient(_PEST, round=0, hops=1), sender=2, now=0)

    table.receive(HopGradient(_PEST, round=1, hops=4), sender=3, now=15 * _SECOND)

    (entry,) = table.live(15 * _SECOND)
    assert (entry.round, entry.hops, entry.parent) == (1, 5, 3)


def test_a_gradient_whose_anchor_stopped_fades_and_is_not_learned_back() -> None:
    config = GradientConfig(max_age=60 * _SECOND)
    table = GradientTable(1, config)
    table.receive(HopGradient(_PEST, round=7, hops=0), sender=5, now=0)

    assert table.live(61 * _SECOND) == []
    # A neighbour still passing on round 7 must not bring the dead gradient back.
    table.receive(HopGradient(_PEST, round=7, hops=2), sender=3, now=61 * _SECOND)
    assert table.live(61 * _SECOND) == []


def test_a_cleared_count_wins_and_drops_the_pest() -> None:
    table = GradientTable(1, GradientConfig())
    table.receive(HopGradient(_PEST, round=0, hops=1), sender=2, now=0)

    table.receive(HopGradient(_PEST, round=0, hops=0, cleared=True), sender=3, now=0)
    table.receive(HopGradient(_PEST, round=1, hops=0), sender=2, now=0)

    assert table.live(0) == []
    assert table.cleared(0) == [_PEST]


def test_an_anchor_sends_hop_zero_in_a_new_round_at_every_decision() -> None:
    table = GradientTable(5, GradientConfig())

    first = table.anchor((3, 3), now=0)
    second = table.anchor((3, 3), now=15 * _SECOND)

    assert first == second == _PEST
    (entry,) = table.live(15 * _SECOND)
    assert (entry.hops, entry.round, entry.parent) == (0, 1, None)


def test_a_killer_leaves_a_pest_to_a_nearer_killer_and_takes_another() -> None:
    table = GradientTable(1, GradientConfig())
    near = PestId(finder=8, number=1)
    far = PestId(finder=9, number=1)
    table.receive(HopGradient(near, round=0, hops=0), sender=8, now=0)
    table.receive(HopGradient(far, round=0, hops=4), sender=2, now=0)

    entry = chosen_gradient(table, 0)
    assert entry is not None
    assert entry.pest == near

    # A killer farther from the pest does not take it away.
    table.receive_claim(HopClaim(near, killer=6, hops=3, sent_at=0), now=0)
    entry = chosen_gradient(table, 0)
    assert entry is not None
    assert entry.pest == near

    # A killer as near and first in the order does.
    rival = 0
    while not ranks_before(rival, 1, 0):
        rival += 2
    table.receive_claim(HopClaim(near, killer=rival, hops=1, sent_at=0), now=0)
    entry = chosen_gradient(table, 0)
    assert entry is not None
    assert entry.pest == far


def test_the_killer_bids_to_drive_to_its_parent_and_treats_on_the_spot() -> None:
    table = GradientTable(1, GradientConfig())
    table.receive(HopGradient(_PEST, round=0, hops=1), sender=3, now=0)
    memory = AgentMemory(1, _FIELD)
    memory.merge([_peer_at(3, (6, 2))])
    behaviour = FollowGradient(table)

    ctx = DecisionContext(time=0, self_state=_self_state(), sensed=[], memory=memory)
    (bid,) = behaviour.propose(ctx)
    assert bid.action == FollowField("gradient")
    assert bid.target == _PEST
    assert bid.objective is AgentObjective.TREAT
    (seed,) = GradientSeeder(table).seeds(memory, (0, 0), 0)
    assert seed.cell == (6, 2)

    pest = InfestationObservation(
        cell=(6, 2), confidence=1.0, observed_at=0, source=1, severity=0.5, pest=None
    )
    memory.merge([pest])
    here = DecisionContext(time=0, self_state=_self_state((6, 2)), sensed=[], memory=memory)
    (treat,) = behaviour.propose(here)
    assert treat.action == Treat((6, 2))


def test_a_scout_stays_on_the_pest_it_anchors_until_the_anchor_time_is_up() -> None:
    config = GradientConfig(anchor_time=10 * TimeUnit.MINUTE)
    table = GradientTable(1, config)
    memory = AgentMemory(1, _FIELD)
    pest = InfestationObservation(
        cell=(2, 2), confidence=1.0, observed_at=0, source=1, severity=0.5, pest=None
    )
    memory.merge([pest])
    table.anchor((2, 2), now=0)
    behaviour = AnchorPest(table)
    state = _self_state((2, 2), BodyType.SCOUT)

    (stay,) = behaviour.propose(DecisionContext(time=0, self_state=state, sensed=[], memory=memory))
    assert stay.action == Idle()
    late = DecisionContext(time=10 * TimeUnit.MINUTE, self_state=state, sensed=[], memory=memory)
    assert behaviour.propose(late) == ()


def test_the_gradient_experiment_runs_with_a_gradient_agent_for_every_robot() -> None:
    scenario = HopPestScenario(
        navigation="gradient", size=16, scouts=3, killers=1, until=20 * TimeUnit.MINUTE
    )
    run = LiveRun.from_config(scenario.into_experiment())

    run.advance_to(scenario.until)

    assert all(isinstance(agent, GradientAgent) for agent in run.agents().values())


def _anchors(robot_id: int, other: int, other_anchoring: bool, since: int) -> bool:
    """Whether ``robot_id``, on a pest since ``since``, stays with ``other`` on the cell at 60 s."""
    now = 60 * _SECOND
    table = GradientTable(robot_id, GradientConfig())
    memory = AgentMemory(robot_id, _FIELD)
    pest = InfestationObservation(
        cell=(2, 2), confidence=1.0, observed_at=now, source=robot_id, severity=0.5, pest=None
    )
    objective = None
    if other_anchoring:
        objective = AgentObjective.VERIFY
    peer = PeerObservation(
        cell=(2, 2),
        confidence=1.0,
        observed_at=now,
        source=robot_id,
        peer=other,
        body=BodyType.SCOUT,
        objective=objective,
    )
    memory.merge([pest, peer])
    table.anchor((2, 2), now=since)
    state = replace(_self_state((2, 2), BodyType.SCOUT), robot_id=robot_id)
    ctx = DecisionContext(time=now, self_state=state, sensed=[], memory=memory)
    return AnchorPest(table).propose(ctx) != ()


def test_of_robots_arriving_on_one_pest_together_only_the_first_in_order_anchors() -> None:
    now = 60 * _SECOND
    first, later = 3, 5
    if not ranks_before(first, later, now):
        first, later = later, first

    assert _anchors(first, later, other_anchoring=False, since=now)
    assert not _anchors(later, first, other_anchoring=False, since=now)


def test_the_robot_already_anchoring_keeps_the_pest_against_a_newcomer() -> None:
    now = 60 * _SECOND
    first, later = 3, 5
    if not ranks_before(first, later, now):
        first, later = later, first

    # Whatever the order, the newcomer leaves and the robot anchoring since 0 stays.
    assert not _anchors(first, later, other_anchoring=True, since=now)
    assert _anchors(later, first, other_anchoring=False, since=0)
