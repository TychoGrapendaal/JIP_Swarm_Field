"""Tests for hop-based pest navigation: beacons, the killer's behaviour and the experiment."""

from dataclasses import replace

import pytest

from swarm_field.contracts import (
    AgentObjective,
    BodyType,
    FieldGeometry,
    InfestationObservation,
    Message,
    MessageKind,
    PeerObservation,
    Perception,
    PestBeacon,
    PestId,
    RobotSelfState,
    Time,
    TimeUnit,
    Treat,
)
from swarm_field.experiments import HopPestScenario, LiveRun
from swarm_field.swarm import (
    AgentMemory,
    BeaconSeeder,
    BeaconTable,
    DecisionContext,
    FieldProvider,
    FollowBeacon,
    FollowField,
    HopAgent,
    HopConfig,
    Navigator,
    UtilityArbiter,
    chosen_beacon,
)

_FIELD = FieldGeometry(width=10, height=10, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)
_PEST = PestId(finder=7, number=1)


def _beacon(hops: int, cell: tuple[int, int] = (5, 5), pest: PestId = _PEST) -> PestBeacon:
    return PestBeacon(pest=pest, cell=cell, hops=hops, found_at=0)


def _self_state(position: tuple[int, int] = (0, 0), robot_id: int = 1) -> RobotSelfState:
    return RobotSelfState(
        robot_id=robot_id,
        position=position,
        battery=1.0,
        body=BodyType.KILLER,
        speed_mps=0.5,
        consumable=1.0,
        health=1.0,
    )


def _pest_seen(cell: tuple[int, int], severity: float, time: Time = 0) -> InfestationObservation:
    return InfestationObservation(
        cell=cell, confidence=1.0, observed_at=time, source=1, severity=severity, pest=None
    )


def _hop_agent(robot_id: int = 1) -> HopAgent:
    memory = AgentMemory(robot_id, _FIELD)
    beacons = BeaconTable(robot_id, HopConfig())
    fields = FieldProvider(memory, flies=False, seeders=[BeaconSeeder(beacons, robot_id)])
    return HopAgent(
        robot_id=robot_id,
        behaviors=[FollowBeacon(beacons)],
        memory=memory,
        fields=fields,
        navigator=Navigator(fields.quadtree),
        arbiter=UtilityArbiter(),
        beacons=beacons,
    )


def test_a_relayed_beacon_is_one_hop_further_and_the_fewest_hops_are_kept() -> None:
    table = BeaconTable(1, HopConfig())

    table.receive(_beacon(hops=4), now=0)
    table.receive(_beacon(hops=1), now=0)
    table.receive(_beacon(hops=6), now=0)

    assert [b.hops for b in table.live(0)] == [2]


def test_a_beacon_beyond_the_hop_limit_or_its_lifetime_is_dropped() -> None:
    config = HopConfig(max_hops=3, lifetime=10 * TimeUnit.MINUTE)
    table = BeaconTable(1, config)

    table.receive(_beacon(hops=3), now=0)
    assert table.live(0) == []

    table.receive(_beacon(hops=0), now=0)
    assert len(table.live(0)) == 1
    assert table.live(11 * TimeUnit.MINUTE) == []


def test_a_cleared_beacon_wins_and_is_never_undone() -> None:
    table = BeaconTable(1, HopConfig())
    table.receive(_beacon(hops=0), now=0)

    table.receive(replace(_beacon(hops=9), cleared=True), now=0)
    table.receive(_beacon(hops=0), now=0)

    assert table.live(0) == []
    (sent,) = table.to_send(0)
    assert sent.cleared


def test_a_robot_numbers_the_pests_it_finds_and_does_not_restart_a_live_beacon() -> None:
    table = BeaconTable(3, HopConfig())

    table.found((2, 2), now=0)
    table.found((2, 2), now=TimeUnit.MINUTE)
    table.found((4, 4), now=TimeUnit.MINUTE)

    pests = sorted((b.pest.number, b.cell, b.hops) for b in table.live(TimeUnit.MINUTE))
    assert pests == [(1, (2, 2), 0), (2, (4, 4), 0)]


def test_a_killer_follows_the_unclaimed_beacon_before_a_nearer_claimed_one() -> None:
    table = BeaconTable(1, HopConfig())
    near = PestId(finder=8, number=1)
    far = PestId(finder=9, number=1)
    table.receive(_beacon(hops=0, cell=(2, 2), pest=near), now=0)
    table.receive(_beacon(hops=5, cell=(8, 8), pest=far), now=0)
    memory = AgentMemory(1, _FIELD)

    chosen = chosen_beacon(table, memory, robot=1, now=0)
    assert chosen is not None
    assert chosen.pest == near

    peer = PeerObservation(
        cell=(1, 1),
        confidence=1.0,
        observed_at=0,
        source=2,
        peer=2,
        body=BodyType.KILLER,
        target=(2, 2),
        objective=AgentObjective.TREAT,
    )
    memory.merge([peer])
    chosen = chosen_beacon(table, memory, robot=1, now=0)
    assert chosen is not None
    assert chosen.pest == far


def test_follow_beacon_bids_the_pest_id_as_its_target_and_treats_on_the_spot() -> None:
    table = BeaconTable(1, HopConfig())
    table.receive(_beacon(hops=2), now=0)
    memory = AgentMemory(1, _FIELD)
    behaviour = FollowBeacon(table)

    ctx = DecisionContext(time=0, self_state=_self_state(), sensed=[], memory=memory)
    (bid,) = behaviour.propose(ctx)
    assert bid.action == FollowField("beacon")
    assert bid.target == _PEST
    assert bid.objective is AgentObjective.TREAT
    assert bid.score == pytest.approx(0.8 - 0.01 * 3)

    memory.merge([_pest_seen((5, 5), 0.5)])
    here = DecisionContext(time=0, self_state=_self_state((5, 5)), sensed=[], memory=memory)
    (treat,) = behaviour.propose(here)
    assert treat.action == Treat((5, 5))


def test_follow_beacon_bids_nothing_without_a_live_beacon() -> None:
    behaviour = FollowBeacon(BeaconTable(1, HopConfig()))
    ctx = DecisionContext(
        time=0, self_state=_self_state(), sensed=[], memory=AgentMemory(1, _FIELD)
    )

    assert behaviour.propose(ctx) == ()


def test_a_hop_agent_starts_a_beacon_for_a_sensed_pest_and_passes_it_on() -> None:
    agent = _hop_agent()
    perception = Perception(
        time=0, self_state=_self_state(), sensed=[_pest_seen((4, 4), 0.6)], inbox=[]
    )

    decision = agent.decide(perception)

    beacon_messages = [m for m in decision.outbox if m.kind is MessageKind.BEACON]
    assert len(beacon_messages) == 1
    payload = beacon_messages[0].payload
    assert isinstance(payload, tuple)
    assert [(b.cell, b.hops) for b in payload] == [((4, 4), 0)]


def test_a_hop_agent_learns_pests_only_from_beacons() -> None:
    agent = _hop_agent()
    gossip = Message(
        sender=2, sent_at=0, kind=MessageKind.KNOWLEDGE, payload=[_pest_seen((6, 6), 1.0)]
    )
    beacon = Message(sender=3, sent_at=0, kind=MessageKind.BEACON, payload=(_beacon(hops=0),))
    perception = Perception(time=0, self_state=_self_state(), sensed=[], inbox=[gossip, beacon])

    agent.decide(perception)

    assert agent.memory.infestation_at((6, 6)) is None
    assert [(b.cell, b.hops) for b in agent.beacons.live(0)] == [((5, 5), 1)]


def test_the_hop_experiment_runs_with_a_hop_agent_for_every_robot() -> None:
    scenario = HopPestScenario(size=16, scouts=3, killers=2, until=20 * TimeUnit.MINUTE)
    run = LiveRun.from_config(scenario.into_experiment())

    run.advance_to(scenario.until)

    agents = run.agents()
    assert len(agents) == 5
    assert all(isinstance(agent, HopAgent) for agent in agents.values())


def test_the_memory_navigation_is_the_plain_pest_experiment() -> None:
    scenario = HopPestScenario(navigation="memory", size=16, scouts=2, killers=1)
    config = scenario.into_experiment()

    assert config.brains_by_body[BodyType.KILLER].behaviors == ("treat-infestation", "explore")
    with pytest.raises(ValueError):
        HopPestScenario(navigation="radio").into_experiment()
