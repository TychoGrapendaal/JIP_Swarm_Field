"""Tests for the hop field and the pest response on hop beacons."""

from swarm_field.contracts import (
    AgentObjective,
    Beacon,
    BodyType,
    CellId,
    FieldGeometry,
    InfestationObservation,
    Message,
    MessageKind,
    Move,
    Observation,
    PeerObservation,
    Perception,
    RobotSelfState,
    Treat,
)
from swarm_field.swarm import (
    AgentMemory,
    BeaconConfig,
    ComposedAgent,
    DecisionContext,
    FieldProvider,
    FollowField,
    HopField,
    KnowledgeSharing,
    Navigator,
    TreatByBeacon,
    UtilityArbiter,
    treat_beacon,
)

_FIELD = FieldGeometry(width=10, height=10, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)
_CONFIG = BeaconConfig()


def _memory_hearing(*beacons: tuple[float, CellId]) -> AgentMemory:
    memory = AgentMemory(1, _FIELD)
    for sender, (value, cell) in enumerate(beacons, start=2):
        memory.hear_beacons(sender, 0, [Beacon(AgentObjective.TREAT, value, cell)])
    return memory


def _self_state(position: CellId) -> RobotSelfState:
    return RobotSelfState(
        robot_id=1,
        position=position,
        battery=1.0,
        body=BodyType.KILLER,
        speed_mps=1.0,
        consumable=1.0,
        health=1.0,
    )


def test_only_the_lowest_beacons_pull_the_robot() -> None:
    memory = _memory_hearing((1.0, (5, 0)), (2.0, (0, 5)), (3.0, (0, 0)), (3.0, (2, 2)))

    heading = HopField(_CONFIG).guidance_for(memory, (0, 0), 0)

    assert heading is not None
    assert heading.offset == (5.0, 0.0)
    assert heading.target == (5, 0)


def test_beacons_at_the_same_value_pull_together() -> None:
    memory = _memory_hearing((1.0, (4, 0)), (1.0, (0, 4)))

    heading = HopField(_CONFIG).guidance_for(memory, (0, 0), 0)

    assert heading is not None
    assert heading.offset[0] > 0
    assert heading.offset[0] == heading.offset[1]


def _bids(memory: AgentMemory, position: CellId, sensed: list[Observation]) -> list[object]:
    ctx = DecisionContext(time=0, self_state=_self_state(position), sensed=sensed, memory=memory)
    return [candidate.action for candidate in TreatByBeacon(_CONFIG).propose(ctx)]


def test_a_robot_on_a_pest_treats_it_and_bids_on_beacons_otherwise() -> None:
    memory = _memory_hearing((1.0, (5, 0)))
    pest = InfestationObservation(
        cell=(3, 3), confidence=1.0, observed_at=0, source=1, severity=0.5, pest=None
    )

    assert _bids(memory, (3, 3), [pest]) == [Treat((3, 3))]
    assert _bids(memory, (0, 0), []) == [FollowField(HopField.name)]
    assert _bids(AgentMemory(1, _FIELD), (0, 0), []) == []


def test_a_robot_does_not_bid_on_a_beacon_on_its_own_cell() -> None:
    cleared_pest = _memory_hearing((0.0, (3, 3)))

    assert _bids(cleared_pest, (3, 3), []) == []


def test_a_hop_killer_steps_towards_the_beacon_it_hears() -> None:
    memory = AgentMemory(1, _FIELD)
    fields = FieldProvider(memory, flies=False, fields=[HopField(_CONFIG)])
    killer = ComposedAgent(
        robot_id=1,
        behaviors=[TreatByBeacon(_CONFIG)],
        memory=memory,
        fields=fields,
        navigator=Navigator(fields.quadtree),
        arbiter=UtilityArbiter(),
        sharing=KnowledgeSharing.NOTHING,
        beacons=_CONFIG,
    )
    beacon = Message(
        sender=2,
        sent_at=0,
        kind=MessageKind.BEACON,
        payload=(Beacon(AgentObjective.TREAT, 0.0, (5, 0)),),
    )

    decision = killer.decide(
        Perception(time=0, self_state=_self_state((0, 0)), sensed=[], inbox=[beacon])
    )

    assert decision.action == Move((1, 0))
    assert killer.inspect().current_target == (5, 0)


def test_with_metric_hops_a_small_gamma_follows_the_lowest_beacon() -> None:
    memory = _memory_hearing((1.0, (0, 4)), (2.0, (4, 0)))
    near_literal = BeaconConfig(gamma=0.01, hop_cells=2.0)

    heading = HopField(near_literal).guidance_for(memory, (0, 0), 0)

    assert heading is not None
    assert heading.target == (0, 4)
    assert heading.offset[1] > 0
    assert abs(heading.offset[0]) < 0.1


def test_with_metric_hops_a_robot_on_a_relay_follows_the_relays_downhill_beacons() -> None:
    memory = _memory_hearing((2.0, (0, 0)), (1.5, (3, 0)))

    heading = HopField(BeaconConfig(hop_cells=2.0)).guidance_for(memory, (0, 0), 0)

    assert heading is not None
    assert heading.target == (3, 0)


def test_two_robots_on_one_cell_do_not_hold_each_others_value() -> None:
    memory = _memory_hearing((3.0, (0, 0)))

    heading = HopField(BeaconConfig(hop_cells=2.0)).guidance_for(memory, (0, 0), 0)
    beacon = treat_beacon(memory, 1, (0, 0), 0, BeaconConfig(hop_cells=2.0))

    assert heading is None
    assert beacon == Beacon(AgentObjective.TREAT, 3.5, (0, 0))


def test_beacons_above_the_cap_are_ignored_and_not_relayed() -> None:
    capped = BeaconConfig(max_value=4.0)

    assert treat_beacon(_memory_hearing((4.5, (1, 0))), 1, (0, 0), 0, capped) is None
    assert treat_beacon(_memory_hearing((3.5, (1, 0))), 1, (0, 0), 0, capped) is None
    assert treat_beacon(_memory_hearing((3.0, (1, 0))), 1, (0, 0), 0, capped) == Beacon(
        AgentObjective.TREAT, 4.0, (0, 0)
    )


def _memory_with_a_killer_east() -> AgentMemory:
    memory = _memory_hearing((1.0, (0, 4)), (1.0, (4, 0)))
    killer = PeerObservation(
        cell=(4, 0), confidence=1.0, observed_at=0, source=3, peer=3, body=BodyType.KILLER
    )
    memory.merge([killer])
    return memory


def test_a_killer_ahead_counters_the_pull_of_its_level() -> None:
    config = BeaconConfig(killer_repulsion=1.0)

    heading = HopField(config).guidance_for(_memory_with_a_killer_east(), (0, 0), 0)

    assert heading is not None
    assert heading.target == (0, 4)
    assert heading.offset[0] < 0


def test_without_repulsion_a_killers_beacon_pulls_like_any_other() -> None:
    heading = HopField(_CONFIG).guidance_for(_memory_with_a_killer_east(), (0, 0), 0)

    assert heading is not None
    assert heading.offset[0] == heading.offset[1]


def test_a_robot_does_not_bid_when_only_a_killer_is_downhill() -> None:
    memory = _memory_hearing((1.0, (4, 0)))
    killer = PeerObservation(
        cell=(4, 0), confidence=1.0, observed_at=0, source=2, peer=2, body=BodyType.KILLER
    )
    memory.merge([killer])
    ctx = DecisionContext(time=0, self_state=_self_state((0, 0)), sensed=[], memory=memory)

    assert TreatByBeacon(BeaconConfig(killer_repulsion=1.0)).propose(ctx) == ()
    assert TreatByBeacon(_CONFIG).propose(ctx) != ()
