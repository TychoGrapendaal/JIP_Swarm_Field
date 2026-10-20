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
