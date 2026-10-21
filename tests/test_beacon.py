"""Tests for hop beacons: sources, relaying and freshness."""

from swarm_field.contracts import (
    Beacon,
    BeaconKind,
    BodyType,
    CellId,
    FieldGeometry,
    InfestationObservation,
    Message,
    MessageKind,
    Observation,
    Perception,
    RobotSelfState,
    Time,
    TimeUnit,
)
from swarm_field.swarm import (
    AgentMemory,
    BeaconConfig,
    ComposedAgent,
    FieldProvider,
    KnowledgeSharing,
    Navigator,
    UtilityArbiter,
)

_FIELD = FieldGeometry(width=10, height=10, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)


def _relay(robot_id: int) -> ComposedAgent:
    memory = AgentMemory(robot_id, _FIELD)
    fields = FieldProvider(memory, flies=False, seeders=[])
    return ComposedAgent(
        robot_id=robot_id,
        behaviors=[],
        memory=memory,
        fields=fields,
        navigator=Navigator(fields.quadtree),
        arbiter=UtilityArbiter(),
        sharing=KnowledgeSharing.NOTHING,
        beacons=BeaconConfig(),
    )


def _perception(
    robot_id: int,
    position: CellId,
    time: Time = 0,
    inbox: list[Message] | None = None,
    sensed: list[Observation] | None = None,
) -> Perception:
    return Perception(
        time=time,
        self_state=RobotSelfState(
            robot_id=robot_id,
            position=position,
            battery=1.0,
            body=BodyType.SCOUT,
            speed_mps=1.0,
            consumable=1.0,
            health=1.0,
        ),
        sensed=sensed or [],
        inbox=inbox or [],
    )


def _beacon_messages(agent: ComposedAgent, perception: Perception) -> list[Message]:
    messages: list[Message] = []
    for message in agent.decide(perception).outbox:
        if message.kind is MessageKind.BEACON:
            messages.append(message)
    return messages


def test_relays_count_the_hops_from_the_source() -> None:
    source = Message(
        sender=-1,
        sent_at=0,
        kind=MessageKind.BEACON,
        payload=(Beacon(BeaconKind.TREAT, 0.0, (9, 9)),),
    )

    received = [source]
    beacons: list[Beacon] = []
    for robot_id, position in ((1, (8, 8)), (2, (6, 6)), (3, (4, 4))):
        received = _beacon_messages(_relay(robot_id), _perception(robot_id, position, 0, received))
        (message,) = received
        assert isinstance(message.payload, tuple)
        beacons.extend(message.payload)

    assert beacons == [
        Beacon(BeaconKind.TREAT, 1.0, (8, 8)),
        Beacon(BeaconKind.TREAT, 2.0, (6, 6)),
        Beacon(BeaconKind.TREAT, 3.0, (4, 4)),
    ]


def test_a_stale_beacon_is_not_relayed() -> None:
    relay = _relay(1)
    source = Message(
        sender=-1,
        sent_at=0,
        kind=MessageKind.BEACON,
        payload=(Beacon(BeaconKind.TREAT, 0.0, (9, 9)),),
    )
    assert _beacon_messages(relay, _perception(1, (8, 8), 0, [source])) != []

    later = BeaconConfig().fresh_for + TimeUnit.SECOND
    assert _beacon_messages(relay, _perception(1, (8, 8), later)) == []


def test_a_robot_that_sees_a_pest_is_its_source() -> None:
    pest = InfestationObservation(
        cell=(3, 3), confidence=1.0, observed_at=0, source=1, severity=0.5, pest=None
    )

    (message,) = _beacon_messages(_relay(1), _perception(1, (3, 3), sensed=[pest]))

    assert message.payload == (Beacon(BeaconKind.TREAT, 0.0, (3, 3)),)
