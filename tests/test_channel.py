"""Tests for the communication channel (B.3)."""

from swarm_field.contracts import BodyType, CellId, FieldGeometry, Message, MessageKind, RobotId
from swarm_field.simulation import Channel, RobotState, WorldState, robots_in_range_of


def _world(robots: list[RobotState], cell_size: float = 1.0) -> WorldState:
    field = FieldGeometry(
        width=10,
        height=10,
        cell_size=cell_size,
        row_sp=1.0,
        in_row_sp=1.0,
        headland=0.0,
        connectivity=4,
    )
    return WorldState(field, robots=robots)


def _robot(robot_id: RobotId, position: CellId, radio_range_m: float) -> RobotState:
    return RobotState(
        robot_id=robot_id,
        position=position,
        body=BodyType.SCOUT,
        speed_mps=1.0,
        radio_range_m=radio_range_m,
    )


def _hello(sender: RobotId, payload: object = None) -> Message:
    return Message(sender=sender, sent_at=0, kind=MessageKind.HELLO, payload=payload)


def test_robots_in_range_excludes_sender_and_distant_robots() -> None:
    world = _world(
        [
            _robot(1, (0, 0), radio_range_m=3.0),
            _robot(2, (0, 0), radio_range_m=0.0),  # same cell
            _robot(3, (3, 0), radio_range_m=0.0),  # exactly at the edge of range
            _robot(4, (4, 0), radio_range_m=0.0),  # just out of range
        ]
    )

    assert robots_in_range_of(1, world, time=0) == [2, 3]


def test_only_the_senders_range_counts() -> None:
    world = _world([_robot(1, (0, 0), radio_range_m=5.0), _robot(2, (2, 0), radio_range_m=1.0)])

    assert robots_in_range_of(1, world, time=0) == [2]
    assert robots_in_range_of(2, world, time=0) == []


def test_range_is_measured_in_metres() -> None:
    world = _world(
        [_robot(1, (0, 0), radio_range_m=3.0), _robot(2, (2, 0), radio_range_m=0.0)],
        cell_size=2.0,
    )

    assert robots_in_range_of(1, world, time=0) == []


def test_broadcast_delivers_to_inboxes_in_order_of_sending() -> None:
    world = _world(
        [
            _robot(1, (0, 0), radio_range_m=2.0),
            _robot(2, (1, 0), radio_range_m=2.0),
            _robot(3, (9, 9), radio_range_m=2.0),
        ]
    )
    channel = Channel()
    first, second, reply = _hello(1, "first"), _hello(1, "second"), _hello(2, "reply")

    channel.broadcast(world, 0, [first, second])
    channel.broadcast(world, 0, [reply])

    assert channel.take_inbox(1) == [reply]
    assert channel.take_inbox(2) == [first, second]
    assert channel.take_inbox(3) == []


def test_take_inbox_empties_it() -> None:
    world = _world([_robot(1, (0, 0), radio_range_m=1.0), _robot(2, (1, 0), radio_range_m=1.0)])
    channel = Channel()
    channel.broadcast(world, 0, [_hello(1)])

    channel.take_inbox(2)

    assert channel.take_inbox(2) == []
