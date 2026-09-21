"""Tests for the high-level B.1 ground-truth world state."""

from swarm_field.contracts import BodyType, FieldGeometry, Idle, Move, PeerObservation, Treat
from swarm_field.simulation import RobotState, WorldState


def _field(connectivity: int = 8) -> FieldGeometry:
    return FieldGeometry(
        width=5,
        height=4,
        cell_size=5.0,
        row_sp=2.0,
        in_row_sp=1.0,
        headland=5.0,
        connectivity=connectivity,
    )


def _robot(
    robot_id: int = 1,
    position: tuple[int, int] = (1, 1),
    speed_mps: float = 1.0,
) -> RobotState:
    return RobotState(
        robot_id=robot_id,
        position=position,
        body=BodyType.KILLER,
        speed_mps=speed_mps,
    )


def test_world_creates_a_complete_cell_grid() -> None:
    world = WorldState(_field())

    assert len(world.cells) == 20
    assert all(not state.obstacle for state in world.cells.values())


def test_four_neighbour_adjacency() -> None:
    world = WorldState(_field(connectivity=4))

    assert set(world.neighbours((1, 1))) == {(1, 2), (2, 1), (1, 0), (0, 1)}


def test_eight_neighbour_adjacency() -> None:
    world = WorldState(_field(connectivity=8))

    assert set(world.neighbours((1, 1))) == {
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (2, 1),
        (0, 2),
        (1, 2),
        (2, 2),
    }


def test_obstacle_can_span_multiple_cells() -> None:
    world = WorldState(_field())
    for x in (1, 2, 3):
        world.cell((x, 2)).obstacle = True

    assert all(world.cell((x, 2)).obstacle for x in (1, 2, 3))
    assert world.is_walkable((0, 2))
    assert not world.is_walkable((2, 2))


def test_multiple_robots_may_share_a_cell() -> None:
    robots = [_robot(1), _robot(2), _robot(3)]
    world = WorldState(_field(), robots=robots)

    assert world.robots_at((1, 1)) == (1, 2, 3)
    assert world.positions() == {1: (1, 1), 2: (1, 1), 3: (1, 1)}


def test_direct_move_has_target_and_arrival_time() -> None:
    world = WorldState(_field(), robots=[_robot(speed_mps=5.0)])

    arrival = world.apply_action(1, Move(to=(2, 1)), time=0)

    assert arrival == 1000
    assert world.robots[1].move_target == (2, 1)
    assert world.positions()[1] == (1, 1)


def test_robot_arrives_when_world_time_advances() -> None:
    world = WorldState(_field(), robots=[_robot(speed_mps=5.0)])
    world.apply_action(1, Move(to=(2, 1)), time=0)

    world.advance_to(999)
    assert world.positions()[1] == (1, 1)

    world.advance_to(1000)
    assert world.positions()[1] == (2, 1)
    assert world.robots[1].current_action == Idle()


def test_faster_robot_arrives_earlier() -> None:
    world = WorldState(
        _field(),
        robots=[_robot(1, position=(0, 0), speed_mps=5.0), _robot(2, position=(0, 0), speed_mps=2.5)],
    )

    fast_arrival = world.apply_action(1, Move(to=(1, 0)), time=0)
    slow_arrival = world.apply_action(2, Move(to=(1, 0)), time=0)

    assert fast_arrival == 1000
    assert slow_arrival == 2000


def test_move_to_obstacle_is_rejected() -> None:
    world = WorldState(_field(), robots=[_robot()])
    world.cell((2, 1)).obstacle = True

    arrival = world.apply_action(1, Move(to=(2, 1)), time=0)

    assert arrival is None
    assert world.positions()[1] == (1, 1)
    assert world.robots[1].move_target is None


def test_treat_changes_field_patch_and_consumable() -> None:
    world = WorldState(_field(), robots=[_robot()])
    world.cell((2, 1)).infestation = 0.8
    world.robots[1].consumable = 0.9

    world.apply_action(1, Treat(cell=(2, 1)))

    assert world.cell((2, 1)).infestation == 0.0
    assert world.robots[1].consumable == 0.8


def test_observe_for_reports_multiple_visible_peers() -> None:
    robots = [
        _robot(1, position=(1, 1)),
        _robot(2, position=(2, 1)),
        _robot(3, position=(2, 1)),
    ]
    world = WorldState(_field(), robots=robots)

    observations = world.observe_for(1, time=10)
    peers = [observation for observation in observations if isinstance(observation, PeerObservation)]

    assert {peer.peer for peer in peers} == {2, 3}


def test_continuous_position_interpolates() -> None:
    world = WorldState(_field(), robots=[_robot(speed_mps=5.0)])
    world.apply_action(1, Move(to=(2, 1)), time=0)

    x, y = world.continuous_position(1, time=500)

    assert x == 2.0
    assert y == 1.5


def test_snapshot_contains_robot_target_without_cell_occupancy() -> None:
    world = WorldState(_field(), robots=[_robot(speed_mps=5.0)])
    world.apply_action(1, Move(to=(2, 1)), time=0)
    snapshot = world.snapshot(time=0)

    assert snapshot.robots[0].position == (1, 1)
    assert snapshot.robots[0].target == (2, 1)
    assert not hasattr(snapshot.cells[(1, 1)], "occupant")
