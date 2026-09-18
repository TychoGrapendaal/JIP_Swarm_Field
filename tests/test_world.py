"""Tests for the B.1 ground-truth world state."""

from swarm_field.contracts import (
    BodyType,
    Direction,
    FieldGeometry,
    Move,
    PeerObservation,
    Treat,
    Turn,
)
from swarm_field.simulation import RobotState, WorldState


def _field(connectivity: int = 4) -> FieldGeometry:
    return FieldGeometry(
        width=4,
        height=3,
        cell_size=1.0,
        row_sp=1.0,
        in_row_sp=1.0,
        headland=1.0,
        connectivity=connectivity,
    )


def _robot(
    robot_id: int = 1,
    position: tuple[int, int] = (1, 1),
    facing: Direction = Direction.E,
) -> RobotState:
    return RobotState(
        robot_id=robot_id,
        position=position,
        body=BodyType.KILLER,
        facing=facing,
    )


def test_world_creates_a_complete_cell_grid() -> None:
    world = WorldState(_field())

    assert len(world.cells) == 12
    assert all(state.occupant is None for state in world.cells.values())


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


def test_corner_only_has_in_bounds_neighbours() -> None:
    world = WorldState(_field(connectivity=8))

    assert set(world.neighbours((0, 0))) == {(1, 0), (0, 1), (1, 1)}


def test_add_robot_updates_position_and_occupancy() -> None:
    robot = _robot()
    world = WorldState(_field(), robots=[robot])

    assert world.positions() == {1: (1, 1)}
    assert world.cell((1, 1)).occupant == 1


def test_move_updates_true_pose_and_occupancy() -> None:
    world = WorldState(_field(), robots=[_robot(position=(1, 1), facing=Direction.E)])

    world.apply_action(1, Move())

    assert world.positions() == {1: (2, 1)}
    assert world.cell((1, 1)).occupant is None
    assert world.cell((2, 1)).occupant == 1


def test_move_into_obstacle_is_ignored() -> None:
    world = WorldState(_field(), robots=[_robot(position=(1, 1), facing=Direction.E)])
    world.cell((2, 1)).obstacle = True

    world.apply_action(1, Move())

    assert world.positions() == {1: (1, 1)}
    assert world.cell((1, 1)).occupant == 1


def test_turn_updates_heading() -> None:
    world = WorldState(_field(), robots=[_robot(facing=Direction.E)])

    world.apply_action(1, Turn(Direction.N))

    assert world.robots[1].facing == Direction.N


def test_treat_removes_infestation_at_robot_cell() -> None:
    world = WorldState(_field(), robots=[_robot()])
    world.cell((1, 1)).infestation = 0.8
    world.robots[1].consumable = 0.9

    world.apply_action(1, Treat())

    assert world.cell((1, 1)).infestation == 0.0
    assert world.robots[1].consumable == 0.0


def test_perceived_self_matches_current_ground_truth_for_v1() -> None:
    world = WorldState(_field(), robots=[_robot()])

    self_state = world.perceived_self(1)

    assert self_state.robot_id == 1
    assert self_state.position == (1, 1)
    assert self_state.facing == Direction.E
    assert self_state.body == BodyType.KILLER


def test_observe_for_returns_local_cell_observations() -> None:
    world = WorldState(_field(), robots=[_robot()])
    world.cell((2, 1)).infestation = 0.75
    world.cell((1, 2)).obstacle = True

    observations = world.observe_for(1, time=25)

    cells = {observation.cell for observation in observations}
    assert cells == {(1, 1), (2, 1), (1, 2), (0, 1), (1, 0)}
    assert any(
        observation.cell == (2, 1)
        and getattr(observation, "severity", None) == 0.75
        for observation in observations
    )
    assert any(
        observation.cell == (1, 2)
        and getattr(observation, "is_obstacle", False)
        for observation in observations
    )


def test_observe_for_reports_a_visible_peer() -> None:
    robots = [
        _robot(robot_id=1, position=(1, 1)),
        _robot(robot_id=2, position=(2, 1)),
    ]
    world = WorldState(_field(), robots=robots)

    observations = world.observe_for(1, time=10)

    peers = [
        observation for observation in observations if isinstance(observation, PeerObservation)
    ]
    assert len(peers) == 1
    assert peers[0].peer == 2


def test_public_states_and_snapshot_are_ground_truth_views() -> None:
    robot = _robot(position=(1, 1), facing=Direction.E)
    world = WorldState(_field(), robots=[robot])
    world.apply_action(1, Move())

    public_states = world.public_states()
    snapshot = world.snapshot(time=7)

    assert public_states[0].position == (2, 1)
    assert public_states[0].current_action == Move()
    assert snapshot.time == 7
    assert snapshot.cells[(2, 1)].occupant == 1
