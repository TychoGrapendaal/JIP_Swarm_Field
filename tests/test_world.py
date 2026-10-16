"""Tests for the high-level B.1 ground-truth world state."""

import pytest

from swarm_field.contracts import (
    BodyType,
    Charge,
    FieldGeometry,
    Idle,
    Move,
    PeerObservation,
    ReplaceBattery,
    TimeUnit,
    Treat,
)
from swarm_field.simulation import RobotState, WorldState
    TerrainObservation,
    Treat,
)
from swarm_field.simulation import EnergyConfig, RobotState, WorldState


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
        robots=[
            _robot(1, position=(0, 0), speed_mps=5.0),
            _robot(2, position=(0, 0), speed_mps=2.5),
        ],
    )

    fast_arrival = world.apply_action(1, Move(to=(1, 0)), time=0)
    slow_arrival = world.apply_action(2, Move(to=(1, 0)), time=0)

    assert fast_arrival == 1000
    assert slow_arrival == 2000


def test_move_to_obstacle_is_rejected_for_ground_robot() -> None:
    world = WorldState(_field(), robots=[_robot()])
    world.cell((2, 1)).obstacle = True

    arrival = world.apply_action(1, Move(to=(2, 1)), time=0)

    assert arrival is None
    assert world.positions()[1] == (1, 1)
    assert world.robots[1].move_target is None


def test_scout_can_fly_over_obstacle_cells() -> None:
    scout = RobotState(
        robot_id=1, position=(1, 1), body=BodyType.SCOUT, speed_mps=8.0
    )
    world = WorldState(_field(), robots=[scout])
    world.cell((2, 1)).obstacle = True

    arrival = world.apply_action(1, Move(to=(2, 1)), time=0)

    assert arrival is not None
    world.advance_to(arrival)
    assert world.positions()[1] == (2, 1)


def test_treat_reduces_severity_over_time_and_consumes_one_dose() -> None:
    world = WorldState(_field(), robots=[_robot()])
    world.cell((1, 1)).infestation = 0.8
    world.robots[1].consumable = 0.9

    world.apply_action(1, Treat(cell=(1, 1)), time=0)
    world.advance_to(30 * TimeUnit.MINUTE)

    assert world.cell((1, 1)).infestation == pytest.approx(0.3)
    assert world.robots[1].consumable == 0.8

    world.advance_to(60 * TimeUnit.MINUTE)
    assert world.cell((1, 1)).infestation == 0.0


def test_last_treatment_dose_can_finish_after_consumable_reaches_zero() -> None:
    world = WorldState(_field(), robots=[_robot()])
    world.cell((1, 1)).infestation = 1.0
    world.robots[1].consumable = 0.1

    world.apply_action(1, Treat(cell=(1, 1)), time=0)
    assert world.robots[1].consumable == 0.0

    # Repeated simulation ticks advance treatment, while reissuing Treat maintains the job.
    world.advance_to(30 * TimeUnit.MINUTE)
    world.apply_action(1, Treat(cell=(1, 1)), time=30 * TimeUnit.MINUTE)
    world.advance_to(60 * TimeUnit.MINUTE)

    assert world.cell((1, 1)).infestation == 0.0


def test_observe_for_reports_the_own_cell_and_the_terrain_around_it() -> None:
    robots = [
        _robot(1, position=(1, 1)),
        _robot(2, position=(1, 1)),
        _robot(3, position=(1, 1)),
        _robot(4, position=(2, 1)),
    ]
    world = WorldState(_field(), robots=robots)

    observations = world.observe_for(1, time=10)
    peers = [
        observation for observation in observations if isinstance(observation, PeerObservation)
    ]

    terrain_cells = {
        observation.cell
        for observation in observations
        if isinstance(observation, TerrainObservation)
    }
    other_cells = {
        observation.cell
        for observation in observations
        if not isinstance(observation, TerrainObservation)
    }

    assert terrain_cells == {(1, 1), *world.neighbours((1, 1))}
    assert other_cells == {(1, 1)}
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


def test_battery_robot_replaces_co_located_peer_battery_and_uses_one_spare() -> None:
    support = RobotState(
        robot_id=1,
        position=(1, 1),
        body=BodyType.BATTERY,
        speed_mps=1.0,
        spare_batteries=3,
    )
    recipient = RobotState(
        robot_id=2,
        position=(1, 1),
        body=BodyType.SCOUT,
        speed_mps=8.0,
        battery=0.1,
    )
    world = WorldState(_field(), robots=[support, recipient])

    world.apply_action(1, ReplaceBattery(robot_id=2), time=0)
    assert world.robots[2].battery == 0.1
    assert world.robots[1].spare_batteries == 3

    world.advance_to(224 * TimeUnit.SECOND)
    assert world.robots[2].battery == 1.0
    assert world.robots[1].spare_batteries == 2


def test_battery_replacement_requires_same_cell_and_spare_stock() -> None:
    support = RobotState(
        robot_id=1,
        position=(1, 1),
        body=BodyType.BATTERY,
        speed_mps=1.0,
        spare_batteries=0,
    )
    recipient = RobotState(
        robot_id=2,
        position=(2, 1),
        body=BodyType.KILLER,
        speed_mps=1.0,
        battery=0.1,
    )
    world = WorldState(_field(), robots=[support, recipient])

    world.apply_action(1, ReplaceBattery(robot_id=2))

    assert world.robots[2].battery == 0.1
    assert world.robots[1].spare_batteries == 0


def test_charge_only_works_at_a_configured_dock() -> None:
    from swarm_field.contracts import Charge

    robot = _robot(position=(1, 1))
    robot.battery = 0.2
    world = WorldState(_field(), robots=[robot], docks=((2, 1),))

    world.apply_action(1, Charge(), time=0)
    assert world.robots[1].battery == 0.2
    assert world.robots[1].current_action == Idle()

    world.robots[1].position = (2, 1)
    world.apply_action(1, Charge(), time=1)
    assert world.robots[1].battery == 0.2
    world.advance_to(3_600_001)
    assert world.robots[1].battery == 1.0


def test_battery_robot_refills_spares_when_charging_at_dock() -> None:
    from swarm_field.contracts import Charge

    support = RobotState(
        robot_id=1,
        position=(1, 1),
        body=BodyType.BATTERY,
        speed_mps=1.0,
        battery=0.4,
        spare_batteries=1,
    )
    world = WorldState(_field(), robots=[support], docks=((1, 1),))

    world.apply_action(1, Charge(), time=0)

    assert world.robots[1].battery == 0.4
    assert world.robots[1].spare_batteries == 3
    world.advance_to(3_600 * TimeUnit.SECOND)
    assert world.robots[1].battery == 1.0
def test_energy_is_drained_per_metre_and_per_decision() -> None:
    world = WorldState(
        _field(),
        robots=[_robot(speed_mps=5.0)],
        energy=EnergyConfig(per_metre=0.01, per_decision=0.02),
    )

    world.apply_action(1, Move(to=(2, 1)), time=0)

    # One cell across at 5 m per cell is 5 m of travel, plus the decision itself.
    assert world.robots[1].battery == pytest.approx(1.0 - 0.05 - 0.02)
    assert world.energy_spent() == pytest.approx(0.07)


def test_treat_costs_energy_and_charging_restores_the_battery() -> None:
    world = WorldState(
        _field(),
        robots=[_robot()],
        energy=EnergyConfig(per_decision=0.1, treat_per_action=0.05),
    )
    world.cell((1, 1)).infestation = 0.5

    world.apply_action(1, Treat(cell=(1, 1)), time=0)
    assert world.robots[1].battery == pytest.approx(0.85)

    world.apply_action(1, Charge(), time=0)
    assert world.robots[1].battery == pytest.approx(1.0)
    # Recharging tops the battery up but never gives back what was spent.
    assert world.energy_spent() == pytest.approx(0.25)


def test_battery_never_goes_below_empty() -> None:
    world = WorldState(
        _field(),
        robots=[_robot()],
        energy=EnergyConfig(per_decision=0.4),
    )

    world.apply_action(1, Idle(), time=0)
    world.apply_action(1, Idle(), time=1)
    world.apply_action(1, Idle(), time=2)

    assert world.robots[1].battery == 0.0
    assert world.energy_spent() == pytest.approx(1.0)
