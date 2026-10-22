"""Acceptance tests for Issue #77: physical battery depletion and dock charging."""

import pytest

from swarm_field.contracts import (
    BodyType,
    Charge,
    FailureMode,
    FieldGeometry,
    Idle,
    Move,
    PeerObservation,
    RobotSpec,
    TimeUnit,
)
from swarm_field.simulation import (
    EnergyConfig,
    MetricsCollector,
    MetricsConfig,
    RobotState,
    WorldState,
)
from swarm_field.swarm import RechargeConfig
from swarm_field.swarm.capabilities import CapabilitySet, capabilities_for


def _field() -> FieldGeometry:
    return FieldGeometry(
        width=8,
        height=6,
        cell_size=5.0,
        row_sp=2.0,
        in_row_sp=1.0,
        headland=5.0,
        connectivity=4,
    )


def _robot(
    robot_id: int,
    *,
    body: BodyType = BodyType.KILLER,
    position: tuple[int, int] = (2, 2),
    battery: float = 1.0,
    spare_batteries: int = 0,
    speed_mps: float = 1.0,
) -> RobotState:
    return RobotState(
        robot_id=robot_id,
        position=position,
        body=body,
        speed_mps=speed_mps,
        battery=battery,
        spare_batteries=spare_batteries,
    )


def test_dock_charges_only_one_robot_and_queues_the_next_fifo() -> None:
    dock = (2, 2)
    first = _robot(1, position=dock, battery=0.5)
    second = _robot(2, position=dock, battery=0.5)
    third = _robot(3, position=dock, battery=0.5)
    world = WorldState(_field(), robots=[first, second, third], docks=[dock])

    world.apply_action(1, Charge(), time=0)
    world.apply_action(2, Charge(), time=0)
    world.apply_action(3, Charge(), time=0)

    active = [robot for robot in world.robots.values() if robot.charging_updated_at is not None]
    assert [robot.robot_id for robot in active] == [1]
    assert world.robots[2].battery == pytest.approx(0.5)
    assert world.robots[3].battery == pytest.approx(0.5)
    assert world.robots[2].charging_requested_at == 0
    assert world.robots[3].charging_requested_at == 0

    # A KILLER with 50% battery needs exactly 30 minutes at its one-hour full-charge rate.
    handoff = 30 * TimeUnit.MINUTE
    world.advance_to(handoff)

    assert world.robots[1].battery == pytest.approx(1.0)
    assert world.robots[2].battery == pytest.approx(0.5)
    assert world.robots[2].charging_updated_at == handoff
    assert world.robots[3].charging_updated_at is None
    assert (
        len([robot for robot in world.robots.values() if robot.charging_updated_at is not None])
        == 1
    )

    world.advance_to(60 * TimeUnit.MINUTE)
    assert world.robots[2].battery == pytest.approx(1.0)
    assert world.robots[3].charging_updated_at == 60 * TimeUnit.MINUTE


def test_an_empty_robot_at_a_dock_can_recover_by_charging() -> None:
    dock = (2, 2)
    robot = _robot(1, position=dock, battery=0.0)
    world = WorldState(_field(), robots=[robot], docks=[dock])

    world.apply_action(1, Charge(), time=0)
    assert world.robots[1].failed is False
    assert world.robots[1].charging_updated_at == 0

    world.advance_to(60 * TimeUnit.MINUTE)
    assert world.robots[1].battery == pytest.approx(1.0)
    assert world.robots[1].failed is False


def test_empty_battery_away_from_dock_fails_and_is_counted_in_metrics() -> None:
    robot = _robot(1, battery=0.00005, position=(4, 3))
    world = WorldState(
        _field(), robots=[robot], docks=[(0, 0)], energy=EnergyConfig(per_decision=0.0001)
    )

    world.apply_action(1, Idle(), time=0)

    assert world.robots[1].battery == 0.0
    assert world.robots[1].failed is True
    assert world.robots[1].failure_mode is FailureMode.DEAD_BATTERY
    assert world.live_robots() == ()
    assert world.battery_failure_count() == 1

    metrics = MetricsCollector(config=MetricsConfig(period=TimeUnit.SECOND))
    sample = metrics.sample(world, now=0, messages_sent=0)
    assert sample.active_robots == 0
    assert sample.battery_failures == 1


def test_charge_action_away_from_a_dock_cannot_restore_battery() -> None:
    robot = _robot(1, position=(4, 3), battery=0.4)
    world = WorldState(_field(), robots=[robot], docks=[(0, 0)])

    world.apply_action(1, Charge(), time=0)
    world.advance_to(60 * TimeUnit.MINUTE)

    assert world.robots[1].battery == pytest.approx(0.4)
    assert world.robots[1].charging_updated_at is None


def test_unaffordable_move_does_not_start_or_partially_pay_for_travel() -> None:
    robot = _robot(1, position=(2, 2), battery=0.6, speed_mps=5.0)
    world = WorldState(
        _field(),
        robots=[robot],
        docks=[(0, 0)],
        energy=EnergyConfig(per_decision=0.1, per_metre=0.1),
    )

    world.apply_action(1, Move(to=(3, 2)), time=0)

    assert world.robots[1].move_target is None
    assert world.robots[1].position == (2, 2)
    assert world.robots[1].battery == pytest.approx(0.5)
    assert world.robots[1].failed is False


def test_spend_and_charge_keep_battery_inside_zero_to_one() -> None:
    dock = (2, 2)
    robot = _robot(1, position=dock, battery=0.2)
    world = WorldState(
        _field(), robots=[robot], docks=[dock], energy=EnergyConfig(per_decision=0.4)
    )

    world.apply_action(1, Idle(), time=0)
    world.apply_action(1, Charge(), time=1)
    world.advance_to(60 * TimeUnit.MINUTE)

    assert 0.0 <= world.robots[1].battery <= 1.0


def test_pr61_numeric_defaults_remove_optional_battery_fields() -> None:
    peer = PeerObservation(
        cell=(1, 1), confidence=1.0, observed_at=0, source=1, peer=2, body=BodyType.SCOUT
    )
    spec = RobotSpec(
        robot_id=3,
        body=BodyType.BATTERY,
        start=(1, 1),
        v_max=1.0,
        battery=1.0,
        radio_range=10.0,
    )
    profile = CapabilitySet(
        body=BodyType.SCOUT,
        action_types=frozenset(),
        can_fly=True,
        speed_mps=8.0,
        pest_sensing_radius=1,
        terrain_sensing_radius=1,
        peer_sensing_radius=1,
        endurance_seconds=1800.0,
    )

    assert peer.battery == 1.0
    assert spec.spare_batteries == 0
    assert profile.dock_recharge_seconds == 0.0


def test_recharge_config_validates_critical_threshold_below_low_threshold() -> None:
    assert RechargeConfig().low_battery_threshold == 0.25
    with pytest.raises(ValueError, match="critical < low"):
        RechargeConfig(low_battery_threshold=0.1, critical_battery_threshold=0.2)


@pytest.mark.parametrize(
    ("body", "full_charge_minutes"),
    [
        (BodyType.SCOUT, 45),
        (BodyType.KILLER, 60),
        (BodyType.BATTERY, 60),
    ],
)
def test_each_body_charges_at_its_configured_rate(body: BodyType, full_charge_minutes: int) -> None:
    dock = (2, 2)
    robot = _robot(1, body=body, position=dock, battery=0.25, spare_batteries=0)
    world = WorldState(_field(), robots=[robot], docks=[dock])
    profile = capabilities_for(body)
    assert profile.dock_recharge_seconds == full_charge_minutes * 60

    world.apply_action(1, Charge(), time=0)
    world.advance_to(int(0.75 * full_charge_minutes * TimeUnit.MINUTE))

    assert world.robots[1].battery == pytest.approx(1.0)
    assert world.robots[1].failed is False


def test_dead_battery_robot_cannot_resume_work() -> None:
    robot = _robot(1, position=(4, 3), battery=0.0)
    world = WorldState(_field(), robots=[robot], docks=[(0, 0)])
    world.advance_to(0)

    assert world.robots[1].failed is True
    world.apply_action(1, Move(to=(5, 3)), time=1)
    world.advance_to(TimeUnit.MINUTE)

    assert world.robots[1].position == (4, 3)
    assert world.robots[1].move_target is None
    assert world.robots[1].failure_mode is FailureMode.DEAD_BATTERY


def test_decision_cost_is_charged_while_a_committed_move_is_underway() -> None:
    robot = _robot(1, position=(2, 2), battery=0.8, speed_mps=1.0)
    world = WorldState(
        _field(),
        robots=[robot],
        docks=[(0, 0)],
        energy=EnergyConfig(per_decision=0.1),
    )
    arrival = world.apply_action(1, Move(to=(3, 2)), time=0)
    assert arrival is not None
    after_start = world.robots[1].battery

    world.apply_action(1, Idle(), time=1)

    assert world.robots[1].move_target == (3, 2)
    assert world.robots[1].battery == pytest.approx(after_start - 0.1)
