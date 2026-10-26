"""Tests for running role rebalancing (#64) in swarm scenarios: presets, assembly and a run."""

from dataclasses import replace
from random import Random

import pytest

from swarm_field.contracts import (
    BodyType,
    FieldGeometry,
    RoleSwitchSpec,
    SwitchRole,
    TimeUnit,
)
from swarm_field.experiments import (
    EXPLORER,
    REBALANCING,
    SEEDED_KILLER,
    LiveRun,
    RebalancingPreset,
    build_named_experiment,
    monitors_needed,
    rebalancing_configurations,
)
from swarm_field.runner import BrainSpec, build_behaviors
from swarm_field.simulation import RobotState, WorldState
from swarm_field.swarm import BodyGate, RebalanceConfig, capabilities_for


def test_scouts_and_killers_may_switch_role_and_battery_robots_may_not() -> None:
    assert capabilities_for(BodyType.SCOUT).can(SwitchRole)
    assert capabilities_for(BodyType.KILLER).can(SwitchRole)
    assert not capabilities_for(BodyType.BATTERY).can(SwitchRole)


def test_the_world_does_not_turn_a_battery_robot_into_a_killer() -> None:
    field = FieldGeometry(4, 4, 10.0, 0.0, 0.0, 0.0)
    robot = RobotState(robot_id=1, position=(0, 0), body=BodyType.BATTERY, speed_mps=1.0)
    world = WorldState(field, [robot], roles=RoleSwitchSpec(speeds={BodyType.KILLER: 0.5}))

    world.apply_action(1, SwitchRole(BodyType.KILLER), time=0)

    assert world.robots[1].body is BodyType.BATTERY
    assert world.robots[1].role_switches == 0


def test_a_switching_robot_carries_both_roles_and_plans_on_the_ground() -> None:
    brain = REBALANCING.brain(RebalancingPreset(BodyType.SCOUT))

    assert brain.flies is False
    assert set(brain.behaviors) == {
        "treat-infestation",
        "explore",
        "return-to-dock",
        "rebalance-roles",
    }
    assert brain.params["body_gates"] == {
        "treat-infestation": (BodyType.KILLER,),
        "explore": (BodyType.SCOUT,),
    }


def test_a_rebalancing_robot_starts_as_a_scout_or_a_killer() -> None:
    with pytest.raises(ValueError):
        RebalancingPreset(BodyType.BATTERY)


def test_the_body_gates_wrap_only_the_behaviours_they_name() -> None:
    spec = BrainSpec(behaviors=("treat-infestation", "explore"), seeders=("treat", "explore"))
    params = {"seeders": spec.seeders, "body_gates": {"treat-infestation": (BodyType.KILLER,)}}

    treat, explore = build_behaviors(spec, params, Random(0))

    assert isinstance(treat, BodyGate)
    assert treat.bodies == frozenset({BodyType.KILLER})
    assert not isinstance(explore, BodyGate)


def test_rebalancing_keeps_the_robots_and_their_starting_roles() -> None:
    arms = rebalancing_configurations(REBALANCING)
    fixed = arms["fixed"].into_experiment().scenario.robots
    switching = arms["rebalancing"].into_experiment().scenario.robots

    assert [robot.body for robot in switching] == [robot.body for robot in fixed]
    assert [robot.start for robot in switching] == [robot.start for robot in fixed]
    assert arms["fixed"] is REBALANCING
    for group in arms["rebalancing"].swarm:
        assert isinstance(group.preset, RebalancingPreset)


def test_a_given_rebalance_config_reaches_every_switching_robot() -> None:
    config = RebalanceConfig(min_monitors=3, switch_at_dock=True)
    switching = rebalancing_configurations(REBALANCING, config)["rebalancing"]

    for group in switching.swarm:
        assert isinstance(group.preset, RebalancingPreset)
        assert group.preset.rebalance == config


def test_the_explorers_needed_grow_with_the_field_and_shrink_with_the_revisit_age() -> None:
    small = REBALANCING.without_robots().add_robots(EXPLORER, 1).add_robots(SEEDED_KILLER, 1)
    needed = monitors_needed(small)

    assert needed >= 1
    assert monitors_needed(small, revisit_age=2 * 30 * TimeUnit.MINUTE) < needed


def test_the_rebalancing_swarm_lets_go_of_the_killers_a_clean_field_does_not_need() -> None:
    until = 20 * TimeUnit.MINUTE
    fixed = LiveRun.from_config(build_named_experiment("rebalancing-fixed", 0, until))
    switching = LiveRun.from_config(build_named_experiment("rebalancing", 0, until))
    fixed.advance_to(until)
    switching.advance_to(until)

    def killers(run: LiveRun) -> int:
        robots = run.context.world.robots.values()
        return sum(1 for robot in robots if robot.body is BodyType.KILLER)

    switches = sum(robot.role_switches for robot in switching.context.world.robots.values())
    assert killers(fixed) == REBALANCING.killers
    assert killers(switching) < REBALANCING.killers
    assert switches > 0


def test_a_switch_takes_the_scenarios_role_switch_time() -> None:
    scenario = replace(
        REBALANCING, robots=replace(REBALANCING.robots, role_switch_time=2 * TimeUnit.MINUTE)
    )

    roles = scenario.into_experiment().scenario.roles

    assert roles.duration == 2 * TimeUnit.MINUTE
    assert roles.speeds[BodyType.SCOUT] == scenario.robots.speed_mps_drone
    assert roles.speeds[BodyType.KILLER] == scenario.robots.speed_mps_ground
