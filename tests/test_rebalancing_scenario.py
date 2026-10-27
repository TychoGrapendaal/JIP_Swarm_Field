"""Tests for running role rebalancing (#64) in swarm scenarios: presets, assembly and runs."""

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
    HOP_BEST,
    REBALANCING,
    REBALANCING_ARMS,
    REBALANCING_HOP,
    SEEDED_BEST,
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
    brain = REBALANCING.brain(RebalancingPreset(BodyType.SCOUT, killer=SEEDED_BEST))

    assert brain.flies is False
    assert brain.keeps_map
    assert {"treat-infestation", "explore", "return-to-dock", "rebalance-roles"} <= set(
        brain.behaviors
    )
    assert brain.params["body_gates"] == {
        "treat-infestation": (BodyType.KILLER,),
        "explore": (BodyType.SCOUT,),
    }


def test_a_switching_robot_treats_with_its_killer_presets_parameters() -> None:
    brain = REBALANCING.brain(RebalancingPreset(BodyType.KILLER, killer=SEEDED_BEST))

    assert brain.params["treat_claim_fresh_for"] == SEEDED_BEST.claim_fresh_for


def test_a_rebalancing_robot_starts_as_a_scout_or_a_killer() -> None:
    with pytest.raises(ValueError):
        RebalancingPreset(BodyType.BATTERY)


def test_its_killer_preset_sets_how_it_treats() -> None:
    assert RebalancingPreset(killer=SEEDED_BEST).response == "seeded"
    assert RebalancingPreset(killer=HOP_BEST).response == "hop"


def test_the_body_gates_wrap_only_the_behaviours_they_name() -> None:
    spec = BrainSpec(behaviors=("treat-infestation", "explore"), seeders=("treat", "explore"))
    params = {"seeders": spec.seeders, "body_gates": {"treat-infestation": (BodyType.KILLER,)}}

    treat, explore = build_behaviors(spec, params, Random(0))

    assert isinstance(treat, BodyGate)
    assert treat.bodies == frozenset({BodyType.KILLER})
    assert not isinstance(explore, BodyGate)


def test_the_arms_keep_the_robots_and_their_starting_roles() -> None:
    arms = rebalancing_configurations(REBALANCING)
    fixed = arms["fixed"].into_experiment().scenario.robots

    assert tuple(arms) == REBALANCING_ARMS
    assert arms["fixed"] is REBALANCING
    switching = arms["rebalancing"].into_experiment().scenario.robots
    assert [robot.body for robot in switching] == [robot.body for robot in fixed]
    assert [robot.start for robot in switching] == [robot.start for robot in fixed]
    for group in arms["rebalancing"].swarm:
        assert isinstance(group.preset, RebalancingPreset)
        assert group.preset.killer == SEEDED_BEST


def test_the_rebalancing_arm_switches_at_a_dock() -> None:
    arms = rebalancing_configurations(REBALANCING, RebalanceConfig(min_monitors=3))

    for group in arms["rebalancing"].swarm:
        assert isinstance(group.preset, RebalancingPreset)
        assert group.preset.rebalance == RebalanceConfig(min_monitors=3, switch_at_dock=True)


def test_the_explorers_needed_grow_with_the_field_and_shrink_with_the_revisit_age() -> None:
    small = REBALANCING.without_robots().add_robots(EXPLORER, 1).add_robots(SEEDED_KILLER, 1)
    needed = monitors_needed(small, revisit_age=5 * TimeUnit.MINUTE)

    assert needed >= 1
    assert monitors_needed(small, revisit_age=20 * TimeUnit.MINUTE) < needed


def test_a_swarm_with_two_kinds_of_killer_is_refused() -> None:
    mixed = REBALANCING.add_robots(HOP_BEST, 1)

    with pytest.raises(ValueError):
        rebalancing_configurations(mixed)


def test_a_hop_swarm_rebalances_into_robots_that_treat_by_beacon() -> None:
    switching = rebalancing_configurations(REBALANCING_HOP)["rebalancing"]
    brain = switching.brain(switching.swarm[-1].preset)

    assert brain.beacons is not None
    assert "treat-hop" in brain.behaviors
    assert "treat-infestation" not in brain.behaviors
    assert brain.params["body_gates"]["treat-hop"] == (BodyType.KILLER,)
    assert brain.keeps_map


def _switches(name: str, minutes: int) -> list[tuple[tuple[int, int], BodyType]]:
    """Every role switch of a run of the catalogue's ``name``: where it happened, and into what.

    Sampled every 15 s; a switch keeps the robot busy for a minute, so it is still where it
    switched.
    """
    until = minutes * TimeUnit.MINUTE
    run = LiveRun.from_config(build_named_experiment(name, 0, until))
    robots = run.context.world.robots
    bodies = {robot_id: robot.body for robot_id, robot in robots.items()}
    switches: list[tuple[tuple[int, int], BodyType]] = []
    for time in range(0, until + 1, 15 * TimeUnit.SECOND):
        run.advance_to(time)
        for robot_id, robot in robots.items():
            if robot.body is not bodies[robot_id]:
                switches.append((robot.position, robot.body))
                bodies[robot_id] = robot.body
    return switches


def test_the_default_swarm_turns_explorers_into_killers_on_a_dock() -> None:
    switches = _switches("rebalancing", 20)

    assert switches
    assert all(cell in REBALANCING.map.docks for cell, _ in switches)
    assert switches[0][1] is BodyType.KILLER


def test_the_hop_rebalancing_swarm_switches_on_a_dock_too() -> None:
    switches = _switches("rebalancing-hop", 20)

    assert switches
    assert all(cell in REBALANCING_HOP.map.docks for cell, _ in switches)


def test_a_switch_takes_the_scenarios_role_switch_time() -> None:
    scenario = replace(
        REBALANCING, robots=replace(REBALANCING.robots, role_switch_time=2 * TimeUnit.MINUTE)
    )

    roles = scenario.into_experiment().scenario.roles

    assert roles.duration == 2 * TimeUnit.MINUTE
    assert roles.speeds[BodyType.SCOUT] == scenario.robots.speed_mps_drone
    assert roles.speeds[BodyType.KILLER] == scenario.robots.speed_mps_ground
