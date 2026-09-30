"""Tests for scenario presets and initial world construction."""

import pytest

from swarm_field.contracts import BodyType, ScenarioConfig, ScenarioId
from swarm_field.simulation import build_scenario, build_world


@pytest.mark.parametrize(
    ("scenario", "width", "height", "robots"),
    [
        (ScenarioId.S1, 30, 30, 10),
        (ScenarioId.S2, 40, 40, 15),
        (ScenarioId.S3, 50, 50, 20),
        (ScenarioId.S4, 60, 60, 30),
        (ScenarioId.S5, 70, 70, 40),
    ],
)
def test_builtin_scenarios_have_expected_scale(
    scenario: ScenarioId,
    width: int,
    height: int,
    robots: int,
) -> None:
    config = build_scenario(scenario)

    assert config.field.width == width
    assert config.field.height == height
    assert len(config.robots) == robots
    assert len({robot.robot_id for robot in config.robots}) == robots


def test_build_world_applies_crops_obstacles_infestation_and_robots() -> None:
    config = build_scenario(ScenarioId.S5, seed=7)
    world = build_world(config)

    assert len(world.cells) == 70 * 70
    assert len(world.robots) == 40
    assert sum(state.obstacle for state in world.cells.values()) == len(config.obstacles)
    assert max(state.infestation for state in world.cells.values()) > 0.0
    assert all(state.crop is not None for state in world.cells.values())
    assert all(
        robot.position == config.robots[robot_id - 1].start
        for robot_id, robot in world.robots.items()
    )


def test_s1_is_single_crop_and_all_scouts() -> None:
    config = build_scenario(ScenarioId.S1)

    assert len(set(config.crops.values())) == 1
    assert all(robot.body is BodyType.SCOUT for robot in config.robots)
    assert config.obstacles == ()


def test_s3_contains_multiple_crop_species_and_body_types() -> None:
    config = build_scenario(ScenarioId.S3)

    assert len(set(config.crops.values())) >= 3
    assert {robot.body for robot in config.robots} == {
        BodyType.SCOUT,
        BodyType.KILLER,
        BodyType.BATTERY,
    }


def test_pest_configuration_is_seed_independent_but_presets_are_distinct() -> None:
    s1 = build_scenario(ScenarioId.S1, seed=1)
    s4 = build_scenario(ScenarioId.S4, seed=1)

    assert s1.pest_dynamics.spawn_probability == 0.0
    assert s4.pest_dynamics.spread_probability > s1.pest_dynamics.spread_probability
    assert s1.seed == s4.seed == 1


def test_custom_scenario_builds_without_a_preset() -> None:
    preset = build_scenario(ScenarioId.S1)
    custom = ScenarioConfig(
        field=preset.field,
        crops={(0, 0): next(iter(preset.crops.values()))},
        robots=preset.robots[:2],
        seed=123,
        initial_infestation={(1, 1): 0.7},
        obstacles=((2, 2),),
        pest_dynamics=preset.pest_dynamics,
    )

    world = build_world(custom)

    assert world.cell((0, 0)).crop is not None
    assert world.cell((1, 1)).infestation == pytest.approx(0.7)
    assert world.cell((2, 2)).obstacle
    assert len(world.robots) == 2
