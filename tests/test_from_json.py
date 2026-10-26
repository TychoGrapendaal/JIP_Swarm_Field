"""Tests for experiments described in a JSON file: parsing, validation and one short run."""

import json
from pathlib import Path

import pytest

from swarm_field.contracts import TimeUnit
from swarm_field.experiments import EXPLORER, HOP_BEST, PESTS, SEEDED_BEST, SEEDED_KILLER
from swarm_field.experiments.from_json import ExperimentFile, run_file

_CONFIGS = sorted(Path(__file__).parent.parent.joinpath("configs").glob("*.json"))


def test_every_condition_changes_the_shared_parameters_group_by_group() -> None:
    experiment = ExperimentFile.from_dict(
        {
            "name": "radio",
            "scenario": "pests",
            "seeds": [3, 4],
            "parameters": {
                "swarm": {"explorer": 7},
                "robots": {"radio_range_m": 300, "speed_mps_drone": 4.0},
            },
            "conditions": {"short": {"robots": {"radio_range_m": 50}}, "long": {}},
        }
    )

    scenarios = experiment.into_scenarios()

    assert list(scenarios) == ["short", "long"]
    for scenario in scenarios.values():
        assert scenario.count(EXPLORER.name) == 7
        assert scenario.count(SEEDED_KILLER.name) == PESTS.count(SEEDED_KILLER.name)
        assert scenario.robots.speed_mps_drone == 4.0
    assert scenarios["short"].robots.radio_range_m == 50
    assert scenarios["long"].robots.radio_range_m == 300


def test_a_file_can_compare_the_frozen_presets() -> None:
    experiment = ExperimentFile.from_dict(
        {
            "name": "best",
            "scenario": "pests",
            "parameters": {"swarm": {"seeded-killer": 0}},
            "conditions": {
                "seeded": {"swarm": {"seeded-best": 3}},
                "hop": {"swarm": {"hop-best": 3}},
            },
        }
    )

    scenarios = experiment.into_scenarios()

    assert scenarios["seeded"].count(SEEDED_BEST.name) == 3
    assert scenarios["hop"].count(HOP_BEST.name) == 3
    assert scenarios["hop"].killers == 3


def test_cells_given_as_json_lists_become_the_scenarios_cells() -> None:
    experiment = ExperimentFile.from_dict(
        {
            "name": "cells",
            "scenario": "pests",
            "parameters": {"map": {"docks": [[0, 0], [31, 31]], "obstacles": [[5, 5]]}},
        }
    )

    (scenario,) = experiment.into_scenarios().values()

    assert scenario.map.docks == ((0, 0), (31, 31))
    assert scenario.map.obstacles == ((5, 5),)


def test_a_revisit_age_from_the_file_replaces_the_one_from_the_sweep_time() -> None:
    experiment = ExperimentFile.from_dict(
        {"name": "x", "scenario": "explore", "parameters": {"explore": {"revisit_age": 600_000}}}
    )

    (scenario,) = experiment.into_scenarios().values()

    explore = scenario.brain(scenario.swarm[0].preset).params["explore"]
    assert explore.revisit_age == 10 * TimeUnit.MINUTE


@pytest.mark.parametrize(
    "data",
    [
        {"name": "x", "scenario": "pests", "parameters": {"robots": {"radio": 5}}},
        {"name": "x", "scenario": "pests", "parameters": {"explore": {"block": 4}}},
        {"name": "x", "scenario": "pests", "parameters": {"swarm": {"explorers": 4}}},
        {"name": "x", "scenario": "pests", "parameters": {"radio_range_m": 50}},
        {"name": "x", "scenario": "pest"},
        {"name": "x", "scenario": "pests", "seed": [1]},
    ],
)
def test_a_typo_is_refused_rather_than_run_with_the_default(data: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ExperimentFile.from_dict(data)


@pytest.mark.parametrize("hours", [0.5, 0])
def test_a_run_must_last_whole_hours(hours: float) -> None:
    with pytest.raises(ValueError):
        ExperimentFile.from_dict({"name": "x", "scenario": "pests", "hours": hours})


@pytest.mark.parametrize("path", _CONFIGS, ids=[path.name for path in _CONFIGS])
def test_every_example_file_is_a_valid_experiment(path: Path) -> None:
    assert ExperimentFile.from_path(path).into_scenarios()


def test_a_run_writes_every_result_and_the_table(tmp_path: Path) -> None:
    experiment = ExperimentFile.from_dict(
        {
            "name": "short",
            "scenario": "explore",
            "seeds": [0, 1],
            "jobs": 1,
            "parameters": {"map": {"size": 8}, "swarm": {"explorer": 2}},
        }
    )

    batch, results = run_file(experiment, tmp_path)

    assert len(results) == 2
    assert len(json.loads((batch.directory / "results.json").read_text())) == 2
    assert (batch.directory / "tables.md").exists()
