"""Tests for experiments described in a JSON file: parsing, validation and one short run."""

from pathlib import Path

import pytest

from swarm_field.contracts import TimeUnit
from swarm_field.experiments import PestScenario
from swarm_field.experiments.from_json import ExperimentFile, run_file

_CONFIGS = sorted(Path(__file__).parent.parent.joinpath("configs").glob("*.json"))


def test_every_condition_runs_with_every_seed_on_top_of_the_shared_parameters() -> None:
    experiment = ExperimentFile.from_dict(
        {
            "name": "radio",
            "scenario": "pests",
            "seeds": [3, 4],
            "hours": 0.5,
            "parameters": {"scouts": 7, "radio_range_m": 300},
            "conditions": {"short": {"radio_range_m": 50}, "long": {}},
        }
    )

    configs = experiment.into_configs()

    assert [config.label for config in configs] == ["short/3", "short/4", "long/3", "long/4"]
    assert [config.scenario.seed for config in configs] == [3, 4, 3, 4]
    assert all(config.scenario_name == "pests" for config in configs)
    assert all(config.until == TimeUnit.HOUR // 2 for config in configs)
    assert all(len(config.scenario.robots) == 7 + PestScenario().killers for config in configs)
    ranges = [config.scenario.robots[0].radio_range for config in configs]
    assert ranges == [50, 50, 300, 300]


def test_cells_given_as_json_lists_become_the_scenarios_cells() -> None:
    experiment = ExperimentFile.from_dict(
        {
            "name": "cells",
            "scenario": "pests",
            "parameters": {"docks": [[0, 0], [31, 31]], "obstacles": [[5, 5]]},
        }
    )

    (config,) = experiment.into_configs()

    assert config.scenario.docks == ((0, 0), (31, 31))
    assert config.scenario.obstacles == ((5, 5),)


def test_without_conditions_the_runs_are_named_after_the_experiment() -> None:
    experiment = ExperimentFile.from_dict({"name": "base", "scenario": "explore", "seeds": [1]})

    (config,) = experiment.into_configs()

    assert config.label == "base/1"


@pytest.mark.parametrize(
    "data",
    [
        {"name": "x", "scenario": "pests", "parameters": {"scuts": 5}},
        {"name": "x", "scenario": "pests", "parameters": {"explore": {"block": 4}}},
        {"name": "x", "scenario": "pest"},
        {"name": "x", "scenario": "pests", "seed": [1]},
    ],
)
def test_a_typo_is_refused_rather_than_run_with_the_default(data: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ExperimentFile.from_dict(data)


@pytest.mark.parametrize("path", _CONFIGS, ids=[path.name for path in _CONFIGS])
def test_every_example_file_is_a_valid_experiment(path: Path) -> None:
    assert ExperimentFile.from_path(path).into_configs()


def test_a_run_writes_the_summary_and_every_series(tmp_path: Path) -> None:
    experiment = ExperimentFile.from_dict(
        {"name": "short", "scenario": "explore", "seeds": [0, 1], "hours": 1 / 60}
    )

    runs = run_file(experiment, tmp_path)

    assert len(runs) == 2
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert len(list((tmp_path / "series").glob("*.csv"))) == 2
