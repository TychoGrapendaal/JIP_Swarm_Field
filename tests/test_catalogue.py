"""Tests for the catalogue of named scenarios: building them by name and sweeping them."""

import pytest

from swarm_field.contracts import TimeUnit
from swarm_field.experiments import (
    EXPLORE,
    build_named_experiment,
    scenario_names,
)
from swarm_field.runner import run_experiment

_SHORT = TimeUnit.MINUTE


def test_the_catalogue_names_the_presets_and_the_hand_built_scenarios() -> None:
    assert scenario_names() == (
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "explore",
        "pests",
        "pests-large",
        "rebalancing",
        "rebalancing-fixed",
        "rebalancing-hop",
        "rebalancing-hop-fixed",
    )


@pytest.mark.parametrize("name", scenario_names())
def test_every_scenario_reproduces_its_run_from_the_same_seed(name: str) -> None:
    first = run_experiment(build_named_experiment(name, 7, _SHORT))
    second = run_experiment(build_named_experiment(name, 7, _SHORT))

    assert first == second


def test_a_named_run_is_labelled_with_its_name_and_seed() -> None:
    config = build_named_experiment("pests", 3)

    assert config.label == "pests/3"
    assert config.scenario_name == "pests"
    assert config.scenario.seed == 3


def test_a_scenario_keeps_its_own_horizon_unless_one_is_given() -> None:
    assert build_named_experiment("S1", 1).until == TimeUnit.HOUR
    assert build_named_experiment("explore", 1).until == EXPLORE.until
    assert build_named_experiment("explore", 1, _SHORT).until == _SHORT


def test_an_unknown_scenario_is_refused() -> None:
    with pytest.raises(KeyError):
        build_named_experiment("S9", 1)
