"""Tests for measuring a run of a pest scenario."""

from dataclasses import replace

import pytest

from swarm_field.contracts import FailureMode, FailureSpec
from swarm_field.experiments import (
    EXPLORER,
    SEEDED_KILLER,
    MapConfig,
    PestConfig,
    PestRunSpec,
    Run,
    SwarmScenario,
    measure_pest_response,
)


def test_a_measured_run_drifts_once_an_hour_and_counts_every_pest_treated_or_open() -> None:
    field = SwarmScenario(map=MapConfig(size=8, obstacles=()))
    scenario = field.add_robots(EXPLORER, 2).add_robots(SEEDED_KILLER, 2)
    spec = PestRunSpec(scenario, hours=1, pests_per_hour=20.0, warmup_hours=0)

    response = measure_pest_response(Run("tiny", spec, seed=0))

    assert len(response.drift) == 1
    assert response.treated + response.open > 0
    assert response.infested >= 0
    assert 0.0 < response.coverage <= 1.0
    assert 0.0 <= response.idleness_minutes <= response.worst_idleness_minutes


def test_explorers_alone_are_measured_by_how_they_cover_the_field() -> None:
    field = SwarmScenario(map=MapConfig(size=8, obstacles=()))
    spec = PestRunSpec(field.add_robots(EXPLORER, 3), hours=1, pests_per_hour=0.0, warmup_hours=0)

    response = measure_pest_response(Run("explorers", spec, seed=0))

    assert response.coverage == 1.0
    assert response.treated == 0
    assert response.drift == (0.0,)


def test_a_warm_up_as_long_as_the_run_is_refused() -> None:
    with pytest.raises(ValueError):
        PestRunSpec(SwarmScenario(), hours=1)


def test_initial_pests_depend_on_the_seed_alone_not_on_the_swarm() -> None:
    field = SwarmScenario(map=MapConfig(size=8, obstacles=()), pests=PestConfig(initial_pests=5))

    explorers = field.add_robots(EXPLORER, 2).into_experiment("explorers").scenario
    killers = field.add_robots(SEEDED_KILLER, 3).into_experiment("killers").scenario

    assert len(explorers.initial_infestation) == 5
    assert explorers.initial_infestation == killers.initial_infestation


def test_a_field_infested_at_the_start_is_cleared_and_explored() -> None:
    pests = PestConfig(pests_per_hour=0.0, growth_rate=0.0, initial_pests=3)
    field = SwarmScenario(map=MapConfig(size=8, obstacles=()), pests=pests)
    scenario = field.add_robots(EXPLORER, 2).add_robots(SEEDED_KILLER, 2)
    spec = PestRunSpec(scenario, hours=1, pests_per_hour=0.0, warmup_hours=0)

    response = measure_pest_response(Run("clearing", spec, seed=0))

    assert response.treated == 3
    assert response.clean_minutes is not None
    assert response.explored_minutes is not None


def test_a_spec_with_failures_fails_robots_during_the_run() -> None:
    field = SwarmScenario(map=MapConfig(size=8, obstacles=()))
    failures = FailureSpec(probability=1.0, mode=FailureMode.DEAD_BATTERY)
    spec = replace(
        PestRunSpec(field.add_robots(EXPLORER, 2), hours=1, warmup_hours=0), failures=failures
    )

    assert spec.into_experiment("failing", seed=0).scenario.failures == failures
