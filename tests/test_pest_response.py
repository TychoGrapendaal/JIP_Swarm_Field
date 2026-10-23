"""Tests for measuring a run of a pest scenario."""

import pytest

from swarm_field.experiments import PestRunSpec, PestScenario, Run, measure_pest_response


def test_a_measured_run_drifts_once_an_hour_and_counts_every_pest_treated_or_open() -> None:
    scenario = PestScenario(size=8, obstacles=(), scouts=2, killers=2, robots_know_terrain=True)
    spec = PestRunSpec(scenario, hours=1, pests_per_hour=20.0, warmup_hours=0)

    response = measure_pest_response(Run("tiny", spec, seed=0))

    assert len(response.drift) == 1
    assert response.treated + response.open > 0
    assert response.infested >= 0


def test_a_warm_up_as_long_as_the_run_is_refused() -> None:
    with pytest.raises(ValueError):
        PestRunSpec(PestScenario(), hours=1)
