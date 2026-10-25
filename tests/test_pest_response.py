"""Tests for measuring a run of a pest scenario."""

import pytest

from swarm_field.experiments import (
    EXPLORER,
    SEEDED_KILLER,
    MapConfig,
    PestRunSpec,
    Run,
    SwarmScenario,
    measure_pest_response,
)
from swarm_field.experiments.pest_response import radio_networks


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
    # The 300 m radio spans the 80 m field, and seeded killers keep no beacons.
    assert response.isolated == 0.0
    assert response.no_fresh_beacon is None
    assert response.cut_off_infested == 0.0
    assert response.unknown_infested <= response.infested


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


def test_radio_networks_join_robots_in_range_and_through_each_other() -> None:
    positions = {1: (0, 0), 2: (4, 0), 3: (8, 0), 4: (20, 20)}

    networks = radio_networks(positions, radio_cells=5.0)

    assert networks[1] == networks[2] == networks[3]
    assert networks[4] != networks[1]
