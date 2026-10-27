"""Tests for measuring a run of a pest scenario."""

import pytest

from swarm_field.contracts import FailureMode, FailureSpec
from swarm_field.experiments import (
    EXPLORER,
    HOP_KILLER,
    SEEDED_KILLER,
    MapConfig,
    PestConfig,
    Run,
    SwarmScenario,
    measure_pest_response,
)

_FIELD = SwarmScenario(
    map=MapConfig(size=8, obstacles=()), pests=PestConfig(pests_per_hour=20.0)
).with_hours(1, warmup_hours=0)


def test_a_measured_run_drifts_once_an_hour_and_counts_every_pest_treated_or_open() -> None:
    scenario = _FIELD.add_robots(EXPLORER, 2).add_robots(SEEDED_KILLER, 2)

    response = measure_pest_response(Run("tiny", scenario, seed=0))

    assert len(response.drift) == 1
    assert response.treated + response.open > 0
    assert response.infested >= 0
    assert 0.0 < response.coverage <= 1.0
    assert 0.0 <= response.idleness_minutes <= response.worst_idleness_minutes
    # The 300 m radio spans the 80 m field, and seeded killers keep no beacons.
    assert response.isolated == 0.0
    assert response.no_fresh_beacon is None


def test_items_count_the_facts_and_the_beacons_that_the_messages_carry() -> None:
    responses = []
    for killer in (SEEDED_KILLER, HOP_KILLER):
        scenario = _FIELD.add_robots(EXPLORER, 2).add_robots(killer, 2)
        responses.append(measure_pest_response(Run(killer.name, scenario, seed=0)))

    # A knowledge snapshot carries many facts, and a hop robot's beacon message a TREAT and a
    # DOCK beacon, so both swarms receive more items than messages.
    for response in responses:
        assert response.items_per_robot_hour > response.messages_per_robot_hour
        by_kind = (
            response.status_items_per_robot_hour
            + response.fact_items_per_robot_hour
            + response.beacon_items_per_robot_hour
        )
        assert by_kind == pytest.approx(response.items_per_robot_hour)
    seeded, hop = responses
    assert seeded.beacon_items_per_robot_hour == 0.0
    assert hop.beacon_items_per_robot_hour > 0.0


def test_explorers_alone_are_measured_by_how_they_cover_the_field() -> None:
    scenario = _FIELD.with_pests(PestConfig(pests_per_hour=0.0)).add_robots(EXPLORER, 3)

    response = measure_pest_response(Run("explorers", scenario, seed=0))

    assert response.coverage == 1.0
    assert response.treated == 0
    assert response.drift == (0.0,)


def test_a_warm_up_as_long_as_the_run_is_refused() -> None:
    with pytest.raises(ValueError):
        SwarmScenario().with_hours(1, warmup_hours=1)


def test_initial_pests_depend_on_the_seed_alone_not_on_the_swarm() -> None:
    field = SwarmScenario(map=MapConfig(size=8, obstacles=()), pests=PestConfig(initial_pests=5))

    explorers = field.add_robots(EXPLORER, 2).into_experiment("explorers").scenario
    killers = field.add_robots(SEEDED_KILLER, 3).into_experiment("killers").scenario

    assert len(explorers.initial_infestation) == 5
    assert explorers.initial_infestation == killers.initial_infestation


def test_a_field_infested_at_the_start_is_cleared_and_explored() -> None:
    pests = PestConfig(pests_per_hour=0.0, growth_rate=0.0, initial_pests=3)
    field = SwarmScenario(map=MapConfig(size=8, obstacles=()), pests=pests)
    scenario = field.add_robots(EXPLORER, 2).add_robots(SEEDED_KILLER, 2).with_hours(1, 0)

    response = measure_pest_response(Run("clearing", scenario, seed=0))

    assert response.treated == 3
    assert response.clean_minutes is not None
    assert response.explored_minutes is not None


def test_a_scenario_with_failures_fails_robots_during_the_run() -> None:
    failures = FailureSpec(probability=1.0, mode=FailureMode.DEAD_BATTERY)
    scenario = SwarmScenario(map=MapConfig(size=8, obstacles=())).add_robots(EXPLORER, 2)

    config = scenario.with_failures(failures).into_experiment("failing")

    assert config.scenario.failures == failures
