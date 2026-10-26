"""Tests for the composition root: assembling brains and runs, and running one to its end."""

from dataclasses import replace
from random import Random

import pytest

from swarm_field.contracts import ScenarioId, Time, TimeUnit
from swarm_field.runner import (
    BrainSpec,
    ExperimentConfig,
    build_agent,
    build_experiment,
    build_run,
    run_experiment,
)
from swarm_field.simulation import MetricsConfig, build_scenario
from swarm_field.swarm import (
    BeaconConfig,
    HopField,
    Regroup,
    RegroupSeeder,
    TreatByBeacon,
    TreatInfestation,
    TreatSeeder,
)


def _config(
    label: str = "test",
    scenario: ScenarioId = ScenarioId.S1,
    seed: int = 101,
    until: Time = 60_000,
    metrics: MetricsConfig | None = None,
) -> ExperimentConfig:
    return build_experiment(
        label,
        scenario,
        seed,
        until=until,
        metrics=metrics if metrics is not None else MetricsConfig(period=30_000),
    )


def test_a_run_samples_at_the_start_and_on_every_period() -> None:
    run = run_experiment(_config(until=60_000, metrics=MetricsConfig(period=30_000)))

    assert [sample.time for sample in run] == [0, 30_000, 60_000]


def test_the_same_seed_reproduces_a_run_exactly() -> None:
    first = run_experiment(_config(seed=101))
    second = run_experiment(_config(seed=101))

    assert first == second


def test_a_different_seed_produces_a_different_run() -> None:
    first = run_experiment(_config(scenario=ScenarioId.S2, seed=101, until=120_000))
    second = run_experiment(_config(scenario=ScenarioId.S2, seed=102, until=120_000))

    assert [sample.targets_remaining for sample in first] != [
        sample.targets_remaining for sample in second
    ]


def test_the_named_seeder_reaches_the_behaviour_and_the_field_provider() -> None:
    """A spec's seeders must decide the field, not a parameter default."""
    spec = BrainSpec(seeders=("treat",), params={"seeders": ("ignored",)})
    agent = build_agent(spec, 1, build_scenario(ScenarioId.S1).field, Random(0))
    behavior = agent.behaviors[0]

    assert isinstance(behavior, TreatInfestation)
    # None rather than a KeyError: the TREAT seeder is registered but has no targets yet (#40).
    assert agent.fields.field_for(TreatSeeder.name, (0, 0), 0) is None


def test_a_hop_brain_gets_the_hop_field_and_needs_beacons() -> None:
    spec = BrainSpec(behaviors=(TreatByBeacon.name,), seeders=(HopField.name,))
    field = build_scenario(ScenarioId.S1).field

    agent = build_agent(replace(spec, beacons=BeaconConfig()), 1, field, Random(0))

    assert isinstance(agent.behaviors[0], TreatByBeacon)
    assert agent.fields.field_for(HopField.name, (0, 0), 0) is None
    with pytest.raises(ValueError):
        build_agent(spec, 1, field, Random(0))


def test_a_brain_without_a_map_has_no_quadtree_and_refuses_a_seeder() -> None:
    hop = BrainSpec(
        behaviors=(TreatByBeacon.name,),
        seeders=(HopField.name,),
        beacons=BeaconConfig(),
        keeps_map=False,
    )
    field = build_scenario(ScenarioId.S1).field

    agent = build_agent(hop, 1, field, Random(0))

    assert agent.fields.quadtree is None
    with pytest.raises(ValueError):
        build_agent(replace(hop, seeders=(HopField.name, "explore")), 1, field, Random(0))


def test_a_spec_with_regroup_assembles_the_behaviour_and_its_seeder() -> None:
    spec = BrainSpec(
        behaviors=("treat-infestation", "regroup"),
        seeders=("treat", "regroup"),
        params={"rally-point": (5, 5), "regroup-silence": TimeUnit.MINUTE},
    )
    agent = build_agent(spec, 1, build_scenario(ScenarioId.S1).field, Random(0))

    assert isinstance(agent.behaviors[1], Regroup)
    # The rally point reaches the seeder, so the REGROUP field leads there with no peer known.
    field = agent.fields.field_for(RegroupSeeder.name, (0, 0), 0)
    assert field is not None


def test_build_run_leaves_the_pest_dynamics_the_run_random_source() -> None:
    config = _config()
    ctx, _ = build_run(config)
    expected = Random(config.scenario.seed)

    assert ctx.rng.random() == expected.random()


def test_an_unregistered_navigator_is_refused() -> None:
    config = ExperimentConfig(
        label="bad",
        scenario=build_scenario(ScenarioId.S1, seed=1),
        brain=BrainSpec(navigator="a-star"),
        until=1_000,
        metrics=MetricsConfig(period=1_000),
    )

    with pytest.raises(ValueError):
        run_experiment(config)


def test_an_unregistered_behavior_is_refused() -> None:
    config = ExperimentConfig(
        label="bad",
        scenario=build_scenario(ScenarioId.S1, seed=1),
        brain=BrainSpec(behaviors=("gossip",)),
        until=1_000,
        metrics=MetricsConfig(period=1_000),
    )

    with pytest.raises(KeyError):
        run_experiment(config)


def test_an_unregistered_seeder_is_refused() -> None:
    config = ExperimentConfig(
        label="bad",
        scenario=build_scenario(ScenarioId.S1, seed=1),
        brain=BrainSpec(seeders=("gossip",)),
        until=1_000,
        metrics=MetricsConfig(period=1_000),
    )

    with pytest.raises(KeyError):
        run_experiment(config)
