"""Tests for the seeded experiment runner and config sweeps (B.5)."""

import csv
import json
from pathlib import Path
from random import Random

import pytest

from swarm_field.contracts import FailureMode, FailureSpec, ScenarioId, Time, TimeUnit
from swarm_field.experiments import build_named_experiment
from swarm_field.runner import (
    BrainSpec,
    ExperimentConfig,
    RobustnessCase,
    RunMetadata,
    RunSummary,
    build_agent,
    build_experiment,
    build_run,
    export_summary_csv,
    export_summary_json,
    robustness_case_config,
    run_experiment,
    sweep,
)
from swarm_field.simulation import MetricsConfig, build_scenario
from swarm_field.swarm import Regroup, RegroupSeeder, TreatInfestation, TreatSeeder


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

    assert [sample.time for sample in run.series] == [0, 30_000, 60_000]


def test_the_summary_takes_its_whole_run_figures_from_the_last_sample() -> None:
    run = run_experiment(_config(until=60_000))

    assert run.targets_remaining == run.series[-1].targets_remaining
    assert run.coverage == pytest.approx(run.series[-1].coverage)
    assert run.energy_per_area == pytest.approx(run.series[-1].energy_per_area)
    assert run.messages_sent == run.series[-1].messages_sent


def test_the_summary_averages_detection_latency_over_the_series() -> None:
    run = run_experiment(_config(until=60_000, metrics=MetricsConfig(period=30_000)))
    present = [
        sample.detection_latency_ms
        for sample in run.series
        if sample.detection_latency_ms is not None
    ]
    expected = None if not present else sum(present) / len(present)

    assert run.mean_detection_latency_ms == pytest.approx(expected)


def test_a_summary_records_the_scenario_and_seed_it_came_from() -> None:
    run = run_experiment(_config(scenario=ScenarioId.S3, seed=103))

    assert (run.scenario, run.seed) == ("S3", 103)


def test_the_same_seed_reproduces_a_run_exactly() -> None:
    first = run_experiment(_config(seed=101))
    second = run_experiment(_config(seed=101))

    assert first.series == second.series


def test_a_different_seed_produces_a_different_run() -> None:
    first = run_experiment(_config(scenario=ScenarioId.S2, seed=101, until=120_000))
    second = run_experiment(_config(scenario=ScenarioId.S2, seed=102, until=120_000))

    assert [sample.targets_remaining for sample in first.series] != [
        sample.targets_remaining for sample in second.series
    ]


def test_sweep_runs_every_configuration_once_in_order() -> None:
    configs = [_config(label="a"), _config(label="b")]

    assert [run.label for run in sweep(configs)] == ["a", "b"]


def test_to_row_flattens_the_summary_into_one_record() -> None:
    run = run_experiment(_config())

    row = run.to_row()

    assert row["label"] == "test"
    assert row["scenario"] == "S1"
    assert row["samples"] == len(run.series)


def test_export_summary_csv_writes_one_row_per_run(tmp_path: Path) -> None:
    runs = sweep(
        [
            _config(label="S1/101", scenario=ScenarioId.S1, until=30_000),
            _config(label="S2/101", scenario=ScenarioId.S2, until=30_000),
        ]
    )
    path = tmp_path / "summary.csv"

    export_summary_csv(runs, path)

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == ["S1/101", "S2/101"]
    assert [int(row["seed"]) for row in rows] == [101, 101]


def test_export_summary_json_round_trips(tmp_path: Path) -> None:
    runs = sweep([_config(label="S1/101", until=30_000)])
    path = tmp_path / "summary.json"

    export_summary_json(runs, path)

    assert json.loads(path.read_text(encoding="utf-8")) == [run.to_row() for run in runs]


def test_export_writes_into_a_missing_directory(tmp_path: Path) -> None:
    run = run_experiment(_config())
    path = tmp_path / "nested" / "summary.csv"

    export_summary_csv([run], path)

    assert path.exists()


def test_exporting_no_runs_is_refused() -> None:
    with pytest.raises(ValueError):
        export_summary_csv([], Path("unused.csv"))


def test_a_run_writes_its_own_series_as_csv_and_json(tmp_path: Path) -> None:
    run = run_experiment(_config(label="S1-101"))

    csv_path, json_path = run.export(tmp_path)

    assert csv_path.name == "S1-101.csv"
    assert json_path.name == "S1-101.json"
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["time"] == 0


def test_a_sweep_label_never_becomes_a_directory(tmp_path: Path) -> None:
    """Sweep labels read as SCENARIO/SEED, and a label is display data, not a path."""
    run = run_experiment(_config(label="S2/failure-0.5"))

    csv_path, _ = run.export(tmp_path)

    assert csv_path.name == "S2_failure-0.5.csv"
    assert csv_path.parent == tmp_path


def test_a_label_cannot_escape_the_output_directory(tmp_path: Path) -> None:
    run = run_experiment(_config(label="../../etc/passwd"))

    csv_path, _ = run.export(tmp_path)

    assert csv_path.parent == tmp_path


def test_the_named_seeder_reaches_the_behaviour_and_the_field_provider() -> None:
    """A spec's seeders must decide the field, not a parameter default."""
    spec = BrainSpec(seeders=("treat",), params={"seeders": ("ignored",)})
    agent = build_agent(spec, 1, build_scenario(ScenarioId.S1).field, Random(0))
    behavior = agent.behaviors[0]

    assert isinstance(behavior, TreatInfestation)
    # None rather than a KeyError: the TREAT seeder is registered but has no targets yet (#40).
    assert agent.fields.field_for(TreatSeeder.name, (0, 0), 0) is None


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


def test_a_summary_needs_at_least_one_sample() -> None:
    with pytest.raises(ValueError):
        RunSummary.from_run(_config(), [])


def test_a_summary_row_records_how_the_run_was_set_up() -> None:
    config = _config(scenario=ScenarioId.S2, until=30_000)

    row = run_experiment(config).to_row()

    assert row["horizon_ms"] == 30_000
    assert row["metrics_period_ms"] == 30_000
    assert row["robots"] == len(config.scenario.robots)
    assert row["brain"] == (
        "behaviors=treat-infestation seeders=treat navigator=potential-field sharing=full"
    )
    assert row["brains_by_body"] == ""
    assert row["failure_probability"] == 0.0


def test_the_metadata_names_a_robustness_condition_and_a_brain_per_body() -> None:
    case = RobustnessCase(
        label="offline", failures=FailureSpec(probability=0.25, mode=FailureMode.OFFLINE)
    )

    faulty = RunMetadata.from_config(robustness_case_config(_config(), case))
    pests = RunMetadata.from_config(build_named_experiment("pests", 1))

    assert (faulty.failure_probability, faulty.failure_mode) == (0.25, "OFFLINE")
    assert pests.bodies == "KILLER=3 SCOUT=5"
    assert pests.brains_by_body.startswith("KILLER: behaviors=treat-infestation+explore")
    assert pests.robots_know_terrain
