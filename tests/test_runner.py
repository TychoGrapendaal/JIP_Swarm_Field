"""Tests for the seeded experiment runner and config sweeps (B.5)."""

import csv
import json
from pathlib import Path
from random import Random

import pytest

from swarm_field.contracts import ScenarioId, Time
from swarm_field.runner import (
    BrainSpec,
    ExperimentConfig,
    RunSummary,
    build_agent,
    build_experiment,
    build_run,
    export_summary_csv,
    export_summary_json,
    run_experiment,
    sweep,
    sweep_scenarios,
)
from swarm_field.simulation import MetricsConfig, build_scenario
from swarm_field.swarm import TreatInfestation, TreatObjective


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


def test_sweep_scenarios_covers_the_cross_product() -> None:
    runs = sweep_scenarios([ScenarioId.S1, ScenarioId.S2], seeds=[101, 102], until=30_000)

    assert [(run.scenario, run.seed) for run in runs] == [
        ("S1", 101),
        ("S1", 102),
        ("S2", 101),
        ("S2", 102),
    ]


def test_sweep_scenarios_labels_each_run_with_its_scenario_and_seed() -> None:
    runs = sweep_scenarios([ScenarioId.S1], seeds=[101, 102], until=30_000)

    assert [run.label for run in runs] == ["S1/101", "S1/102"]


def test_sweep_scenarios_with_no_seeds_yields_nothing() -> None:
    assert sweep_scenarios([ScenarioId.S1], seeds=[], until=1_000) == ()


def test_to_row_flattens_the_summary_into_one_record() -> None:
    run = run_experiment(_config())

    row = run.to_row()

    assert row["label"] == "test"
    assert row["scenario"] == "S1"
    assert row["samples"] == len(run.series)


def test_export_summary_csv_writes_one_row_per_run(tmp_path: Path) -> None:
    runs = sweep_scenarios([ScenarioId.S1, ScenarioId.S2], seeds=[101], until=30_000)
    path = tmp_path / "summary.csv"

    export_summary_csv(runs, path)

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == ["S1/101", "S2/101"]
    assert [int(row["seed"]) for row in rows] == [101, 101]


def test_export_summary_json_round_trips(tmp_path: Path) -> None:
    runs = sweep_scenarios([ScenarioId.S1], seeds=[101], until=30_000)
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


def test_the_named_objective_reaches_the_assembled_brain() -> None:
    """A spec's objective must decide the objective, not a parameter default."""
    spec = BrainSpec(objective="treat", params={"objective": "ignored"})
    agent = build_agent(spec, 1, build_scenario(ScenarioId.S1).field, Random(0))
    behavior = agent.behaviors[0]

    assert isinstance(behavior, TreatInfestation)
    assert isinstance(behavior.objective, TreatObjective)


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


def test_a_summary_needs_at_least_one_sample() -> None:
    with pytest.raises(ValueError):
        RunSummary.from_series("x", "S1", 1, [])
