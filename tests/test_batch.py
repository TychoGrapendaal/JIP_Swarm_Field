"""Tests for running a batch of runs in parallel into an experiment's output folder."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from swarm_field.experiments import (
    Experiment,
    Run,
    arm_name,
    by_arm,
    markdown_table,
    paired_difference,
    runs_for,
)


@dataclass(frozen=True)
class _Square:
    """A toy result."""

    value: int


def _square_seed_times(run: Run[int]) -> _Square:
    """A toy run: the arm's factor times the seed squared."""
    return _Square(run.config * run.seed**2)


def test_a_batch_runs_in_parallel_and_writes_its_results_in_run_order(tmp_path: Path) -> None:
    experiment = Experiment("toy sweep!", root=tmp_path)
    runs = runs_for({arm_name("toy", factor=1): 1, arm_name("toy", factor=2): 2}, range(3))

    results = experiment.run(runs, _square_seed_times, processes=2)

    assert re.fullmatch(r"\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_toy_sweep", experiment.directory.name)
    grouped = by_arm(results)
    assert grouped == {
        "toy/factor=1": [_Square(0), _Square(1), _Square(4)],
        "toy/factor=2": [_Square(0), _Square(2), _Square(8)],
    }
    written = json.loads((experiment.directory / "results.json").read_text())
    assert [record["name"] for record in written][:2] == [
        "toy/factor=1/seed0",
        "toy/factor=1/seed1",
    ]
    assert written[5]["result"] == {"value": 8}
    described = json.loads((experiment.directory / "experiment.json").read_text())
    assert described["arms"] == {"toy/factor=1": [0, 1, 2], "toy/factor=2": [0, 1, 2]}


def test_runs_with_the_same_arm_and_seed_are_refused(tmp_path: Path) -> None:
    experiment = Experiment("twice", root=tmp_path)
    runs = [Run("toy", 1, seed=0), Run("toy", 2, seed=0)]

    with pytest.raises(ValueError):
        experiment.run(runs, _square_seed_times, processes=1)


def test_the_table_helpers_format_markdown_and_paired_differences() -> None:
    table = markdown_table(["arm", "value"], [["a", 1], ["b", 2.5]])

    assert table == "| arm | value |\n| --- | --- |\n| a | 1 |\n| b | 2.5 |\n"
    assert paired_difference([3.0, 5.0], [1.0, 2.0]) == "+2.50 ± 0.50"
