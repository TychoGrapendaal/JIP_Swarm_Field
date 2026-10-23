"""Experiments described in a JSON file, so running one means editing the file, not the code.

A file names a hand-built scenario (``pests`` or ``explore``), the seeds and the horizon, the
parameters every run shares, and optionally several conditions that each change some of them:

    {
      "name": "radio-range",
      "scenario": "pests",
      "seeds": [0, 1, 2],
      "hours": 1,
      "jobs": 3,
      "parameters": {"scouts": 5, "killers": 3},
      "conditions": {"radio-50": {"radio_range_m": 50}, "radio-300": {"radio_range_m": 300}}
    }

A parameter is a field of ``PestScenario`` or ``ExploreScenario``, with the same name and unit
(times in milliseconds); ``explore`` and ``arbiter`` take an object of ``ExploreConfig`` and
``ArbiterConfig`` fields. Every condition runs with every seed, labelled ``CONDITION/SEED``.
Run from the repository root with:

    uv run python -m swarm_field.experiments.from_json configs/pests.json
    uv run python -m swarm_field.experiments.from_json --list pests   # every parameter

The runs' metrics series, one summary row per run and a copy of the file go to ``runs/NAME/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path
from statistics import mean
from typing import Any

from swarm_field.contracts import TimeUnit
from swarm_field.experiments.explore import ExploreScenario
from swarm_field.experiments.pests import PestScenario
from swarm_field.runner import (
    ExperimentConfig,
    RunSummary,
    export_summary_csv,
    export_summary_json,
    run_experiment,
    sweep,
)
from swarm_field.swarm import ArbiterConfig, ExploreConfig

Scenario = PestScenario | ExploreScenario

SCENARIOS: dict[str, type[PestScenario] | type[ExploreScenario]] = {
    "pests": PestScenario,
    "explore": ExploreScenario,
}
"""The scenarios a file can name, by name."""

_FILE_KEYS = frozenset({"name", "scenario", "seeds", "hours", "jobs", "parameters", "conditions"})

_TABLE_MEASURES: tuple[tuple[str, str], ...] = (
    ("coverage", "coverage"),
    ("targets_remaining", "infested"),
    ("mean_detection_latency_ms", "latency"),
    ("messages_sent", "sent"),
)
"""The summary measures the printed table shows, with their column titles."""


@dataclass(frozen=True)
class ExperimentFile:
    """One experiment as a JSON file describes it.

    Attributes:
        name: Names the experiment and its output directory.
        scenario: The scenario every run starts from, a key of ``SCENARIOS``.
        seeds: The seeds every condition runs with.
        hours: How long every run lasts, or ``None`` for the scenario's own horizon.
        jobs: How many runs at a time; more is faster but takes more memory.
        parameters: The scenario parameters every run shares.
        conditions: Per condition name, the parameters it changes on top of ``parameters``.
            Empty means one condition, called ``name``.
    """

    name: str
    scenario: str
    seeds: tuple[int, ...] = (0,)
    hours: float | None = None
    jobs: int = 1
    parameters: dict[str, Any] | None = None
    conditions: dict[str, dict[str, Any]] | None = None

    @classmethod
    def from_path(cls, path: Path) -> ExperimentFile:
        """Read the experiment from the JSON file at ``path``."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentFile:
        """Read the experiment from a parsed JSON object.

        Raises:
            ValueError: If a key, the scenario or a parameter is unknown, so that a typo is
                reported instead of silently running the default.
        """
        unknown = sorted(set(data) - _FILE_KEYS)
        if unknown:
            raise ValueError(f"unknown key(s) {unknown}; known: {sorted(_FILE_KEYS)}")
        experiment = cls(
            name=data["name"],
            scenario=data["scenario"],
            seeds=tuple(data.get("seeds", (0,))),
            hours=data.get("hours"),
            jobs=data.get("jobs", 1),
            parameters=data.get("parameters"),
            conditions=data.get("conditions"),
        )
        experiment.into_configs()
        return experiment

    def condition_parameters(self) -> dict[str, dict[str, Any]]:
        """The full parameters of every condition: the shared ones, then the condition's own."""
        shared = self.parameters or {}
        if not self.conditions:
            return {self.name: dict(shared)}
        merged: dict[str, dict[str, Any]] = {}
        for condition, overrides in self.conditions.items():
            merged[condition] = {**shared, **overrides}
        return merged

    def into_configs(self) -> tuple[ExperimentConfig, ...]:
        """One run per condition and seed, labelled ``CONDITION/SEED``, conditions first."""
        configs: list[ExperimentConfig] = []
        for condition, parameters in self.condition_parameters().items():
            for seed in self.seeds:
                scenario = scenario_from_parameters(self.scenario, parameters, seed, self.hours)
                config = scenario.into_experiment(f"{condition}/{seed}")
                configs.append(replace(config, scenario_name=self.scenario))
        return tuple(configs)


def scenario_from_parameters(
    name: str, parameters: dict[str, Any], seed: int, hours: float | None = None
) -> Scenario:
    """The scenario called ``name`` with its defaults changed by ``parameters``.

    Raises:
        ValueError: If the scenario or a parameter is unknown.
    """
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
    default = SCENARIOS[name]()
    settable = [field.name for field in fields(default) if field.name != "seed"]
    _check_names(parameters, settable, f"{name} parameter")

    values: dict[str, Any] = {}
    for key, value in parameters.items():
        if key in ("obstacles", "docks"):
            # JSON has no tuples: the cells arrive as lists, and ``null`` keeps the default.
            values[key] = value
            if value is not None:
                values[key] = tuple(tuple(cell) for cell in value)
        elif key == "explore":
            _check_names(value, [field.name for field in fields(ExploreConfig)], "explore field")
            values[key] = ExploreConfig(**value)
        elif key == "arbiter":
            _check_names(value, [field.name for field in fields(ArbiterConfig)], "arbiter field")
            values[key] = ArbiterConfig(**value)
        else:
            values[key] = value
    scenario = replace(default, seed=seed, **values)
    if hours is not None:
        scenario = replace(scenario, until=int(hours * TimeUnit.HOUR))
    return scenario


def _check_names(values: dict[str, Any], known: list[str], what: str) -> None:
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"unknown {what}(s) {unknown}; known: {sorted(known)}")


def run_file(experiment: ExperimentFile, out: Path) -> tuple[RunSummary, ...]:
    """Run every condition and seed, and write the series and the summaries under ``out``."""
    configs = experiment.into_configs()
    if experiment.jobs > 1:
        with ProcessPoolExecutor(experiment.jobs) as pool:
            runs = tuple(pool.map(run_experiment, configs))
    else:
        runs = sweep(configs)
    series = out / "series"
    series.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run.export(series)
    export_summary_csv(runs, out / "summary.csv")
    export_summary_json(runs, out / "summary.json")
    return runs


def format_table(runs: tuple[RunSummary, ...]) -> str:
    """One row per condition: the mean over its seeds of a few measures, latency in minutes."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        condition = run.label.rsplit("/", 1)[0]
        groups.setdefault(condition, []).append(run.to_row())
    width = max(len("condition"), *(len(condition) for condition in groups))
    lines = [f"{'condition':<{width}}" + "".join(f"  {title:>9}" for _, title in _TABLE_MEASURES)]
    for condition, rows in groups.items():
        cells = ""
        for key, _title in _TABLE_MEASURES:
            values = [row[key] for row in rows if row[key] is not None]
            if not values:
                cells += f"  {'-':>9}"
                continue
            value = mean(values)
            if key == "mean_detection_latency_ms":
                value /= TimeUnit.MINUTE
            cells += f"  {value:>9.3g}"
        lines.append(f"{condition:<{width}}{cells}")
    return "\n".join(lines)


def format_parameters(name: str) -> str:
    """Every parameter of the scenario called ``name``, with its default value."""
    default = SCENARIOS[name]()
    lines: list[str] = []
    for field in fields(default):
        if field.name != "seed":
            lines.append(f"{field.name} = {getattr(default, field.name)!r}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path, nargs="?", help="the experiment's JSON file")
    parser.add_argument("--out", type=Path, help="where the results go; runs/NAME by default")
    parser.add_argument("--list", choices=sorted(SCENARIOS), help="print a scenario's parameters")
    args = parser.parse_args()
    if args.list is not None:
        print(format_parameters(args.list))
        return
    if args.file is None:
        parser.error("give an experiment file, or --list SCENARIO")

    experiment = ExperimentFile.from_path(args.file)
    out: Path = args.out or Path("runs") / experiment.name
    runs = run_file(experiment, out)
    shutil.copyfile(args.file, out / "experiment.json")
    print(format_table(runs))
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
