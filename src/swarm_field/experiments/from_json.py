"""Experiments described in a JSON file, so running one means editing the file, not the code.

A file names a ready scenario, the seeds and the length of the runs, the parameters every run
shares, and optionally several conditions that each change some of them:

    {
      "name": "radio-range",
      "scenario": "pests",
      "seeds": [0, 1, 2],
      "hours": 1,
      "jobs": 3,
      "parameters": {"swarm": {"explorer": 5, "seeded-killer": 3}},
      "conditions": {
        "radio-50": {"robots": {"radio_range_m": 50}},
        "radio-300": {"robots": {"radio_range_m": 300}}
      }
    }

The parameters come in groups, one per part of a ``SwarmScenario``: ``map``, ``pests``,
``robots`` and ``metrics`` take the fields of ``MapConfig``, ``PestConfig``, ``RobotConfig`` and
``MetricsConfig``, with the same names and units (times in milliseconds); ``explore`` takes the
fields of the explorers' ``ExploreConfig``, and a ``revisit_age`` given there replaces the one
set from the sweep time; ``swarm`` sets how many robots of each preset there
are, by the preset's name. A condition changes the fields it names, group by group. Every
condition runs with every seed, as the arm ``CONDITION``, and is measured with
``measure_pest_response``. Run from the repository root with:

    uv run python -m swarm_field.experiments.from_json configs/pests.json
    uv run python -m swarm_field.experiments.from_json --list pests   # every parameter

The runs, their results, the table and a copy of the file go to a new folder in ``output/``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from swarm_field.experiments.batch import (
    OUTPUT,
    Experiment,
    RunResult,
    by_arm,
    markdown_table,
    runs_for,
)
from swarm_field.experiments.maps import MapConfig
from swarm_field.experiments.pest_response import (
    PestResponse,
    PestRunSpec,
    measure_pest_response,
    pest_table_cells,
    pest_table_headers,
)
from swarm_field.experiments.presets import (
    EXPLORER,
    HOP_KILLER,
    RELAY,
    SEEDED_KILLER,
    ExplorerPreset,
    RobotPreset,
)
from swarm_field.experiments.scenario import (
    EXPLORE,
    PESTS,
    RELAYED_KILLERS,
    RIVER_SMALL,
    RIVERS_LARGE,
    PestConfig,
    RobotConfig,
    RobotGroup,
    SwarmScenario,
)
from swarm_field.simulation import MetricsConfig
from swarm_field.swarm import ExploreConfig

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

SCENARIOS: dict[str, SwarmScenario] = {
    "pests": PESTS,
    "explore": EXPLORE,
    "river-small": RIVER_SMALL,
    "rivers-large": RIVERS_LARGE,
    "relayed-killers": RELAYED_KILLERS,
}
"""The scenarios a file can name, by name."""

PRESETS: dict[str, RobotPreset] = {
    preset.name: preset for preset in (EXPLORER, RELAY, SEEDED_KILLER, HOP_KILLER)
}
"""The presets ``swarm`` can add to a scenario that has none of them yet, by name."""

GROUPS: dict[str, type[Any]] = {
    "map": MapConfig,
    "pests": PestConfig,
    "robots": RobotConfig,
    "metrics": MetricsConfig,
    "explore": ExploreConfig,
}
"""The parameter groups that set the fields of a config, by name; ``swarm`` is the other one."""

_FILE_KEYS = frozenset(
    {"name", "scenario", "seeds", "hours", "warmup_hours", "jobs", "parameters", "conditions"}
)
_CELL_FIELDS = frozenset({"obstacles", "docks"})
"""Fields that hold cells, which JSON writes as lists of lists."""


@dataclass(frozen=True)
class ExperimentFile:
    """One experiment as a JSON file describes it.

    Attributes:
        name: Names the experiment and its output folder.
        scenario: The scenario every run starts from, a key of ``SCENARIOS``.
        seeds: The seeds every condition runs with.
        hours: How long every run lasts, in whole simulated hours.
        warmup_hours: How long at the start is left out of the infested cells; shorter than
            ``hours``.
        jobs: How many runs at a time, or ``None`` for one per CPU; more is faster but takes
            more memory.
        parameters: The parameters every run shares, by group.
        conditions: Per condition name, the parameters it changes on top of ``parameters``.
            Empty means one condition, called ``name``.
    """

    name: str
    scenario: str
    seeds: tuple[int, ...] = (0,)
    hours: int = 1
    warmup_hours: int = 0
    jobs: int | None = None
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
            ValueError: If a key, the scenario, a group, a field or a preset is unknown, so that
                a typo is reported instead of silently running the default.
        """
        _check_names(data, _FILE_KEYS, "key")
        experiment = cls(
            name=data["name"],
            scenario=data["scenario"],
            seeds=tuple(data.get("seeds", (0,))),
            hours=data.get("hours", 1),
            warmup_hours=data.get("warmup_hours", 0),
            jobs=data.get("jobs"),
            parameters=data.get("parameters"),
            conditions=data.get("conditions"),
        )
        experiment.into_specs()
        return experiment

    def condition_parameters(self) -> dict[str, dict[str, Any]]:
        """The full parameters of every condition: the shared ones, then the condition's own.

        A condition replaces the fields it names and keeps the other fields of their group.
        """
        shared = self.parameters or {}
        if not self.conditions:
            return {self.name: _merge_groups(shared, {})}
        merged: dict[str, dict[str, Any]] = {}
        for condition, overrides in self.conditions.items():
            merged[condition] = _merge_groups(shared, overrides)
        return merged

    def into_specs(self) -> dict[str, PestRunSpec]:
        """One run spec per condition, by its name."""
        specs: dict[str, PestRunSpec] = {}
        for condition, parameters in self.condition_parameters().items():
            scenario = scenario_from_parameters(self.scenario, parameters)
            specs[condition] = PestRunSpec(
                scenario, self.hours, scenario.pests.pests_per_hour, self.warmup_hours
            )
        return specs


def scenario_from_parameters(name: str, parameters: dict[str, Any]) -> SwarmScenario:
    """The scenario called ``name`` with its defaults changed by ``parameters``.

    Raises:
        ValueError: If the scenario, a group, a field or a preset is unknown.
    """
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
    _check_names(parameters, {*GROUPS, "swarm"}, "parameter group")
    scenario = SCENARIOS[name]
    if "map" in parameters:
        scenario = scenario.with_map(_with_fields(scenario.map, parameters["map"], "map"))
    if "pests" in parameters:
        scenario = scenario.with_pests(_with_fields(scenario.pests, parameters["pests"], "pests"))
    if "robots" in parameters:
        robots = _with_fields(scenario.robots, parameters["robots"], "robots")
        scenario = scenario.with_robot_config(robots)
    if "metrics" in parameters:
        metrics = _with_fields(scenario.metrics, parameters["metrics"], "metrics")
        scenario = scenario.with_metrics(metrics)
    if "explore" in parameters:
        scenario = _with_explore(scenario, parameters["explore"])
    if "swarm" in parameters:
        scenario = _with_counts(scenario, parameters["swarm"])
    return scenario


def _with_fields[C: DataclassInstance](config: C, values: dict[str, Any], group: str) -> C:
    """``config`` with the fields in ``values`` replaced; cells may come as lists of lists."""
    _check_names(values, {field.name for field in fields(config)}, f"{group} field")
    replaced: dict[str, Any] = {}
    for key, value in values.items():
        if key in _CELL_FIELDS and value is not None:
            replaced[key] = tuple(tuple(cell) for cell in value)
        else:
            replaced[key] = value
    return replace(config, **replaced)


def _with_explore(scenario: SwarmScenario, values: dict[str, Any]) -> SwarmScenario:
    """``scenario`` with the ``ExploreConfig`` fields in ``values`` set on every explorer.

    A ``revisit_age`` given here is kept, rather than set from the swarm's sweep time.
    """
    groups: list[RobotGroup] = []
    for group in scenario.swarm:
        preset = group.preset
        if isinstance(preset, ExplorerPreset):
            explore = _with_fields(preset.explore, values, "explore")
            revisit_sweeps = preset.revisit_sweeps
            if "revisit_age" in values:
                revisit_sweeps = None
            preset = replace(preset, explore=explore, revisit_sweeps=revisit_sweeps)
        groups.append(RobotGroup(preset, group.count))
    return replace(scenario, swarm=tuple(groups))


def _with_counts(scenario: SwarmScenario, counts: dict[str, int]) -> SwarmScenario:
    """``scenario`` with as many robots of each named preset as ``counts`` says.

    A preset the scenario has keeps its place and parameters; a ready preset it lacks is added
    after the others.

    Raises:
        ValueError: If a name is neither a preset of the scenario nor a ready preset.
    """
    known = {group.preset.name for group in scenario.swarm} | set(PRESETS)
    _check_names(counts, known, "preset")
    groups: list[RobotGroup] = []
    for group in scenario.swarm:
        groups.append(RobotGroup(group.preset, counts.get(group.preset.name, group.count)))
    present = {group.preset.name for group in scenario.swarm}
    for name, count in counts.items():
        if name not in present:
            groups.append(RobotGroup(PRESETS[name], count))
    return replace(scenario, swarm=tuple(groups))


def _merge_groups(shared: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """The groups of ``shared`` with the fields of ``overrides`` put over them, group by group.

    Raises:
        ValueError: If a group is unknown, e.g. a field given outside its group.
    """
    _check_names(shared, {*GROUPS, "swarm"}, "parameter group")
    _check_names(overrides, {*GROUPS, "swarm"}, "parameter group")
    merged: dict[str, Any] = {}
    for group in (*shared, *overrides):
        merged[group] = {**shared.get(group, {}), **overrides.get(group, {})}
    return merged


def _check_names(values: dict[str, Any], known: set[str] | frozenset[str], what: str) -> None:
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"unknown {what}(s) {unknown}; known: {sorted(known)}")


def run_file(
    experiment: ExperimentFile, root: Path = OUTPUT
) -> tuple[Experiment, list[RunResult[PestRunSpec, PestResponse]]]:
    """Run every condition with every seed into a new folder under ``root``, and write the table.

    Returns:
        The experiment, whose ``directory`` holds the results, and every run's result.
    """
    batch = Experiment(experiment.name, root)
    runs = runs_for(experiment.into_specs(), experiment.seeds, PestRunSpec.cost)
    results = batch.run(runs, measure_pest_response, experiment.jobs)
    batch.write_text("tables.md", format_table(experiment, results))
    return batch, results


def format_table(
    experiment: ExperimentFile, results: list[RunResult[PestRunSpec, PestResponse]]
) -> str:
    """One row per condition: the standard columns of ``measure_pest_response``, over the seeds."""
    rows: list[list[str]] = []
    for condition, condition_results in by_arm(results).items():
        rows.append(pest_table_cells(condition, condition_results, experiment.hours))
    headers = pest_table_headers(experiment.hours, first="condition")
    return markdown_table(headers, rows)


def format_parameters(name: str) -> str:
    """Every parameter of the scenario called ``name`` by group, with its value there."""
    scenario = SCENARIOS[name]
    lines: list[str] = []
    configs: dict[str, Any] = {
        "map": scenario.map,
        "pests": scenario.pests,
        "robots": scenario.robots,
        "metrics": scenario.metrics,
        "explore": ExploreConfig(),
    }
    for group, config in configs.items():
        lines.append(f"{group}:")
        for field in fields(config):
            lines.append(f"  {field.name} = {getattr(config, field.name)!r}")
    lines.append("swarm:")
    for group in scenario.swarm:
        lines.append(f"  {group.preset.name} = {group.count}")
    lines.append(f"  (ready presets: {', '.join(sorted(PRESETS))})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path, nargs="?", help="the experiment's JSON file")
    parser.add_argument("--list", choices=sorted(SCENARIOS), help="print a scenario's parameters")
    args = parser.parse_args()
    if args.list is not None:
        print(format_parameters(args.list))
        return
    if args.file is None:
        parser.error("give an experiment file, or --list SCENARIO")

    experiment = ExperimentFile.from_path(args.file)
    batch, results = run_file(experiment)
    batch.write_text("experiment-file.json", args.file.read_text(encoding="utf-8"))
    print(format_table(experiment, results))
    print(f"\nResults in {batch.directory}")


if __name__ == "__main__":
    main()
