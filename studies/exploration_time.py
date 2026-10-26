"""EXPLORE: how long a swarm of explorers takes to explore a whole field.

Question: how long do 1, 5, 10 and 20 explorers need until every crop cell of a field has been
sensed at least once, when they follow the EXPLORE field (the seeder, ``ExplorerPreset``)
against when they sweep the field in fixed strips (the grid search, ``grid_explorer_brain``)?

Setup: the default settings of ``studies/defaults.py``, on the small and the large field with
rivers and rocks, with explorers only: no killers and no pests. The explorers start side by side
on the bottom row. A run stops as soon as the field is explored, or at ``--max-hours``.

Measured per run, from the coverage of ``MetricsCollector``, sampled every ``SAMPLE_PERIOD``:

- ``explored``: minutes until every crop cell had been sensed, or ``-`` if some never was;
- ``t50`` and ``t90``: minutes until half and nine tenths of the cells had been sensed;
- ``coverage``: the share of the cells sensed when the run stopped.

These are custom measures: the standard ``measure_pest_response`` runs to a fixed horizon, while
an exploration run only needs to last until the field is explored.

Hypotheses, before running:

1. Every configuration explores the whole field within the limit, except perhaps one explorer
   on the large field.
2. The time to explore falls about as 1 / n with the number of explorers n: each explorer covers
   its share of the cells at the same rate. It falls less than that once the explorers' trips
   from the start row to their part of the field take a noticeable part of the time.
3. The grid search explores the whole field faster than the seeder: a sweep visits every cell
   exactly once, while the seeder's explorers go for the stalest cells and cross each other's
   paths. The seeder's advantage, keeping the field fresh afterwards, is not measured here.
4. ``t90`` is closer between the methods than ``explored``: the seeder covers most of the field
   quickly and spends its last minutes on scattered cells it missed.

The runs go through ``Experiment``, which writes every run's result to a new folder in
``output/``, with the table, a PDF figure per map and their LaTeX fragments. Run from the
repository root with:

    uv run python -m studies.exploration_time --seeds 5 --jobs 6

and for a quick check that it works:

    uv run python -m studies.exploration_time --maps rivers-small --explorers 5 --seeds 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from studies.baselines import grid_explorer_brain
from studies.defaults import DEFAULT_ROBOTS, MAPS, StandardMap, with_grid_sweepers
from studies.plotting import plot_method, save_figure, save_table
from swarm_field.contracts import BodyType, Time, TimeUnit
from swarm_field.experiments import (
    EXPLORER,
    Experiment,
    LiveRun,
    PestConfig,
    Run,
    SwarmScenario,
    arm_name,
    by_arm,
    markdown_table,
    mean_and_error,
    runs_for,
)
from swarm_field.simulation import MetricsConfig

METHODS = ("seeded", "grid")
"""The explorers' methods: the EXPLORE field, and the grid search."""

DEFAULT_EXPLORERS = (1, 5, 10, 20)
DEFAULT_MAPS = ("rivers-small", "rivers-large")
DEFAULT_MAX_HOURS = {"rivers-small": 4, "rivers-large": 8}
"""How long a run may last before it is stopped unexplored, per map."""

SAMPLE_PERIOD: Time = 10 * TimeUnit.SECOND
"""How often the coverage is sampled: a few explorers explore the small field in minutes."""


@dataclass(frozen=True)
class ExplorationSpec:
    """One arm: a map, a method and a number of explorers.

    Attributes:
        map_name: A name of ``MAPS``.
        method: ``seeded`` for the EXPLORE field, ``grid`` for the grid search.
        explorers: How many explorers.
        max_hours: How long a run may last before it is stopped unexplored.
    """

    map_name: str
    method: str
    explorers: int
    max_hours: int

    def scenario(self, seed: int) -> SwarmScenario:
        """The run with ``seed``: the explorers alone on the map, without pests."""
        standard: StandardMap = MAPS[self.map_name]
        scenario = SwarmScenario(
            map=standard.map,
            pests=PestConfig(pests_per_hour=0.0, growth_rate=0.0),
            robots=DEFAULT_ROBOTS,
            metrics=MetricsConfig(period=SAMPLE_PERIOD),
            until=self.max_hours * TimeUnit.HOUR,
            seed=seed,
        )
        if self.method == "grid":
            return with_grid_sweepers(scenario, self.explorers, BodyType.SCOUT, grid_explorer_brain)
        return scenario.add_robots(EXPLORER, self.explorers)

    def cost(self) -> float:
        """A rough relative cost: the field's cells times the explorers."""
        return float(MAPS[self.map_name].map.size ** 2 * self.explorers)


@dataclass(frozen=True)
class Exploration:
    """When one run had explored its field.

    Attributes:
        explored_minutes: When every crop cell had been sensed, or ``None`` if not by the end.
        t50_minutes: When half of the crop cells had been sensed, or ``None``.
        t90_minutes: When nine tenths of them had been sensed, or ``None``.
        coverage: The share of the crop cells sensed when the run stopped.
    """

    explored_minutes: float | None
    t50_minutes: float | None
    t90_minutes: float | None
    coverage: float


def measure_exploration(run: Run[ExplorationSpec]) -> Exploration:
    """Run one arm with the run's seed, sample by sample, until its field is explored."""
    config = run.config.scenario(run.seed).into_experiment(run.arm)
    live = LiveRun.from_config(config)
    reached: dict[float, Time] = {}
    coverage = 0.0
    now: Time = 0
    while now < live.horizon:
        now += SAMPLE_PERIOD
        live.advance_to(now)
        sample = live.metrics.latest()
        if sample is None:
            continue
        coverage = sample.coverage
        for share in (0.5, 0.9, 1.0):
            if share not in reached and coverage >= share:
                reached[share] = now
        if 1.0 in reached:
            break
    return Exploration(
        explored_minutes=_minutes(reached.get(1.0)),
        t50_minutes=_minutes(reached.get(0.5)),
        t90_minutes=_minutes(reached.get(0.9)),
        coverage=coverage,
    )


def _minutes(time: Time | None) -> float | None:
    if time is None:
        return None
    return time / TimeUnit.MINUTE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", nargs="+", choices=list(MAPS), default=list(DEFAULT_MAPS))
    parser.add_argument("--explorers", type=int, nargs="+", default=list(DEFAULT_EXPLORERS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", type=int, default=5, help="runs per arm, seeds 0, 1, ...")
    parser.add_argument(
        "--max-hours", type=int, help="stop a run unexplored after this; by default per map"
    )
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs: dict[str, ExplorationSpec] = {}
    for map_name in args.maps:
        max_hours = args.max_hours or DEFAULT_MAX_HOURS.get(map_name, 8)
        for method in args.methods:
            for explorers in args.explorers:
                arm = arm_name(map_name, method, explorers=explorers)
                specs[arm] = ExplorationSpec(map_name, method, explorers, max_hours)

    experiment = Experiment("exploration_time")
    runs = runs_for(specs, range(args.seeds), ExplorationSpec.cost)
    results = by_arm(experiment.run(runs, measure_exploration, args.jobs))

    headers = ["map", "method", "explorers", "explored (min)", "t50 (min)", "t90 (min)", "coverage"]
    rows: list[list[str]] = []
    for arm, spec in specs.items():
        arm_results = results[arm]
        rows.append(
            [
                spec.map_name,
                spec.method,
                str(spec.explorers),
                mean_and_error([result.explored_minutes for result in arm_results]),
                mean_and_error([result.t50_minutes for result in arm_results]),
                mean_and_error([result.t90_minutes for result in arm_results]),
                mean_and_error([result.coverage for result in arm_results]),
            ]
        )
    table = markdown_table(headers, rows)
    print(table)
    experiment.write_text("tables.md", table)
    save_table(
        experiment,
        "exploration_time_table",
        headers,
        rows,
        "Minutes until the explorers had sensed every crop cell (explored), half and nine tenths "
        "of them, and the coverage when the run stopped; mean ± standard error over "
        f"{args.seeds} seeds. Radio range {DEFAULT_ROBOTS.radio_range_m:g} m, explorers only.",
        "exploration_time",
    )
    plot_exploration(experiment, args, specs, results)


def plot_exploration(
    experiment: Experiment,
    args: argparse.Namespace,
    specs: dict[str, ExplorationSpec],
    results: dict[str, list[Exploration]],
) -> None:
    """One figure per map: minutes until the field is explored against the explorers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for map_name in args.maps:
        figure, axes = plt.subplots(figsize=(6.0, 4.0))
        for method in args.methods:
            values: list[list[float]] = []
            for explorers in args.explorers:
                arm = arm_name(map_name, method, explorers=explorers)
                explored: list[float] = []
                for result in results[arm]:
                    if result.explored_minutes is not None:
                        explored.append(result.explored_minutes)
                values.append(explored)
            plot_method(axes, method, [float(n) for n in args.explorers], values)
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xticks(list(args.explorers), [str(n) for n in args.explorers])
        axes.set_xlabel("explorers")
        axes.set_ylabel("minutes until every cell was sensed")
        axes.grid(alpha=0.3, which="both")
        axes.legend()
        standard = MAPS[map_name]
        save_figure(
            experiment,
            figure,
            f"exploration_time_{map_name}",
            f"Minutes until a swarm of explorers had sensed every crop cell of the field "
            f"({standard.label}), against the number of explorers, for the seeded EXPLORE "
            f"field and the grid search; mean ± standard error over {args.seeds} seeds, only "
            f"seeds that explored the whole field within {specs_max_hours(specs, map_name)} h. "
            f"Radio range {DEFAULT_ROBOTS.radio_range_m:g} m, explorers at "
            f"{DEFAULT_ROBOTS.speed_mps_drone:g} m/s, no killers.",
            f"exploration_time_{map_name}",
        )


def specs_max_hours(specs: dict[str, ExplorationSpec], map_name: str) -> int:
    """The limit on a run on ``map_name``, as the arms were built."""
    for spec in specs.values():
        if spec.map_name == map_name:
            return spec.max_hours
    return 0


if __name__ == "__main__":
    main()
