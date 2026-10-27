"""SPEED: how fast the hop-based, the seeded and the grid-search killers clear an infested field.

Question: when 1% of the crop cells of the large field are infested at the start, and no pest
appears or spreads after that, how long do 1, 5, 10, 20 and 40 killers need until the field is
clear, with hop-based killers, seeded killers or killers that sweep the field in strips?

Setup: the default settings of ``studies/defaults.py`` on the large field with two rivers and
rocks, with the pests replaced by ``PestConfig.initial_pests`` at the start alone. The seeded and
the hop-based killers (``killer_methods``, each with its best parameters) work with the default
10 explorers, which find the pests for them; the grid search is the killers alone, each
sweeping its own strip of the field and treating what it drives over (``grid_sweep_brain``), so
it needs no explorers. With the same seed every arm starts with the same pests. A run stops as
soon as the field is clear, or at ``--max-hours``.

Measured per run with ``MeasuredPestRun``, the standard measurement: ``clean_minutes``, the
minutes until the last pest was treated, and the share of the pests treated and the mean
minutes to treat one when the run stopped.

Hypotheses, before running:

1. The time to clear falls with the killers for every method, about as 1 / n for the grid
   search, whose strips shrink with the killers, and less for the swarm, whose killers already
   reach the pests near them quickly and spend most of their time travelling between pests.
2. With few killers the seeded and the hop-based killers clear the field far sooner than the grid
   search: the explorers find all pests within minutes, and the killers drive only to the pests,
   while a lone sweeper drives over every cell of the field.
3. With many killers the grid search catches up, and may overtake the swarm at 40: every strip
   is then a few columns wide, while the swarm's killers still cross the field to the pests and
   detour over the bridges.
4. The seeded killers clear the field sooner than the hop-based killers, which follow beacons
   that the relays have to pass on hop by hop instead of the pests on a shared map.

The runs go through ``Experiment``, which writes every run's result to a new folder in
``output/``, with the table, a PDF figure and their LaTeX fragments. Run from the repository
root with:

    uv run python -m studies.clearing_speed --seeds 5 --jobs 6

and for a quick check that it works:

    uv run python -m studies.clearing_speed --map rivers-small --killers 5 --seeds 1
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from studies.baselines import grid_sweep_brain
from studies.defaults import (
    DEFAULT_ROBOTS,
    EXPLORERS,
    MAPS,
    default_pests,
    default_scenario,
    killer_methods,
    with_grid_sweepers,
)
from studies.plotting import plot_method, save_figure, save_table
from swarm_field.contracts import BodyType
from swarm_field.experiments import (
    SEEDED_BEST,
    Experiment,
    MeasuredPestRun,
    PestResponse,
    Run,
    SwarmScenario,
    arm_name,
    by_arm,
    markdown_table,
    mean_and_error,
    runs_for,
    treated_share,
)

METHODS = ("hop", "seeded", "grid")
DEFAULT_MAP = "rivers-large"
DEFAULT_KILLERS = (1, 5, 10, 20, 40)
DEFAULT_INFESTED_SHARE = 0.01
DEFAULT_MAX_HOURS = 12
"""How long a run may last before it is stopped uncleared: a lone sweeper on the large field
takes over 11 h to cover its 4096 cells at one cell per 10 s."""


def initial_pests(map_name: str, share: float) -> int:
    """The pests that infest ``share`` of the crop cells of the standard map ``map_name``."""
    field = MAPS[map_name].map
    crops = field.size**2 - len(field.blocked_cells())
    return max(1, round(share * crops))


def clearing_scenarios(map_name: str, killers: int, pests: int) -> dict[str, SwarmScenario]:
    """The arms of ``killers`` killers on ``map_name`` with ``pests`` pests at the start, by method.

    The hop-based and the seeded killers work with the default explorers; the grid search is the
    killers alone.
    """
    standard = MAPS[map_name]
    static = replace(default_pests(standard), pests_per_hour=0.0, initial_pests=pests)
    swarm = replace(default_scenario(map_name), pests=static).without_robots(BodyType.KILLER)
    methods = killer_methods(swarm.add_robots(SEEDED_BEST, killers))
    sweepers = swarm.without_robots(BodyType.SCOUT)
    methods["grid"] = with_grid_sweepers(sweepers, killers, BodyType.KILLER, grid_sweep_brain)
    return methods


def measure_clearing(run: Run[SwarmScenario]) -> PestResponse:
    """Make one run, measured as ``measure_pest_response`` does, until the field is clear."""
    measured = MeasuredPestRun(run)
    while not measured.finished and measured.tracker.clean_at is None:
        measured.advance()
    return measured.response()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", choices=list(MAPS), default=DEFAULT_MAP)
    parser.add_argument("--killers", type=int, nargs="+", default=list(DEFAULT_KILLERS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--infested", type=float, default=DEFAULT_INFESTED_SHARE, help="share infested at start"
    )
    parser.add_argument("--seeds", type=int, default=5, help="runs per arm, seeds 0, 1, ...")
    parser.add_argument("--max-hours", type=int, default=DEFAULT_MAX_HOURS)
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pests = initial_pests(args.map, args.infested)
    specs: dict[str, SwarmScenario] = {}
    for killers in args.killers:
        for method, scenario in clearing_scenarios(args.map, killers, pests).items():
            if method in args.methods:
                specs[arm_name(method, killers=killers)] = scenario.with_hours(
                    args.max_hours, warmup_hours=0
                )

    experiment = Experiment("clearing_speed")
    runs = runs_for(specs, range(args.seeds), SwarmScenario.run_cost)
    results = by_arm(experiment.run(runs, measure_clearing, args.jobs))

    headers = ["method", "killers", "clear (min)", "treated", "response (min)"]
    rows: list[list[str]] = []
    for method in args.methods:
        for killers in args.killers:
            arm_results = results[arm_name(method, killers=killers)]
            rows.append(
                [
                    method,
                    str(killers),
                    mean_and_error([result.clean_minutes for result in arm_results]),
                    treated_share(arm_results),
                    mean_and_error([result.response_minutes for result in arm_results]),
                ]
            )
    table = markdown_table(headers, rows)
    print(table)
    experiment.write_text("tables.md", table)
    configuration = _configuration(args, pests)
    save_table(
        experiment,
        "clearing_speed_table",
        headers,
        rows,
        "Minutes until the field was clear (- if some seed did not clear it within "
        f"{args.max_hours} h), the share of the pests treated and the mean minutes to treat one; "
        f"mean ± standard error over {args.seeds} seeds. {configuration}",
        "clearing_speed",
    )
    plot_clearing(experiment, args, results, configuration)


def plot_clearing(
    experiment: Experiment,
    args: argparse.Namespace,
    results: dict[str, list[PestResponse]],
    configuration: str,
) -> None:
    """Minutes until the field is clear against the killers, one line per method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6.0, 4.0))
    for method in args.methods:
        values: list[list[float]] = []
        for killers in args.killers:
            cleared: list[float] = []
            for result in results[arm_name(method, killers=killers)]:
                if result.clean_minutes is not None:
                    cleared.append(result.clean_minutes)
            values.append(cleared)
        plot_method(axes, method, [float(n) for n in args.killers], values)
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xticks(list(args.killers), [str(n) for n in args.killers])
    axes.set_xlabel("killers")
    axes.set_ylabel("minutes until the field was clear")
    axes.grid(alpha=0.3, which="both")
    axes.legend()
    save_figure(
        experiment,
        figure,
        "clearing_speed",
        "Minutes until the field was clear of the pests infesting it at the start, against the "
        "number of killers, for the hop-based, the seeded and the grid-search killers; mean ± "
        f"standard error over {args.seeds} seeds, only seeds that cleared the field within "
        f"{args.max_hours} h. {configuration}",
        "clearing_speed",
    )


def _configuration(args: argparse.Namespace, pests: int) -> str:
    """The field, the pests and the robots of a run, for a caption."""
    return (
        f"Field {MAPS[args.map].label}, {pests} static pests ({args.infested:.0%} of the crop "
        "cells) at the start and none after; the hop-based and seeded killers with "
        f"{EXPLORERS} explorers at {DEFAULT_ROBOTS.speed_mps_drone:g} m/s, the grid search "
        f"without; killers at {DEFAULT_ROBOTS.speed_mps_ground:g} m/s, radio "
        f"{DEFAULT_ROBOTS.radio_range_m:g} m."
    )


if __name__ == "__main__":
    main()
