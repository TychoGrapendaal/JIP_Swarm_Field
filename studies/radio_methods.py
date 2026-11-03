"""RADIO: how the seeded and the hop-based killers depend on how far the radio reaches.

Question: how do the seeded killers and the hop-based killers respond to pests when every
robot's radio reaches 50, 100 or 300 m (5, 10 or 30 cells), on a field with a river and rocks and
on an open field? The seeded killers act on the pests their peers' map shows, which the radio
spreads through the swarm hop by hop; the hop-based killers follow beacons relayed by the robots
in range, so a shorter radio both thins the relay network and shortens every hop.

Setup: the default settings of ``studies/defaults.py`` (10 explorers and 4 killers, static pests
at the start and every hour) on the small field with a river and rocks and the small open field.
The two arms of a map and range differ only in the killers' brains: ``killer_methods`` gives
the tuned seeded killers (``SEEDED_BEST``) and the tuned hop killers (``HOP_BEST``), whose
beacons follow the range. Every arm runs with the same seeds, so the same pests appear at the same
times in both.

Measured per run with the standard ``measure_pest_response``: the table has its standard columns,
and the figures show the infested cells (averaged after the first hour) and the mean minutes from
a pest's appearance to its treatment, against the radio range.

Hypotheses, before running:

1. Both methods get worse as the radio gets shorter: news of a pest takes longer to reach a
   killer, so the response time and the infested cells rise.
2. The hop-based killers suffer more from a short radio than the seeded killers: a beacon only
   reaches a killer along a chain of robots within range of each other, and at 50 m the 14
   robots on a 32 x 32 field often form no chain. A seeded killer still learns of a pest later,
   through the shared map.
3. At 300 m nearly every robot hears every other, and the two methods come close: both then
   know of every pest at once, and differ only in how they divide the pests.
4. The river makes both methods slower, as ground killers detour over the bridge, but it does
   not change how they rank.

The runs go through ``Experiment``, which writes every run's result to a new folder in
``output/``, with the standard table, a PDF figure per map and their LaTeX fragments. Run from
the repository root with:

    uv run python -m studies.radio_methods --seeds 5 --jobs 6

and for a quick check that it works:

    uv run python -m studies.radio_methods --maps rivers-small --ranges 100 --seeds 1
"""

from __future__ import annotations

import argparse

from studies.defaults import (
    DEFAULT_ROBOTS,
    EXPLORERS,
    KILLERS,
    MAPS,
    default_scenario,
    killer_methods,
    measured_run,
    with_radio,
)
from studies.plotting import plot_method, save_figure, save_table
from swarm_field.experiments import (
    Experiment,
    PestResponse,
    SwarmScenario,
    arm_name,
    by_arm,
    markdown_table,
    measure_pest_response,
    pest_table_cells,
    pest_table_headers,
    runs_for,
)

METHODS = ("seeded", "hop")
DEFAULT_MAPS = ("rivers-small", "open-small")
DEFAULT_RANGES = (50.0, 100.0, 300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", nargs="+", choices=list(MAPS), default=list(DEFAULT_MAPS))
    parser.add_argument(
        "--ranges", type=float, nargs="+", default=list(DEFAULT_RANGES), help="radio ranges in m"
    )
    parser.add_argument("--seeds", type=int, default=5, help="runs per arm, seeds 0, 1, ...")
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs: dict[str, SwarmScenario] = {}
    for map_name in args.maps:
        for radio_range in args.ranges:
            scenario = with_radio(default_scenario(map_name), radio_range)
            for method, configured in killer_methods(scenario).items():
                arm = arm_name(map_name, method, radio=radio_range)
                specs[arm] = measured_run(configured, map_name)

    experiment = Experiment("radio_methods")
    runs = runs_for(specs, range(args.seeds), SwarmScenario.run_cost)
    results = by_arm(experiment.run(runs, measure_pest_response, args.jobs))

    tables: list[str] = []
    for map_name in args.maps:
        hours = MAPS[map_name].hours
        headers = pest_table_headers(hours)
        rows: list[list[str]] = []
        for method in METHODS:
            for radio_range in args.ranges:
                arm = arm_name(map_name, method, radio=radio_range)
                rows.append(pest_table_cells(f"{method} {radio_range:g} m", results[arm], hours))
        tables.append(f"## {map_name}\n\n{markdown_table(headers, rows)}")
        save_table(
            experiment,
            f"radio_methods_{map_name}_table",
            headers,
            rows,
            f"The standard measures of the seeded and hop-based killers against the radio range, "
            f"on the field {MAPS[map_name].label}; mean ± standard error over {args.seeds} seeds. "
            f"{_configuration(map_name)}",
            f"radio_methods_{map_name}",
        )
    text = "\n\n".join(tables)
    print(text)
    experiment.write_text("tables.md", text)
    plot_radio(experiment, args, results)


def plot_radio(
    experiment: Experiment, args: argparse.Namespace, results: dict[str, list[PestResponse]]
) -> None:
    """One figure per map: infested cells and response time against the radio range."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for map_name in args.maps:
        figure, (infested_axes, response_axes) = plt.subplots(1, 2, figsize=(9.0, 3.6))
        for method in METHODS:
            infested: list[list[float]] = []
            response: list[list[float]] = []
            for radio_range in args.ranges:
                arm_results = results[arm_name(map_name, method, radio=radio_range)]
                infested.append([result.infested for result in arm_results])
                responded: list[float] = []
                for result in arm_results:
                    if result.response_minutes is not None:
                        responded.append(result.response_minutes)
                response.append(responded)
            plot_method(infested_axes, method, args.ranges, infested)
            plot_method(response_axes, method, args.ranges, response)
        infested_axes.set_ylabel("infested cells")
        response_axes.set_ylabel("minutes to treat a pest")
        for axes in (infested_axes, response_axes):
            axes.set_xscale("log")
            axes.set_xticks(list(args.ranges), [f"{radio:g}" for radio in args.ranges])
            # The log scale adds minor ticks, labelled 6 x 10^1 and so on, between the ranges.
            axes.set_xticks([], minor=True)
            axes.set_xlabel("radio range (m)")
            axes.set_ylim(bottom=0)
            axes.grid(alpha=0.3)
        infested_axes.legend()
        save_figure(
            experiment,
            figure,
            f"radio_methods_{map_name}",
            f"Infested cells, averaged after the first hour (left), and the mean minutes from a "
            f"pest's appearance to its treatment (right), against the radio range, for the "
            f"seeded and hop-based killers on the field {MAPS[map_name].label}; mean ± standard "
            f"error over {args.seeds} seeds. {_configuration(map_name)}",
            f"radio_methods_{map_name}",
        )


def _configuration(map_name: str) -> str:
    """The swarm and the pests of a run on ``map_name``, for a caption."""
    standard = MAPS[map_name]
    return (
        f"{EXPLORERS} explorers at {DEFAULT_ROBOTS.speed_mps_drone:g} m/s and {KILLERS} killers "
        f"at {DEFAULT_ROBOTS.speed_mps_ground:g} m/s; {standard.pests_per_hour:g} static pests "
        f"at the start and {standard.pests_per_hour:g} per hour, over {standard.hours} h."
    )


if __name__ == "__main__":
    main()
