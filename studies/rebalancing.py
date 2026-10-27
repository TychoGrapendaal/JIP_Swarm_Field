"""REBALANCE: does switching robots between explorer and killer at the home base pay off?

Question: on the default fields, with the default swarm of 10 explorers and 4 killers, is a swarm
whose robots switch role as the pests need (#64) better at clearing pests than the same swarm with
fixed roles, for seeded and for hop-based killers?

Setup: the experiment plan's default scenarios (``docs/experiment-plan.md``): the small field
with one river (``RIVER_SMALL``, 2 h) and the large field with two rivers (``RIVERS_LARGE``, 4 h),
static pests, ``DEFAULT_ROBOTS``, and the killers as ``SEEDED_BEST`` or ``HOP_BEST``. Each is run
with fixed roles and rebalancing (``rebalancing_configurations``): every robot carries the brain
of an explorer and of the scenario's killer, counts the explorers, killers and pests it knows of,
and when the split stays off, the robots nearest a dock drive to it and switch there, a role
being a module swapped at the home base. The swarm keeps at least half its explorers
(``kept_explorers``). Same robots, same starts, same pests per seed.

Measured per run with ``MeasuredPestRun``, the standard measurement: the latency (mean minutes
from a pest's appearance to its treatment), the pests still infested at the end and the pests
cleared per hour; and per step the robots that are killers, the cells they drove while they were
killers, per pest cleared, and the role switches over the whole swarm. The standard measure of
cells per pest follows the robots that start as killers, which in a rebalancing swarm also fly
as explorers, so it is replaced by the cells driven in the killer role.

Hypotheses, before running:

1. Both fields start with as many pests as appear in an hour, more than 4 killers clear quickly,
   so rebalancing first turns explorers into killers, then gives some back as the backlog
   clears.
2. With seeded killers rebalancing clears more pests per hour and leaves fewer at the end than
   fixed roles: extra killers work through the backlog, and explorers return once it is gone.
3. Its gain is smaller on the large field: the docks are in two corners, so a robot drives up to
   half the field to switch, and a new killer starts from a corner at 1 m/s.
4. With hop-based killers rebalancing helps less than with seeded ones, or hurts: on the earlier
   river field hop rebalancing (switching anywhere) left far more pests than fixed roles.

The runs go through ``Experiment``, which writes every run's result to a new folder in
``output/``, with the tables, a PDF figure per field and their LaTeX fragments. Run from the
repository root with:

    uv run python -m studies.rebalancing --seeds 5

and for a quick check that it works:

    uv run python -m studies.rebalancing --maps rivers-small --seeds 1
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, stdev
from typing import TYPE_CHECKING

from swarm_field.contracts import BodyType, TimeUnit
from swarm_field.experiments import (
    HOP_BEST,
    RIVER_SMALL,
    RIVERS_LARGE,
    SEEDED_BEST,
    Experiment,
    MeasuredPestRun,
    PestResponse,
    Run,
    SwarmScenario,
    arm_name,
    by_arm,
    kept_explorers,
    markdown_table,
    mean_and_error,
    paired_difference,
    rebalancing_configurations,
    runs_for,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

MAPS: dict[str, tuple[SwarmScenario, str]] = {
    "rivers-small": (RIVER_SMALL, "32 x 32 cells, one river and rocks"),
    "rivers-large": (RIVERS_LARGE, "64 x 64 cells, two rivers and rocks"),
}
"""The default scenarios by name, with how a caption describes their field."""
METHODS: tuple[str, ...] = ("seeded", "hop")
ARMS: tuple[str, ...] = ("fixed", "rebalancing")
METHOD_COLORS: dict[str, str] = {"seeded": "#e07b28", "hop": "#1f5fa8"}
"""Warm for seeded and cool for hop-based killers, as in every figure of the report."""
METHOD_LABELS: dict[str, str] = {"seeded": "seeded", "hop": "hop-based"}
METHOD_MARKERS: dict[str, str] = {"seeded": "o", "hop": "s"}


@dataclass(frozen=True)
class RebalancingResult:
    """What one run measured.

    Attributes:
        pests: The standard pest-response measures.
        hours: How long the run lasted.
        killers: The robots that were killers, averaged over the run's steps.
        fewest_killers: The fewest killers at any step.
        most_killers: The most killers at any step.
        switches: Role switches over the whole swarm.
        killer_cells: The cells robots moved while they were killers.
    """

    pests: PestResponse
    hours: float
    killers: float
    fewest_killers: int
    most_killers: int
    switches: int
    killer_cells: int

    @property
    def cells_per_pest(self) -> float | None:
        """The cells driven in the killer role per pest cleared, if any was."""
        if self.pests.treated == 0:
            return None
        return self.killer_cells / self.pests.treated

    @property
    def cleared_per_hour(self) -> float:
        """The pests cleared per hour of the run."""
        return self.pests.treated / self.hours


def method_scenario(scenario: SwarmScenario, method: str) -> SwarmScenario:
    """``scenario`` with its killers as ``SEEDED_BEST`` (``seeded``) or ``HOP_BEST`` (``hop``)."""
    killer = SEEDED_BEST
    if method == "hop":
        killer = HOP_BEST
    killers = scenario.killers
    return scenario.without_robots(BodyType.KILLER).add_robots(killer, killers)


def measure_rebalancing(run: Run[SwarmScenario]) -> RebalancingResult:
    """Make one run and measure its pest response and its roles; a run function."""
    measured = MeasuredPestRun(run)
    world = measured.live.context.world
    last: dict[int, tuple[int, int]] = {}
    for robot_id, robot in world.robots.items():
        last[robot_id] = robot.position
    killers: list[int] = []
    killer_cells = 0
    while not measured.finished:
        measured.advance()
        count = 0
        for robot_id, robot in world.robots.items():
            if robot.body is BodyType.KILLER:
                count += 1
                x, y = last[robot_id]
                killer_cells += max(abs(robot.position[0] - x), abs(robot.position[1] - y))
            last[robot_id] = robot.position
        killers.append(count)
    robots = world.robots.values()
    return RebalancingResult(
        pests=measured.response(),
        hours=run.config.until / TimeUnit.HOUR,
        killers=mean(killers),
        fewest_killers=min(killers),
        most_killers=max(killers),
        switches=sum(robot.role_switches for robot in robots),
        killer_cells=killer_cells,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", nargs="+", choices=list(MAPS), default=list(MAPS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", type=int, default=5, help="runs per arm, seeds 0, 1, ...")
    parser.add_argument("--large-seeds", type=int, default=3, help="runs per arm, large field")
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs: list[Run[SwarmScenario]] = []
    seeds: dict[str, int] = {}
    for map_name in args.maps:
        seeds[map_name] = args.seeds
        if map_name == "rivers-large":
            seeds[map_name] = args.large_seeds
        specs: dict[str, SwarmScenario] = {}
        for method in args.methods:
            scenario = method_scenario(MAPS[map_name][0], method)
            for arm, arm_scenario in rebalancing_configurations(scenario).items():
                specs[arm_name(map_name, method, arm)] = arm_scenario
        runs.extend(runs_for(specs, range(seeds[map_name]), SwarmScenario.run_cost))

    experiment = Experiment("rebalancing")
    results = by_arm(experiment.run(runs, measure_rebalancing, args.jobs))

    tables: list[str] = []
    for map_name in args.maps:
        headers, rows = table_rows(map_name, args.methods, results)
        configuration = _configuration(map_name)
        caption = (
            "Fixed roles against rebalancing at the home base: latency (minutes from a pest's "
            "appearance to its treatment), pests infested at the end, pests cleared per hour, "
            "cells driven per pest cleared, the robots that were killers (mean and range) and "
            "the role switches; Δ open is paired by seed with fixed roles. Mean ± standard "
            f"error over {seeds[map_name]} seeds. {configuration}"
        )
        save_table(experiment, f"rebalancing_{map_name}_table", headers, rows, caption)
        tables.append(f"## {map_name}\n\n{markdown_table(headers, rows)}")
        plot_field(experiment, map_name, args.methods, results, seeds[map_name], configuration)
    text = "\n\n".join(tables)
    print(text)
    experiment.write_text("tables.md", text + "\n")
    print(f"\nResults in {experiment.directory}")


def table_rows(
    map_name: str, methods: Sequence[str], results: dict[str, list[RebalancingResult]]
) -> tuple[list[str], list[list[str]]]:
    """The table of one field: a row per method and arm."""
    headers = ["killers", "roles", "latency (min)", "open", "Δ open", "cleared / h"]
    headers.extend(["cells / pest", "killers (mean)", "killers (range)", "switches"])
    rows: list[list[str]] = []
    for method in methods:
        fixed = results[arm_name(map_name, method, "fixed")]
        for arm in ARMS:
            arm_results = results[arm_name(map_name, method, arm)]
            pests = [result.pests for result in arm_results]
            difference = "-"
            if arm != "fixed":
                open_now = [float(result.open) for result in pests]
                open_fixed = [float(result.pests.open) for result in fixed]
                difference = paired_difference(open_now, open_fixed)
            fewest = min(result.fewest_killers for result in arm_results)
            most = max(result.most_killers for result in arm_results)
            rows.append(
                [
                    METHOD_LABELS[method],
                    arm,
                    mean_and_error([result.response_minutes for result in pests]),
                    mean_and_error([float(result.open) for result in pests]),
                    difference,
                    mean_and_error([result.cleared_per_hour for result in arm_results]),
                    mean_and_error([result.cells_per_pest for result in arm_results]),
                    mean_and_error([result.killers for result in arm_results]),
                    f"{fewest}-{most}",
                    mean_and_error([float(result.switches) for result in arm_results]),
                ]
            )
    return headers, rows


def plot_field(
    experiment: Experiment,
    map_name: str,
    methods: Sequence[str],
    results: dict[str, list[RebalancingResult]],
    seeds: int,
    configuration: str,
) -> None:
    """Latency, pests open at the end and pests cleared per hour, fixed against rebalancing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.4))
    panels = (
        ("latency (min)", _latency),
        ("pests open at the end", _open),
        ("pests cleared per hour", _cleared),
    )
    for panel, (ylabel, measure) in zip(axes, panels, strict=True):
        for offset, method in enumerate(methods):
            values = [measure(results[arm_name(map_name, method, arm)]) for arm in ARMS]
            xs = [index + 0.08 * offset for index in range(len(ARMS))]
            _plot_method(panel, method, xs, values)
        panel.set_xticks(list(range(len(ARMS))), ["fixed roles", "rebalancing"])
        panel.set_xlim(-0.4, len(ARMS) - 0.6)
        panel.set_ylabel(ylabel)
        panel.grid(alpha=0.3)
    axes[0].legend()
    caption = (
        "Fixed roles against rebalancing at the home base, for seeded and hop-based killers: "
        "latency, pests still infested at the end, and pests cleared per hour; mean ± standard "
        f"error over {seeds} seeds. {configuration}"
    )
    save_figure(experiment, figure, f"rebalancing_{map_name}", caption)


def _latency(results: list[RebalancingResult]) -> list[float]:
    values: list[float] = []
    for result in results:
        if result.pests.response_minutes is not None:
            values.append(result.pests.response_minutes)
    return values


def _open(results: list[RebalancingResult]) -> list[float]:
    return [float(result.pests.open) for result in results]


def _cleared(results: list[RebalancingResult]) -> list[float]:
    return [result.cleared_per_hour for result in results]


def _plot_method(
    axes: Axes, method: str, xs: Sequence[float], values: Sequence[Sequence[float]]
) -> None:
    """One method's mean at every x, with its standard error over the seeds as error bars."""
    means: list[float] = []
    errors: list[float] = []
    for point in values:
        means.append(mean(point))
        error = 0.0
        if len(point) > 1:
            error = stdev(point) / math.sqrt(len(point))
        errors.append(error)
    axes.errorbar(
        list(xs),
        means,
        yerr=errors,
        color=METHOD_COLORS[method],
        marker=METHOD_MARKERS[method],
        capsize=3,
        label=METHOD_LABELS[method],
    )


def save_figure(experiment: Experiment, figure: Figure, stem: str, caption: str) -> None:
    """Save ``figure`` as ``STEM.pdf`` with a ``STEM.tex`` fragment that includes it.

    The same fragment as ``studies/plotting.py`` writes, so the report ``\\input``s it.
    """
    import matplotlib.pyplot as plt

    figure.tight_layout()
    figure.savefig(experiment.path(f"{stem}.pdf"))
    plt.close(figure)
    fragment = "\n".join(
        [
            r"\begin{figure}[h]",
            r"    \centering",
            rf"    \includegraphics[width=\linewidth]{{{stem}.pdf}}",
            rf"    \caption{{{_latex(caption)}}}",
            rf"    \label{{fig:{stem}}}",
            r"\end{figure}",
            "",
        ]
    )
    experiment.write_text(f"{stem}.tex", fragment)


def save_table(
    experiment: Experiment,
    stem: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    caption: str,
) -> None:
    """Write a table as a ``booktabs`` ``STEM.tex`` fragment, labelled ``tab:STEM``."""
    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        r"    \small",
        rf"    \begin{{tabular}}{{l{'c' * (len(headers) - 1)}}}",
        r"        \toprule",
        "        " + " & ".join(_latex(header) for header in headers) + r" \\",
        r"        \midrule",
    ]
    for row in rows:
        lines.append("        " + " & ".join(_latex(cell) for cell in row) + r" \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            rf"    \caption{{{_latex(caption)}}}",
            rf"    \label{{tab:{stem}}}",
            r"\end{table}",
            "",
        ]
    )
    experiment.write_text(f"{stem}.tex", "\n".join(lines))


_LATEX: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "±": r"$\pm$",
    "Δ": r"$\Delta$",
}


def _latex(text: str) -> str:
    """``text`` with LaTeX's special characters escaped."""
    escaped: list[str] = []
    for character in text:
        escaped.append(_LATEX.get(character, character))
    return "".join(escaped)


def _configuration(map_name: str) -> str:
    """The field and the swarm of a run, for a caption."""
    scenario, label = MAPS[map_name]
    robots = scenario.robots
    explorers = scenario.count() - scenario.killers
    return (
        f"Field {label}, docks at {', '.join(str(dock) for dock in scenario.map.docks)}; "
        f"{explorers} explorers at {robots.speed_mps_drone:g} m/s and {scenario.killers} killers "
        f"at {robots.speed_mps_ground:g} m/s, radio {robots.radio_range_m:g} m, "
        f"{scenario.pests.pests_per_hour:g} pests per hour and as many at the start; a switch "
        f"takes {robots.role_switch_time / TimeUnit.SECOND:g} s on a dock, and rebalancing "
        f"keeps at least {kept_explorers(scenario)} explorers."
    )


if __name__ == "__main__":
    main()
