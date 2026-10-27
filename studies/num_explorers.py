"""Study how many explorers a fixed group of killers needs: NUM_EXPLORERS.

Question: on the two river fields, with a fixed number of killers and a fixed pest rate, how do
the pest response (infested cells, response time, treated share) and the explorers' coverage
change with the number of explorers, for seeded and for hop-based killers? There are no relays
and no mock sources: pests are found by the robots' own sensing (mostly the explorers', though
killers sense the cells they cross too), and messages hop between explorers and killers, so with
a short radio a few explorers form a sparse, broken network and many a denser one.

Every arm is the field's default experiment (``RIVER_SMALL`` or ``RIVERS_LARGE``: the team's
default robots, ``DEFAULT_ROBOTS``, and static pests that start on the field) with its swarm
replaced by explorers (``EXPLORER``) and killers of one tuned preset, ``SEEDED_BEST`` or
``HOP_BEST``. Two pest rates are run, a low and a high load per killer; as many pests as arrive
in an hour are on the field at the start.
All arms of a field share the same seeds. Hop and seeded arms with the same explorer count see
the same pests, but the pest stream depends on the number of robots, so arms with different
explorer counts do not: compare across explorer counts with that in mind.

The runs are measured with ``measure_pest_response`` and run in parallel by ``Experiment``,
which writes them, the tables, the PDF figures and their LaTeX fragments to a new folder in
``output/``. Not part of ``pytest``; run it from the repository root with:

    uv run python -m studies.num_explorers --seeds 3

and redraw the figures of an earlier run without simulating with:

    uv run python -m studies.num_explorers --replot output/<date>_num_explorers
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, stdev

from swarm_field.contracts import TimeUnit
from swarm_field.experiments import (
    EXPLORER,
    HOP_BEST,
    RIVER_SMALL,
    RIVERS_LARGE,
    SEEDED_BEST,
    Experiment,
    PestResponse,
    RobotPreset,
    SwarmScenario,
    arm_name,
    by_arm,
    markdown_table,
    measure_pest_response,
    paired_difference,
    pest_table_cells,
    pest_table_headers,
    runs_for,
    treated_share,
)

logger = logging.getLogger(__name__)

KILLERS: dict[str, RobotPreset] = {"seeded": SEEDED_BEST, "hop": HOP_BEST}
"""The pest responses compared, by the name used in arm names, tables and figures."""
RADIO_RANGE_M = RIVER_SMALL.robots.radio_range_m
"""The default radio, ``DEFAULT_ROBOTS``': 100 m, ten cells."""
LOADS: dict[str, float] = {"low": 2.0, "high": 4.0}
"""Pests per killer and hour of the two infestation modes; ``PESTS_PER_KILLER_HOUR`` is 3."""
COLORS: dict[str, str] = {"hop": "#1f5fa8", "seeded": "#e07020"}
"""Cool blue for hop-based, warm orange for seeded, as the plotting guidelines ask."""
MARKERS: dict[str, str] = {"low": "o", "high": "s"}
LINESTYLES: dict[str, str] = {"low": "--", "high": "-"}


@dataclass(frozen=True)
class FieldSetup:
    """One field of the study and how its runs are sized.

    Attributes:
        name: The field's name in arm names.
        scenario: The ready scenario whose map is used.
        killers: Killers in every arm on this field.
        explorers: The explorer counts swept.
        hours: How long a run lasts.
        warmup_hours: How long at the start is left out of ``infested``.
    """

    name: str
    scenario: SwarmScenario
    killers: int
    explorers: tuple[int, ...]
    hours: int
    warmup_hours: int


FIELDS: dict[str, FieldSetup] = {
    "river-small": FieldSetup("river-small", RIVER_SMALL, 5, (2, 5, 10, 20, 40), 2, 0),
    "rivers-large": FieldSetup("rivers-large", RIVERS_LARGE, 10, (5, 10, 20, 40, 80), 4, 0),
}


@dataclass(frozen=True)
class ArmKey:
    """What an arm varies, read back from its name.

    Attributes:
        field: The field's name.
        killer: ``seeded`` or ``hop``.
        explorers: The number of explorers.
        load: The infestation mode, ``low`` or ``high``.
    """

    field: str
    killer: str
    explorers: int
    load: str

    def into_arm_name(self) -> str:
        return arm_name(self.field, self.killer, self.load, explorers=self.explorers)

    @classmethod
    def from_arm_name(cls, name: str) -> ArmKey:
        """The key of an arm name made by ``into_arm_name``."""
        field_name, killer, load, explorers = name.split("/")
        return cls(field_name, killer, int(explorers.removeprefix("explorers=")), load)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fields", nargs="+", choices=list(FIELDS), default=list(FIELDS))
    parser.add_argument(
        "--explorers", type=int, nargs="+", help="explorer counts; each field's own if not given"
    )
    parser.add_argument("--loads", nargs="+", choices=list(LOADS), default=list(LOADS))
    parser.add_argument("--radio", type=float, default=RADIO_RANGE_M, help="radio range in m")
    parser.add_argument("--seeds", type=int, default=3, help="runs per arm, seeds 0, 1, ...")
    parser.add_argument("--hours", type=int, help="simulated hours; each field's own if not given")
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    parser.add_argument("--replot", type=Path, help="only redraw the figures of this folder")
    return parser.parse_args()


def explorer_scenario(
    setup: FieldSetup, killer: RobotPreset, explorers: int, rate: float, radio_range_m: float
) -> SwarmScenario:
    """The default experiment of ``setup``'s field with explorers and killers of one preset.

    There are no relays and no mock sources. ``rate`` pests arrive per hour, and as many are on
    the field at the start.
    """
    base = setup.scenario
    robots = replace(base.robots, radio_range_m=radio_range_m)
    pests = replace(base.pests, pests_per_hour=rate, initial_pests=round(rate))
    return (
        base.without_robots()
        .with_pests(pests)
        .with_robot_config(robots)
        .add_robots(EXPLORER, explorers)
        .add_robots(killer, setup.killers)
        .with_hours(setup.hours, setup.warmup_hours)
    )


def build_scenarios(args: argparse.Namespace) -> dict[str, SwarmScenario]:
    """One scenario per field, killer, load and explorer count, by arm name."""
    scenarios: dict[str, SwarmScenario] = {}
    for field_name in args.fields:
        setup = FIELDS[field_name]
        if args.hours is not None:
            setup = replace(setup, hours=args.hours)
        counts = setup.explorers
        if args.explorers is not None:
            counts = tuple(args.explorers)
        for load in args.loads:
            rate = LOADS[load] * setup.killers
            for killer_name, killer in KILLERS.items():
                for explorers in counts:
                    key = ArmKey(field_name, killer_name, explorers, load)
                    scenarios[key.into_arm_name()] = explorer_scenario(
                        setup, killer, explorers, rate, args.radio
                    )
    return scenarios


def write_tables(
    experiment: Experiment,
    scenarios: dict[str, SwarmScenario],
    results: dict[str, list[PestResponse]],
) -> str:
    """One standard table per field and load, with hop compared to seeded seed by seed."""
    sections: list[str] = []
    groups: dict[tuple[str, str], list[ArmKey]] = {}
    for arm in scenarios:
        key = ArmKey.from_arm_name(arm)
        groups.setdefault((key.field, key.load), []).append(key)
    for (field_name, load), keys in groups.items():
        first = scenarios[keys[0].into_arm_name()]
        hours = first.until // TimeUnit.HOUR
        warmup_hours = first.metrics.measure_from // TimeUnit.HOUR
        headers = pest_table_headers(hours, first="killer / explorers")
        headers.extend(["treated %", "Δ infested vs seeded"])
        rows: list[list[str]] = []
        for key in sorted(keys, key=lambda item: (item.killer, item.explorers)):
            arm_results = results[key.into_arm_name()]
            difference = "-"
            if key.killer != "seeded":
                seeded_key = replace(key, killer="seeded")
                baseline = [result.infested for result in results[seeded_key.into_arm_name()]]
                infested = [result.infested for result in arm_results]
                difference = paired_difference(infested, baseline)
            cells = pest_table_cells(f"{key.killer} / {key.explorers}", arm_results, hours)
            cells.extend([treated_share(arm_results), difference])
            rows.append(cells)
        title = (
            f"{field_name}, {load} infestation: {first.killers} killers, "
            f"{first.pests.pests_per_hour:g} pests per hour ({first.pests.initial_pests} at the "
            f"start), {first.robots.radio_range_m:g} m radio, no relays or mock sources, "
            f"static pests, {hours} h (warm-up {warmup_hours} h)"
        )
        sections.append(f"## {title}\n\n{markdown_table(headers, rows)}")
    text = "\n\n".join(sections)
    experiment.write_text("tables.md", text)
    return text


@dataclass(frozen=True)
class SeriesPoint:
    """The mean and standard error of one measure of one arm.

    Attributes:
        explorers: The arm's explorer count.
        mean: The mean over the seeds.
        error: The standard error of the mean, 0 for fewer than two values.
    """

    explorers: int
    mean: float
    error: float


METRICS: dict[str, str] = {
    "infested": "Infested cells (mean after warm-up)",
    "response_minutes": "Response time (min)",
    "treated_share": "Treated share",
    "coverage": "Coverage by end of run",
    "idleness_minutes": "Cell idleness (min)",
    "cells_per_treatment": "Killer cells moved per treated pest",
}
"""The measures plotted against the explorer count, with their axis labels."""


def _number(value: object) -> float | None:
    """``value`` as a float, or ``None`` if it is not a number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _record_value(result: dict[str, object], metric: str) -> float | None:
    """``metric`` of one run's ``PestResponse`` as written to ``results.json``."""
    if metric == "treated_share":
        treated = _number(result["treated"]) or 0.0
        total = treated + (_number(result["open"]) or 0.0)
        if total == 0:
            return None
        return treated / total
    return _number(result[metric])


def series_from_records(
    records: list[dict[str, object]], metric: str
) -> dict[tuple[str, str, str], list[SeriesPoint]]:
    """Per field, killer and load, the points of ``metric`` against the explorer count."""
    values: dict[ArmKey, list[float]] = {}
    for record in records:
        key = ArmKey.from_arm_name(str(record["arm"]))
        result = record["result"]
        if not isinstance(result, dict):
            continue
        value = _record_value(result, metric)
        values.setdefault(key, [])
        if value is not None:
            values[key].append(value)
    series: dict[tuple[str, str, str], list[SeriesPoint]] = {}
    for key in sorted(values, key=lambda item: item.explorers):
        found = values[key]
        if not found:
            continue
        error = 0.0
        if len(found) > 1:
            error = stdev(found) / math.sqrt(len(found))
        point = SeriesPoint(key.explorers, mean(found), error)
        series.setdefault((key.field, key.killer, key.load), []).append(point)
    return series


def plot_results(directory: Path) -> list[Path]:
    """Draw one PDF per field from ``results.json``, and a LaTeX figure fragment for each.

    Each figure has a panel per measure of ``METRICS`` against the explorer count, on a log
    axis, with a line per killer (blue hop, orange seeded) and load (dashed low, solid high),
    and error bars of one standard error over the seeds.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = json.loads((directory / "results.json").read_text())
    description = json.loads((directory / "experiment.json").read_text())
    fields = sorted({ArmKey.from_arm_name(str(record["arm"])).field for record in records})
    seeds = len({record["seed"] for record in records})
    written: list[Path] = []
    for field_name in fields:
        figure, axes = plt.subplots(2, 3, figsize=(11, 6.2), constrained_layout=True)
        for axis, (metric, label) in zip(axes.flat, METRICS.items(), strict=True):
            counts: set[int] = set()
            for (series_field, killer, load), points in series_from_records(
                records, metric
            ).items():
                if series_field != field_name:
                    continue
                counts.update(point.explorers for point in points)
                axis.errorbar(
                    [point.explorers for point in points],
                    [point.mean for point in points],
                    yerr=[point.error for point in points],
                    color=COLORS[killer],
                    marker=MARKERS[load],
                    linestyle=LINESTYLES[load],
                    capsize=3,
                    label=f"{killer}, {load} infestation",
                )
            axis.set_xscale("log", base=2)
            ticks = sorted(counts)
            axis.set_xticks(ticks, [str(count) for count in ticks])
            axis.minorticks_off()
            axis.set_xlabel("Explorers")
            axis.set_ylabel(label)
            axis.grid(alpha=0.3)
        axes.flat[0].legend(fontsize=8)
        pdf = directory / f"num_explorers_{field_name}.pdf"
        figure.savefig(pdf)
        plt.close(figure)
        written.append(pdf)
        caption = _caption(field_name, _radio_from_command(description.get("command")), seeds)
        fragment = (
            "\\begin{figure}[t]\n  \\centering\n"
            f"  \\includegraphics[width=\\textwidth]{{{pdf.name}}}\n"
            f"  \\caption{{{caption}}}\n"
            f"  \\label{{fig:num-explorers-{field_name}}}\n"
            "\\end{figure}\n"
        )
        tex = directory / f"num_explorers_{field_name}.tex"
        tex.write_text(fragment)
        written.append(tex)
    logger.info("figures of %s written from commit %s", directory, description.get("commit"))
    return written


def _radio_from_command(command: object) -> float:
    """The ``--radio`` of a recorded command line, or the default if it was not given."""
    if isinstance(command, list) and "--radio" in command:
        return float(command[command.index("--radio") + 1])
    return RADIO_RANGE_M


def _caption(field_name: str, radio_range_m: float, seeds: int) -> str:
    """A figure caption with the configuration the plotting guidelines ask for."""
    setup = FIELDS[field_name]
    killers = setup.killers
    robots = setup.scenario.robots
    rates: list[str] = []
    for load, per_killer in LOADS.items():
        rates.append(f"{load} {per_killer * killers:g}")
    map_text = "32$\\times$32 cells, one river and rocks"
    if field_name == "rivers-large":
        map_text = "64$\\times$64 cells, two rivers and rocks"
    return (
        f"Number of explorers against the pest response on the {field_name} field ({map_text}, "
        f"10\\,m cells). {killers} killers, hop-based (\\texttt{{hop-best}}, blue) or seeded "
        f"(\\texttt{{seeded-best}}, orange); explorers at {robots.speed_mps_drone:g}\\,m/s, "
        f"killers at {robots.speed_mps_ground:g}\\,m/s; "
        f"{radio_range_m:g}\\,m radio; no relays or mock sources; static pests at "
        f"{' and '.join(rates)} pests per hour (dashed and solid). "
        f"Mean $\\pm$ one standard error over {seeds} seeds."
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    if args.replot is not None:
        for path in plot_results(args.replot):
            print(path)
        return
    scenarios = build_scenarios(args)
    experiment = Experiment("num_explorers")
    runs = runs_for(scenarios, range(args.seeds), SwarmScenario.run_cost)
    results = by_arm(experiment.run(runs, measure_pest_response, args.jobs))
    print(write_tables(experiment, scenarios, results))
    for path in plot_results(experiment.directory):
        print(path)


if __name__ == "__main__":
    main()
