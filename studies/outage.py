"""OUTAGE: how the seeded and the hop-based killers degrade as robots fail during a run.

Question: when robots fail one by one during a run, until about 40% of the swarm is gone by the
end, how much worse do the seeded killers and the hop-based killers respond to pests, compared
with the same run without failures? And how does that depend on how many robots are left?

Setup: the default settings of ``studies/defaults.py`` (10 explorers and 4 killers, static pests
at the start and every hour) on the small field with a river and rocks. Six arms: the seeded
and the hop-based killers (``killer_methods``, each with its best parameters), each without
failures and with each of two kinds of failure (``FAILURE_KINDS``):

- ``dark``: the robot stops and goes dark (``FailureMode.DEAD_BATTERY``), as if it were removed;
- ``immobile``: the robot stops moving and acting, but still senses its own cell, decides and
  communicates (``FailureMode.IMMOBILE``), so it stays in the network as a relay fixed in place.
  It does not know it is stuck: an immobile killer still commits to pests it cannot reach.

A failure is checked for every working robot, explorer or killer, once a minute, with the
probability that leaves ``--lost`` of the swarm failed by the end on average. The failures are
drawn from the run's fault-injection randomness, apart from the pests, so with the same seed every
arm with failures loses the same robots at the same times, whatever the method or the kind of
failure, and every arm sees the same pests.

Measured per run: the standard ``measure_pest_response``, and a timeline every
``TIMELINE_PERIOD``: the robots and the killers still working (an immobile robot no longer
counts), the infested cells (the mean over the period) and the pests treated so far. These are
custom measures, for the figure of the response over time and against the robots left.

Hypotheses, before running:

1. Both methods get worse as robots fail: fewer killers treat the pests, and fewer explorers
   find them, so the infested cells rise over the run with failures, while they stay level
   without.
2. Losing a killer matters more than losing an explorer: with 4 killers each one is a quarter of
   the treating, while 10 explorers cover the small field many times over.
3. The hop-based killers degrade more than the seeded killers: their beacons travel only along
   chains of live robots, and every failure breaks links in those chains, while a seeded killer
   keeps the pests it already knows of in its map.
4. The difference between the methods grows with the robots lost, so it is largest at the end of
   the run.
5. Immobile robots hurt the hop-based killers less than dark ones: the relay chains the beacons
   travel along keep their links, so only the lost killers and the lost sensing count.
6. For the seeded killers, immobile robots hurt about as much as dark ones, or more: a seeded
   killer learns of the pests through the shared map either way, and an immobile killer may hold
   on to a pest near it that it can never reach, since a farther killer leaves that pest to it.

The runs go through ``Experiment``, which writes every run's result, timeline included, to a
new folder in ``output/``, with the standard table, a PDF figure and its LaTeX fragment. Run from
the repository root with:

    uv run python -m studies.outage --seeds 5 --jobs 6

and for a quick check that it works:

    uv run python -m studies.outage --seeds 1 --hours 1 --warmup-hours 0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
from statistics import mean
from typing import TYPE_CHECKING

from studies.defaults import (
    DEFAULT_ROBOTS,
    EXPLORERS,
    KILLERS,
    MAPS,
    default_scenario,
    killer_methods,
    pest_run_spec,
)
from studies.plotting import (
    METHOD_COLORS,
    METHOD_LABELS,
    mean_and_standard_error,
    plot_method,
    save_figure,
    save_table,
)
from swarm_field.contracts import BodyType, FailureMode, FailureSpec, Time, TimeUnit
from swarm_field.experiments import (
    Experiment,
    MeasuredPestRun,
    PestResponse,
    PestRunSpec,
    Run,
    arm_name,
    by_arm,
    markdown_table,
    pest_table_cells,
    pest_table_headers,
    runs_for,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

METHODS = ("seeded", "hop")
DEFAULT_MAP = "rivers-small"
DEFAULT_LOST = 0.4
"""The share of the swarm failed by the end of a run, on average."""

FAILURE_KINDS: dict[str, FailureMode] = {
    "dark": FailureMode.DEAD_BATTERY,
    "immobile": FailureMode.IMMOBILE,
}
"""The kinds of failure compared, by the name in the arm names."""

NO_FAILURES = "none"
LINESTYLES: dict[str, str] = {NO_FAILURES: "--", "dark": "-", "immobile": ":"}

FAILURE_PERIOD: Time = TimeUnit.MINUTE
TIMELINE_PERIOD: Time = 10 * TimeUnit.MINUTE


@dataclass(frozen=True)
class TimelinePoint:
    """The state of a run at the end of one timeline period.

    Attributes:
        minutes: When, in minutes since the start.
        live_robots: The robots that had not failed; an immobile robot has failed.
        live_killers: The killers among them.
        infested: The infested cells, averaged over the period's steps.
        treated: The pests treated since the start.
    """

    minutes: float
    live_robots: int
    live_killers: int
    infested: float
    treated: int


@dataclass(frozen=True)
class OutageResult:
    """What one run measured: the standard measures and the timeline.

    Attributes:
        response: The standard measures of the whole run.
        timeline: The run's state every ``TIMELINE_PERIOD``.
    """

    response: PestResponse
    timeline: tuple[TimelinePoint, ...]


def failure_probability(lost: float, hours: int) -> float:
    """The chance per check that a robot fails, so that ``lost`` of them fail in ``hours``.

    Every live robot fails independently on every check with the same chance ``p``, so after
    ``n`` checks a robot has survived with the chance ``(1 - p) ** n``.
    """
    checks = hours * TimeUnit.HOUR / FAILURE_PERIOD
    return 1.0 - (1.0 - lost) ** (1.0 / checks)


def measure_outage(run: Run[PestRunSpec]) -> OutageResult:
    """Make one run, measured as ``measure_pest_response`` does, with a timeline of its state."""
    measured = MeasuredPestRun(run)
    world = measured.live.context.world
    timeline: list[TimelinePoint] = []
    infested: list[int] = []
    while not measured.finished:
        measured.advance()
        infested.append(len(measured.tracker.appeared))
        now = measured.live.now
        if now % TIMELINE_PERIOD != 0:
            continue
        live = world.live_robots()
        live_killers = 0
        for robot_id in live:
            if world.robots[robot_id].body is BodyType.KILLER:
                live_killers += 1
        timeline.append(
            TimelinePoint(
                minutes=now / TimeUnit.MINUTE,
                live_robots=len(live),
                live_killers=live_killers,
                infested=mean(infested),
                treated=len(measured.tracker.responses),
            )
        )
        infested = []
    return OutageResult(measured.response(), tuple(timeline))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", choices=list(MAPS), default=DEFAULT_MAP)
    parser.add_argument(
        "--lost", type=float, default=DEFAULT_LOST, help="share of the swarm failed by the end"
    )
    parser.add_argument(
        "--kinds", nargs="+", choices=list(FAILURE_KINDS), default=list(FAILURE_KINDS)
    )
    parser.add_argument("--seeds", type=int, default=5, help="runs per arm, seeds 0, 1, ...")
    parser.add_argument("--hours", type=int, help="simulated hours per run; the map's by default")
    parser.add_argument("--warmup-hours", type=int, default=1, help="left out of infested")
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hours = args.hours or MAPS[args.map].hours
    probability = failure_probability(args.lost, hours)
    specs: dict[str, PestRunSpec] = {}
    for method, scenario in killer_methods(default_scenario(args.map)).items():
        spec = replace(pest_run_spec(scenario, args.map, args.warmup_hours), hours=hours)
        specs[arm_name(method, failures=NO_FAILURES)] = spec
        for kind in args.kinds:
            failures = FailureSpec(FAILURE_PERIOD, probability, FAILURE_KINDS[kind])
            specs[arm_name(method, failures=kind)] = replace(spec, failures=failures)

    experiment = Experiment("outage")
    runs = runs_for(specs, range(args.seeds), PestRunSpec.cost)
    results = by_arm(experiment.run(runs, measure_outage, args.jobs))

    headers = pest_table_headers(hours)
    rows: list[list[str]] = []
    for arm, arm_results in results.items():
        rows.append(pest_table_cells(arm, [result.response for result in arm_results], hours))
    table = markdown_table(headers, rows)
    print(table)
    experiment.write_text("tables.md", table)
    configuration = _configuration(args, hours, probability)
    save_table(
        experiment,
        "outage_table",
        headers,
        rows,
        "The standard measures of the seeded and hop-based killers without robot failures, and "
        "with failed robots that go dark or that stay in place and keep communicating "
        f"(immobile); mean ± standard error over {args.seeds} seeds. {configuration}",
        "outage",
    )
    plot_outage(experiment, args, results, configuration)


def plot_outage(
    experiment: Experiment,
    args: argparse.Namespace,
    results: dict[str, list[OutageResult]],
    configuration: str,
) -> None:
    """Infested cells and live robots against time, and infested cells against robots left."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (time_axes, robots_axes, left_axes) = plt.subplots(1, 3, figsize=(13.0, 3.8))
    for method in METHODS:
        for kind in [NO_FAILURES, *args.kinds]:
            arm_results = results[arm_name(method, failures=kind)]
            minutes = [point.minutes for point in arm_results[0].timeline]
            infested: list[list[float]] = []
            for index in range(len(minutes)):
                infested.append([result.timeline[index].infested for result in arm_results])
            label = f"{METHOD_LABELS[method]}, {kind}"
            if kind == NO_FAILURES:
                label = f"{METHOD_LABELS[method]}, no failures"
            plot_method(time_axes, method, minutes, infested, LINESTYLES[kind], label)

        for kind in args.kinds:
            by_robots_left: dict[int, list[float]] = defaultdict(list)
            for result in results[arm_name(method, failures=kind)]:
                for point in result.timeline:
                    by_robots_left[point.live_robots].append(point.infested)
            left = sorted(by_robots_left)
            left_axes.plot(
                left,
                [mean(by_robots_left[count]) for count in left],
                color=METHOD_COLORS[method],
                linestyle=LINESTYLES[kind],
                marker="o",
                label=f"{METHOD_LABELS[method]}, {kind}",
            )
    plot_robots_left(robots_axes, results[arm_name(METHODS[0], failures=args.kinds[0])])
    time_axes.set_xlabel("minutes")
    time_axes.set_ylabel("infested cells")
    time_axes.legend(fontsize="small")
    robots_axes.set_xlabel("minutes")
    robots_axes.set_ylabel("robots left")
    robots_axes.legend(fontsize="small")
    left_axes.set_xlabel("robots left")
    left_axes.set_ylabel("infested cells")
    left_axes.invert_xaxis()
    left_axes.legend(fontsize="small")
    for axes in (time_axes, robots_axes, left_axes):
        axes.set_ylim(bottom=0)
        axes.grid(alpha=0.3)
    save_figure(
        experiment,
        figure,
        "outage",
        "Robot failures: the infested cells over time without failures (dashed), with failed "
        "robots that go dark (solid) and with failed robots that stay in place and keep "
        "communicating (immobile, dotted), left; the robots and the killers still working over "
        "time, the same for every arm with failures with the same seeds, middle; and the "
        "infested cells against the robots still working, over every ten minutes of every run "
        "with failures, right. Mean ± standard error over "
        f"{args.seeds} seeds, the infested cells averaged over each ten minutes. {configuration}",
        "outage",
    )


def plot_robots_left(axes: Axes, results: list[OutageResult]) -> None:
    """The robots and the killers left over time, the means over the seeds of one arm."""
    minutes = [point.minutes for point in results[0].timeline]
    for name, color in (("robots", "black"), ("killers", "#7f7f7f")):
        means: list[float] = []
        errors: list[float] = []
        for index in range(len(minutes)):
            left: list[float] = []
            for result in results:
                point = result.timeline[index]
                if name == "robots":
                    left.append(float(point.live_robots))
                else:
                    left.append(float(point.live_killers))
            average, error = mean_and_standard_error(left)
            means.append(average)
            errors.append(error)
        axes.errorbar(minutes, means, yerr=errors, color=color, marker=".", capsize=3, label=name)


def _configuration(args: argparse.Namespace, hours: int, probability: float) -> str:
    """The swarm, the pests and the failures of a run, for a caption."""
    standard = MAPS[args.map]
    return (
        f"Field {standard.label}, {hours} h; {EXPLORERS} explorers at "
        f"{DEFAULT_ROBOTS.speed_mps_drone:g} m/s and {KILLERS} killers at "
        f"{DEFAULT_ROBOTS.speed_mps_ground:g} m/s, radio {DEFAULT_ROBOTS.radio_range_m:g} m; "
        f"{standard.pests_per_hour:g} static pests at the start and per hour. Every robot fails "
        f"with chance {probability:.4f} per minute, {args.lost:.0%} of the swarm by "
        "the end on average."
    )


if __name__ == "__main__":
    main()
