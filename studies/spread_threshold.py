"""Study from which pest spread rate a swarm is needed, against a grid-search baseline (#74).

Question: up to which spread rate do utility robots that each sweep a strip of the field keep the
pests under control, and does the swarm hold out longer? Three systems, on the pest experiment's
field (``PestScenario``):

- ``grid-10``: ten utility robots at 0.5 m/s that sweep their own strip and treat what they drive
  over, sharing nothing (``tests/baselines.py``);
- ``swarm-5+5``: the base demo of #67 (``PestScenario.smoke``): five scouts, five killers;
- ``swarm-10+10``: the same swarm with twice the robots of each kind.

Every run starts with ``INITIAL_PESTS`` pests of severity ``INITIAL_SEVERITY`` on random cells
off the bottom row, drawn from the seed, so all systems face the same pests for the same seed. No
pest appears after that: the only way pests multiply is by spreading, so the spread rate is the
only variable.

Each system is run at every spread rate with the same seeds. A run is out of control when its
infested cells still grow at the end: their mean over the last hour is more than ``GROWTH_FACTOR``
times, and more than ``GROWTH_FLOOR`` cells above, their mean over the hour before. Hourly means
rather than single samples keep the ordinary ups and downs of a steady run from counting. A
system's threshold is the lowest spread rate at which most of its seeds are out of control.

Every run writes its metrics series, and the study one summary row per run, to ``--out``
(``runs/spread_threshold`` by default), with a plot of the infested cells against the spread rate.
Runs are spread over processes. Run from the repository root with:

    uv run python -m studies.spread_threshold --seeds 5 --hours 3 --jobs 8
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from statistics import mean, stdev

from tests.baselines import grid_sweep_brain

from swarm_field.contracts import BodyType, TimeUnit
from swarm_field.experiments import PestScenario
from swarm_field.runner import (
    ExperimentConfig,
    RunSummary,
    export_summary_csv,
    export_summary_json,
    run_experiment,
)

DEFAULT_SPREADS = (0.0, 0.005, 0.01, 0.02, 0.05)
INITIAL_PESTS = 10
INITIAL_SEVERITY = 0.1
"""The severity of a pest that has just appeared, as ``PestScenario.spawn_severity``."""
GROWTH_FACTOR = 1.5
GROWTH_FLOOR = 10


def _grid_sweep(robots: int) -> Callable[[int, float, float], ExperimentConfig]:
    def build(seed: int, hours: float, spread: float) -> ExperimentConfig:
        smoke = PestScenario.smoke(seed)
        scenario = replace(
            smoke,
            scouts=0,
            killers=robots,
            radio_range_m=0.0,
            pests_per_hour=0.0,
            spread_probability=spread,
            until=int(hours * TimeUnit.HOUR),
        )
        config = scenario.into_experiment()
        return replace(config, brains_by_body={BodyType.KILLER: grid_sweep_brain(robots)})

    return build


def _swarm(scouts: int, killers: int) -> Callable[[int, float, float], ExperimentConfig]:
    def build(seed: int, hours: float, spread: float) -> ExperimentConfig:
        scenario = replace(
            PestScenario.smoke(seed),
            scouts=scouts,
            killers=killers,
            pests_per_hour=0.0,
            spread_probability=spread,
            until=int(hours * TimeUnit.HOUR),
        )
        return scenario.into_experiment()

    return build


SYSTEMS: dict[str, Callable[[int, float, float], ExperimentConfig]] = {
    "grid-10": _grid_sweep(10),
    "swarm-5+5": _swarm(5, 5),
    "swarm-10+10": _swarm(10, 10),
}
"""Each system by name, building its run from a seed, a horizon in hours and a spread rate."""


@dataclass(frozen=True)
class Job:
    """One run of the study.

    Attributes:
        system: A name of ``SYSTEMS``.
        spread: The pests' spread probability.
        seed: The seed of the run.
        hours: The simulated time of the run.
    """

    system: str
    spread: float
    seed: int
    hours: float


def with_initial_pests(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    """``config`` with ``INITIAL_PESTS`` pests at the start, on random cells off the bottom row.

    The cells depend only on the seed and the field, not on the robots, so every system faces the
    same pests for the same seed; the bottom row is left free because the robots start there.
    """
    free: list[tuple[int, int]] = []
    for cell in sorted(config.scenario.crops):
        if cell[1] > 0:
            free.append(cell)
    infestation: dict[tuple[int, int], float] = {}
    for cell in sorted(Random(seed).sample(free, INITIAL_PESTS)):
        infestation[cell] = INITIAL_SEVERITY
    return replace(config, scenario=replace(config.scenario, initial_infestation=infestation))


def run(job: Job) -> RunSummary:
    """Build and run one job, labelled ``SYSTEM/spread-SPREAD/SEED``."""
    config = with_initial_pests(SYSTEMS[job.system](job.seed, job.hours, job.spread), job.seed)
    label = f"{job.system}/spread-{job.spread:g}/{job.seed}"
    return run_experiment(replace(config, label=label, scenario_name="spread-threshold"))


def is_out_of_control(summary: RunSummary) -> bool:
    """Whether the run's infested cells still grew over its last hour, by their hourly means.

    A run shorter than two hours compares the second half of its samples with the first.
    """
    end = summary.series[-1].time
    last: list[int] = []
    before: list[int] = []
    for sample in summary.series:
        if sample.time > end - TimeUnit.HOUR:
            last.append(sample.targets_remaining)
        elif sample.time > end - 2 * TimeUnit.HOUR:
            before.append(sample.targets_remaining)
    if not before:
        half = len(summary.series) // 2
        before = [sample.targets_remaining for sample in summary.series[:half]]
        last = [sample.targets_remaining for sample in summary.series[half:]]
    if not before or not last:
        return False
    growth = mean(last) - mean(before)
    return mean(last) > GROWTH_FACTOR * mean(before) and growth > GROWTH_FLOOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--systems", nargs="+", choices=list(SYSTEMS), default=list(SYSTEMS), help="default: all"
    )
    parser.add_argument(
        "--spreads", type=float, nargs="+", default=list(DEFAULT_SPREADS), help="spread rates"
    )
    parser.add_argument("--seeds", type=int, default=5, help="runs per point, seeds 0, 1, ...")
    parser.add_argument("--hours", type=float, default=3.0, help="simulated time per run")
    parser.add_argument("--jobs", type=int, default=8, help="processes")
    parser.add_argument(
        "--out", type=Path, default=Path("runs/spread_threshold"), help="where results go"
    )
    return parser.parse_args()


@dataclass(frozen=True)
class Point:
    """One system at one spread rate, over its seeds.

    Attributes:
        mean_infested: The mean infested cells, averaged over the seeds.
        spread_infested: Their standard deviation over the seeds; 0 for one seed.
        out_of_control: How many seeds ended out of control.
        seeds: How many seeds ran.
        clean: How many seeds cleared every pest before the end.
        clean_after: When those seeds' fields were clean, averaged over them, or ``None``.
    """

    mean_infested: float
    spread_infested: float
    out_of_control: int
    seeds: int
    clean: int
    clean_after: float | None

    @classmethod
    def from_runs(cls, runs: list[RunSummary]) -> Point:
        infested = [run.mean_infested_cells for run in runs]
        deviation = 0.0
        if len(infested) > 1:
            deviation = stdev(infested)
        lost = 0
        clean_times: list[int] = []
        for summary in runs:
            if is_out_of_control(summary):
                lost += 1
            for sample in summary.series:
                if sample.targets_remaining == 0:
                    clean_times.append(sample.time)
                    break
        clean_after = None
        if clean_times:
            clean_after = mean(clean_times)
        return cls(mean(infested), deviation, lost, len(runs), len(clean_times), clean_after)

    def is_lost(self) -> bool:
        """Whether most seeds ended out of control."""
        return 2 * self.out_of_control > self.seeds


def collect_points(jobs: list[Job], summaries: list[RunSummary]) -> dict[tuple[str, float], Point]:
    """Every system and spread rate's point, from the runs."""
    grouped: dict[tuple[str, float], list[RunSummary]] = {}
    for job, summary in zip(jobs, summaries, strict=True):
        grouped.setdefault((job.system, job.spread), []).append(summary)
    points: dict[tuple[str, float], Point] = {}
    for key, runs in grouped.items():
        points[key] = Point.from_runs(runs)
    return points


def print_table(
    systems: list[str], spreads: list[float], points: dict[tuple[str, float], Point]
) -> None:
    """Mean infested cells per system and spread rate, with the seeds out of control."""
    print("\n### Mean infested cells (standard deviation over the seeds; seeds out of control)\n")
    print("| Spread | " + " | ".join(systems) + " |")
    print("| --- |" + " --- |" * len(systems))
    for spread in spreads:
        cells: list[str] = []
        for system in systems:
            point = points[(system, spread)]
            cells.append(
                f"{point.mean_infested:.1f} (sd {point.spread_infested:.1f}; "
                f"{point.out_of_control}/{point.seeds} lost)"
            )
        print(f"| {spread:g} | " + " | ".join(cells) + " |")
    print("\n### Field clean: seeds that cleared every pest, and after how many minutes\n")
    print("| Spread | " + " | ".join(systems) + " |")
    print("| --- |" + " --- |" * len(systems))
    for spread in spreads:
        cells = []
        for system in systems:
            point = points[(system, spread)]
            after = "-"
            if point.clean_after is not None:
                after = f"{point.clean_after / TimeUnit.MINUTE:.0f} min"
            cells.append(f"{point.clean}/{point.seeds}, {after}")
        print(f"| {spread:g} | " + " | ".join(cells) + " |")
    print("\n### Threshold: the lowest spread rate at which most seeds are out of control\n")
    for system in systems:
        threshold = "none in the range tested"
        for spread in spreads:
            if points[(system, spread)].is_lost():
                threshold = f"{spread:g}"
                break
        print(f"- {system}: {threshold}")


def plot_infested(
    systems: list[str], spreads: list[float], points: dict[tuple[str, float], Point], path: Path
) -> None:
    """Plot the mean infested cells against the spread rate, one line per system."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    for system in systems:
        means: list[float] = []
        deviations: list[float] = []
        for spread in spreads:
            point = points[(system, spread)]
            means.append(point.mean_infested)
            deviations.append(point.spread_infested)
        axes.errorbar(spreads, means, yerr=deviations, marker="o", capsize=3, label=system)
    axes.set_xlabel("pest spread probability per minute")
    axes.set_ylabel("mean infested cells")
    axes.set_yscale("log")
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    jobs: list[Job] = []
    for system in args.systems:
        for spread in args.spreads:
            for seed in range(args.seeds):
                jobs.append(Job(system, spread, seed, args.hours))
    print(f"{len(jobs)} runs of {args.hours:g} h on {args.jobs} processes")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        summaries = list(pool.map(run, jobs))

    export_summary_csv(summaries, args.out / "summary.csv")
    export_summary_json(summaries, args.out / "summary.json")
    for summary in summaries:
        summary.export(args.out)
    points = collect_points(jobs, summaries)
    print_table(args.systems, args.spreads, points)
    plot_infested(args.systems, args.spreads, points, args.out / "infested.png")
    print(f"\nWrote {len(summaries)} runs and infested.png to {args.out}")


if __name__ == "__main__":
    main()
