"""Study the base demo: five scouts that only explore, five killers that only treat (#67).

Question: does the simplest swarm that does the whole job clear the pests its scouts find, and
which part of the pipeline limits it? The base run is ``PestScenario.smoke``; each comparison
changes one part of it:

- ``scouts-only``: no killers, the upper bound on infested cells;
- ``random-waypoint``: scouts drive to random cells instead of following the EXPLORE field;
- ``radio-50``, ``radio-150``, ``radio-unlimited``: where news of a pest stops reaching the
  killers (the base run has 300 m);
- ``killers-0.1``: killers at 0.1 m/s instead of 0.5, how much of the response is driving;
- ``spread``: pests spread to their neighbours, whether five killers keep up.

Every condition runs with the same seeds. Every run writes its metrics series, and the study one
summary row per run and the wall-clock time per run, to ``--out`` (``runs/smoke`` by default).
The tables printed at the end, in Markdown, are the means over the seeds. Runs are spread over
processes. Run from the repository root with:

    uv run python -m studies.smoke --seeds 5 --hours 3 --jobs 16
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean

from tests.baselines import RANDOM_WAYPOINT_BRAIN

from swarm_field.contracts import TimeUnit
from swarm_field.experiments import PestScenario
from swarm_field.runner import (
    ExperimentConfig,
    RunSummary,
    export_summary_csv,
    export_summary_json,
    run_experiment,
)

SPREAD_PROBABILITY = 0.05
"""The spread variant's chance per minute that a pest of severity 1 infests one neighbour."""


def _base(seed: int, hours: float) -> PestScenario:
    return replace(PestScenario.smoke(seed), until=int(hours * TimeUnit.HOUR))


def _smoke(seed: int, hours: float) -> ExperimentConfig:
    return _base(seed, hours).into_experiment()


def _scouts_only(seed: int, hours: float) -> ExperimentConfig:
    return replace(_base(seed, hours), killers=0).into_experiment()


def _random_waypoint(seed: int, hours: float) -> ExperimentConfig:
    # The scouts have the shared brain; the killers have one of their own.
    return replace(_base(seed, hours).into_experiment(), brain=RANDOM_WAYPOINT_BRAIN)


def _radio(radio_range_m: float) -> Callable[[int, float], ExperimentConfig]:
    def build(seed: int, hours: float) -> ExperimentConfig:
        return replace(_base(seed, hours), radio_range_m=radio_range_m).into_experiment()

    return build


def _slow_killers(seed: int, hours: float) -> ExperimentConfig:
    return replace(_base(seed, hours), speed_mps_ground=0.1).into_experiment()


def _spread(seed: int, hours: float) -> ExperimentConfig:
    return replace(_base(seed, hours), spread_probability=SPREAD_PROBABILITY).into_experiment()


CONDITIONS: dict[str, Callable[[int, float], ExperimentConfig]] = {
    "smoke": _smoke,
    "scouts-only": _scouts_only,
    "random-waypoint": _random_waypoint,
    "radio-50": _radio(50.0),
    "radio-150": _radio(150.0),
    "radio-unlimited": _radio(float("inf")),
    "killers-0.1": _slow_killers,
    "spread": _spread,
}
"""Each condition by name, building its run from a seed and a horizon in hours."""


@dataclass(frozen=True)
class Job:
    """One run of the study: a condition and a seed.

    Attributes:
        condition: A name of ``CONDITIONS``.
        seed: The seed of the run.
        hours: The simulated time of the run.
    """

    condition: str
    seed: int
    hours: float


@dataclass(frozen=True)
class Result:
    """What one run produced.

    Attributes:
        job: The run.
        summary: Its summary, labelled ``CONDITION/SEED``.
        wall_seconds: The computation time of the run.
    """

    job: Job
    summary: RunSummary
    wall_seconds: float


def run(job: Job) -> Result:
    """Build and run one job, timing it."""
    config = CONDITIONS[job.condition](job.seed, job.hours)
    config = replace(config, label=f"{job.condition}/{job.seed}", scenario_name="smoke")
    started = time.perf_counter()
    summary = run_experiment(config)
    return Result(job, summary, time.perf_counter() - started)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS),
        default=list(CONDITIONS),
        help="conditions to run (default: all)",
    )
    parser.add_argument("--seeds", type=int, default=5, help="runs per condition, seeds 0, 1, ...")
    parser.add_argument("--hours", type=float, default=3.0, help="simulated time per run")
    parser.add_argument("--jobs", type=int, default=8, help="processes")
    parser.add_argument("--out", type=Path, default=Path("runs/smoke"), help="where results go")
    return parser.parse_args()


def _minutes(values: Sequence[float | None]) -> str:
    """The mean of the values present, in minutes, or ``-`` if none is."""
    present: list[float] = []
    for value in values:
        if value is not None:
            present.append(value)
    if not present:
        return "-"
    return f"{mean(present) / TimeUnit.MINUTE:.1f}"


def _seconds(values: Sequence[float | None]) -> str:
    """The mean of the values present, in seconds, or ``-`` if none is."""
    present: list[float] = []
    for value in values:
        if value is not None:
            present.append(value)
    if not present:
        return "-"
    return f"{mean(present) / TimeUnit.SECOND:.0f}"


def _killer_columns(runs: Sequence[RunSummary]) -> str:
    """The killers' time split and duplicate trips, or dashes for a run without killers."""
    killers = 0
    for run in runs:
        killers += run.metadata.bodies.count("KILLER")
    if killers == 0:
        return "- | -"
    return (
        f"{mean(r.killer_driving_share for r in runs):.0%} / "
        f"{mean(r.killer_treating_share for r in runs):.0%} / "
        f"{mean(r.killer_idle_share for r in runs):.0%} "
        f"| {mean(r.duplicate_trip_share for r in runs):.0%}"
    )


def print_tables(conditions: list[str], results: list[Result], hours: float) -> None:
    """Print the pest response, the monitoring and the cost, one row per condition."""
    groups: dict[str, list[RunSummary]] = {}
    walls: dict[str, list[float]] = {}
    for result in results:
        groups.setdefault(result.job.condition, []).append(result.summary)
        walls.setdefault(result.job.condition, []).append(result.wall_seconds)

    print("\n### Pest response\n")
    print(
        "| Condition | Infested (mean / peak) | Cleared / appeared | Detection (min) "
        "| News to killer (s) | Response (min) | Killers driving / treating / idle "
        "| Same pest |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for condition in conditions:
        runs = groups[condition]
        print(
            f"| {condition} "
            f"| {mean(r.mean_infested_cells for r in runs):.1f} / "
            f"{mean(r.peak_infested_cells for r in runs):.1f} "
            f"| {mean(r.pests_cleared for r in runs):.1f} / "
            f"{mean(r.pests_appeared for r in runs):.1f} "
            f"| {_minutes([r.mean_detection_latency_ms for r in runs])} "
            f"| {_seconds([r.knowledge_latency_ms for r in runs])} "
            f"| {_minutes([r.response_time_ms for r in runs])} "
            f"| {_killer_columns(runs)} |"
        )

    print("\n### Monitoring and cost\n")
    print(
        "| Condition | Coverage | Idleness (min, mean / worst) | Redundancy "
        "| Messages received | Wall-clock (s per simulated hour) |"
    )
    print("| --- | --- | --- | --- | --- | --- |")
    for condition in conditions:
        runs = groups[condition]
        idleness = [r.series[-1].mean_idleness_ms for r in runs]
        worst = [r.series[-1].worst_idleness_ms for r in runs]
        print(
            f"| {condition} "
            f"| {mean(r.coverage for r in runs):.0%} "
            f"| {_minutes(idleness)} / {_minutes(worst)} "
            f"| {mean(r.series[-1].redundancy for r in runs):.0%} "
            f"| {mean(r.messages_received for r in runs):,.0f} "
            f"| {mean(walls[condition]) / hours:.0f} |"
        )


def write_results(results: list[Result], out: Path) -> None:
    """Write the summaries, every run's series and the wall-clock times under ``out``."""
    summaries = [result.summary for result in results]
    export_summary_csv(summaries, out / "summary.csv")
    export_summary_json(summaries, out / "summary.json")
    for summary in summaries:
        summary.export(out)
    with (out / "timing.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "wall_seconds"])
        for result in results:
            writer.writerow([result.summary.label, f"{result.wall_seconds:.2f}"])


def main() -> None:
    args = parse_args()
    jobs: list[Job] = []
    for condition in args.conditions:
        for seed in range(args.seeds):
            jobs.append(Job(condition, seed, args.hours))
    print(f"{len(jobs)} runs of {args.hours:g} h on {args.jobs} processes")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, jobs))
    write_results(results, args.out)
    print_tables(args.conditions, results, args.hours)
    print(f"\nWrote {len(results)} runs to {args.out}")


if __name__ == "__main__":
    main()
