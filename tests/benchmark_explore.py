"""Benchmark the EXPLORE seeder: variants and parameters of a swarm exploring one field.

Each run is a ``swarm_field.experiments`` exploration on the real simulation loop, repeated with
several seeds (which stagger the robots' decisions). A row reports the ground-truth coverage
measures of ``CoverageTracker``, as the mean over the seeds:

- ``t90`` and ``full``: when 90 % and all of the field had been visited, in minutes; ``-`` if
  some seed never got there;
- ``idle`` and ``worst``: mean and worst idleness over the second half of the run, in minutes;
  the mean is the expected detection latency of a pest appearing at a random time and cell;
- ``redund``: the share of visits to a cell visited less than 30 minutes before;
- ``wall``: seconds of computation per run.

Runs are spread over processes. Not part of ``pytest``; run it with:

    uv run python -m tests.benchmark_explore --size 32 --hours 3 --seeds 4 --sections values
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from statistics import mean

from swarm_field.contracts import CellId, RobotId, Time, TimeUnit
from swarm_field.experiments import (
    ExploreScenario,
    RandomWaypointSeeder,
    build_explore_run,
    river_and_rocks,
)
from swarm_field.simulation import CoverageSummary
from swarm_field.swarm import ExploreConfig, ExploreSeeder, FieldSeeder
from swarm_field.swarm.explore import Growth

_REFERENCE = ExploreConfig()
"""The variant the others are compared with: the defaults, 8 x 8 blocks with the discount."""


@dataclass(frozen=True)
class Variant:
    """One configuration of the swarm.

    Attributes:
        name: The row's label.
        explore: The EXPLORE parameters, or ``None`` for the random-waypoint baseline.
        share_observations: Whether sensed facts reach every robot (``ExploreScenario``).
        obstacles: Whether the field has the river and rocks of ``river_and_rocks``.
    """

    name: str
    explore: ExploreConfig | None
    share_observations: bool = True
    obstacles: bool = False


@dataclass(frozen=True)
class Job:
    """One run to make: a variant on a field of ``size`` cells for ``hours``, with ``seed``."""

    variant: Variant
    size: int
    robots: int
    hours: float
    seed: int


@dataclass(frozen=True)
class Result:
    """The measures of one run, and the seconds it took."""

    summary: CoverageSummary
    wall: float


def variants() -> Iterator[Variant]:
    """The baseline, the discount, the block sizes, the window, sharing and terrain."""
    plain = replace(_REFERENCE, spread_peers=False)
    yield Variant("random waypoint", None)
    yield Variant("explore 16", replace(plain, block_sides=(16,)))
    yield Variant("explore 16 + discount", replace(_REFERENCE, block_sides=(16,)))
    yield Variant("explore 8", plain)
    yield Variant("explore 8 + discount", _REFERENCE)
    yield Variant("explore 8/16 + discount", replace(_REFERENCE, block_sides=(8, 16)))
    yield Variant("explore 8 + discount, window 0", replace(_REFERENCE, window_blocks=0))
    yield Variant("explore 8, no sharing", plain, share_observations=False)
    yield Variant("explore 8 + discount, no sharing", _REFERENCE, share_observations=False)
    yield Variant("random waypoint, obstacles", None, obstacles=True)
    yield Variant("explore 8, obstacles", plain, obstacles=True)
    yield Variant("explore 8 + discount, obstacles", _REFERENCE, obstacles=True)


def parameter_sweep() -> Iterator[Variant]:
    """The revisit age and ``beta``, one at a time, around the reference."""
    yield Variant("defaults (revisit age 28 min, beta 64)", _REFERENCE)
    for minutes in (14, 42, 84):
        age = minutes * TimeUnit.MINUTE
        yield Variant(f"revisit age {minutes} min", replace(_REFERENCE, revisit_age=age))
    for beta in (16.0, 32.0, 128.0):
        yield Variant(f"beta {beta:g}", replace(_REFERENCE, beta=beta))


def value_rules() -> Iterator[Variant]:
    """How stale cells grow and how blocks are valued, at two revisit ages.

    The saturating growth with the plain block mean is the rule of the earlier benchmarks; the
    others are the two fixes for holes, alone and together.
    """
    for minutes in (28, 56):
        aged = replace(_REFERENCE, revisit_age=minutes * TimeUnit.MINUTE)
        linear = replace(aged, growth=Growth.LINEAR)
        yield Variant(f"A {minutes}: saturating, mean", aged)
        for peak in (0.25, 0.5, 1.0):
            yield Variant(
                f"A {minutes}: saturating, lambda {peak:g}", replace(aged, block_peak=peak)
            )
        yield Variant(f"A {minutes}: linear, mean", linear)
        yield Variant(f"A {minutes}: linear, lambda 0.5", replace(linear, block_peak=0.5))


SECTIONS: dict[str, tuple[str, Callable[[], Iterator[Variant]]]] = {
    "variants": ("Variants", variants),
    "parameters": ("Parameters around the defaults", parameter_sweep),
    "values": ("Value rules", value_rules),
}
"""The tables the benchmark can print, by their name on the command line."""


def run(job: Job) -> Result:
    """Make one run and measure it."""
    variant = job.variant
    obstacles: tuple[CellId, ...] = ()
    if variant.obstacles:
        obstacles = river_and_rocks(job.size)
    end: Time = int(job.hours * TimeUnit.HOUR)
    scenario = ExploreScenario(
        width=job.size,
        height=job.size,
        robots=job.robots,
        obstacles=obstacles,
        share_observations=variant.share_observations,
        measure_from=end // 2,
        seed=job.seed,
    )

    def seeder_for(robot: RobotId) -> FieldSeeder:
        if variant.explore is None:
            return RandomWaypointSeeder(scenario.field_geometry(), obstacles, job.seed + robot)
        return ExploreSeeder(variant.explore)

    started = time.perf_counter()
    experiment = build_explore_run(scenario, seeder_for)
    experiment.advance_to(end)
    return Result(experiment.summary(), time.perf_counter() - started)


def table_row(name: str, results: list[Result]) -> str:
    """One Markdown table row: the means over the seeds' results."""
    summaries = [result.summary for result in results]
    cells = [
        name,
        _mean_minutes([summary.time_to_90 for summary in summaries]),
        _mean_minutes([summary.time_to_full for summary in summaries]),
        f"{mean(summary.mean_idleness for summary in summaries) / TimeUnit.MINUTE:.1f}",
        f"{mean(summary.worst_idleness for summary in summaries) / TimeUnit.MINUTE:.1f}",
        f"{mean(summary.redundancy for summary in summaries):.3f}",
        f"{mean(result.wall for result in results):.0f}",
    ]
    return "| " + " | ".join(cells) + " |"


def _mean_minutes(times: list[Time | None]) -> str:
    """The mean of ``times`` in minutes, or ``-`` if any is missing."""
    reached: list[Time] = []
    for time_ms in times:
        if time_ms is None:
            return "-"
        reached.append(time_ms)
    return f"{mean(reached) / TimeUnit.MINUTE:.1f}"


def print_table(title: str, section: list[Variant], results: list[Result], seeds: int) -> None:
    """Print one section as a Markdown table, ``seeds`` consecutive results per variant."""
    print(f"\n### {title}\n")
    headers = ["variant", "t90", "full", "idle", "worst", "redund", "wall (s)"]
    print("| " + " | ".join(headers) + " |")
    print("|" + " --- |" * len(headers))
    for index, variant in enumerate(section):
        print(table_row(variant.name, results[index * seeds : (index + 1) * seeds]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=32, help="field side in cells")
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--hours", type=float, default=3.0, help="simulated time per run")
    parser.add_argument("--seeds", type=int, default=4, help="runs per variant")
    parser.add_argument("--jobs", type=int, default=8, help="processes")
    parser.add_argument(
        "--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS), help="tables to print"
    )
    args = parser.parse_args()

    print(
        f"{args.robots} robots, {args.size} x {args.size} cells of 10 m, {args.hours} h, "
        f"mean over {args.seeds} seeds"
    )
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for name in args.sections:
            title, make_variants = SECTIONS[name]
            section = list(make_variants())
            jobs: list[Job] = []
            for variant in section:
                jobs.extend(
                    Job(variant, args.size, args.robots, args.hours, seed)
                    for seed in range(args.seeds)
                )
            print_table(title, section, list(pool.map(run, jobs)), args.seeds)


if __name__ == "__main__":
    main()
