"""Study the EXPLORE seeder: variants and parameters of a swarm exploring one field.

Each run is a ``swarm_field.experiments`` exploration run by ``swarm_field.runner``, repeated
with several seeds (which stagger the robots' decisions). A row reports the ground-truth
measures of ``MetricsCollector``, sampled once a minute, as the mean over the seeds:

- ``t90`` and ``full``: when 90 % and all of the field had been sensed, in minutes; ``-`` if
  some seed never got there;
- ``idle`` and ``worst``: mean and worst idleness over the second half of the run, in minutes;
  the mean is the expected detection latency of a pest appearing at a random time and cell;
- ``redund``: the share of visits to a cell sensed less than 30 minutes before;
- ``wall``: seconds of computation per run.

Runs are spread over processes. Not part of ``pytest``; run it with:

    uv run python -m studies.explore_variants --size 32 --hours 3 --seeds 4
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from statistics import mean

from tests.baselines import RANDOM_WAYPOINT_BRAIN

from swarm_field.contracts import CellId, MetricsSample, Time, TimeUnit
from swarm_field.experiments import EXPLORE, ExplorerPreset, MapConfig, river_and_rocks
from swarm_field.runner import run_experiment
from swarm_field.swarm import ExploreConfig

_REFERENCE = ExploreConfig()
"""The variant the others are compared with: the defaults, 8 x 8 blocks with the discount."""


@dataclass(frozen=True)
class Variant:
    """One configuration of the swarm.

    Attributes:
        name: The row's label.
        explore: The EXPLORE parameters, or ``None`` for the random-waypoint baseline.
        radio_range_m: How far the robots' broadcasts reach; 0 for robots that share nothing.
        obstacles: Whether the field has the river and rocks of ``river_and_rocks``.
    """

    name: str
    explore: ExploreConfig | None
    radio_range_m: float = float("inf")
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
    """The metrics series of one run, and the seconds it took."""

    series: tuple[MetricsSample, ...]
    wall: float

    def time_to_cover(self, share: float) -> Time | None:
        """The first sample at which ``share`` of the field had been sensed, or ``None``."""
        for sample in self.series:
            if sample.coverage >= share:
                return sample.time
        return None


def variants() -> Iterator[Variant]:
    """The baseline, the discount, the block sizes, the window, sharing and terrain."""
    plain = replace(_REFERENCE, spread_peers=False)
    yield Variant("random waypoint", None)
    yield Variant("explore 16", replace(plain, block_side=16))
    yield Variant("explore 16 + discount", replace(_REFERENCE, block_side=16))
    yield Variant("explore 8", plain)
    yield Variant("explore 8 + discount", _REFERENCE)
    yield Variant("explore 8 + discount, window 0", replace(_REFERENCE, window_blocks=0))
    yield Variant("explore 8, no sharing", plain, radio_range_m=0.0)
    yield Variant("explore 8 + discount, no sharing", _REFERENCE, radio_range_m=0.0)
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


SECTIONS: dict[str, tuple[str, Callable[[], Iterator[Variant]]]] = {
    "variants": ("Variants", variants),
    "parameters": ("Parameters around the defaults", parameter_sweep),
}
"""The tables the benchmark can print, by their name on the command line."""


def run(job: Job) -> Result:
    """Make one run and measure it."""
    variant = job.variant
    obstacles: tuple[CellId, ...] = ()
    if variant.obstacles:
        obstacles = river_and_rocks(job.size)
    end: Time = int(job.hours * TimeUnit.HOUR)
    explore = _REFERENCE
    if variant.explore is not None:
        explore = variant.explore
    field = (
        EXPLORE.without_robots()
        .with_map(MapConfig(size=job.size, obstacles=obstacles))
        .with_robot_config(replace(EXPLORE.robots, radio_range_m=variant.radio_range_m))
        .with_metrics(replace(EXPLORE.metrics, measure_from=end // 2))
    )
    scenario = replace(
        field.add_robots(ExplorerPreset(explore=explore), job.robots), until=end, seed=job.seed
    )
    experiment = scenario.into_experiment(variant.name)
    if variant.explore is None:
        experiment = replace(experiment, brain=RANDOM_WAYPOINT_BRAIN)
    started = time.perf_counter()
    summary = run_experiment(experiment)
    return Result(summary.series, time.perf_counter() - started)


def table_row(name: str, results: list[Result]) -> str:
    """One Markdown table row: the means over the seeds' results."""
    finals = [result.series[-1] for result in results]
    cells = [
        name,
        _mean_minutes([result.time_to_cover(0.9) for result in results]),
        _mean_minutes([result.time_to_cover(1.0) for result in results]),
        f"{mean(final.mean_idleness_ms for final in finals) / TimeUnit.MINUTE:.1f}",
        f"{mean(final.worst_idleness_ms for final in finals) / TimeUnit.MINUTE:.1f}",
        f"{mean(final.redundancy for final in finals):.3f}",
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
