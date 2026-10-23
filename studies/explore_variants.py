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

The runs go through ``Experiment``, in parallel, which writes them and the tables to a new
folder in ``output/``. Not part of ``pytest``; run it with:

    uv run python -m studies.explore_variants --size 32 --hours 3 --seeds 4
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from statistics import mean

from tests.baselines import RANDOM_WAYPOINT_BRAIN

from swarm_field.contracts import BodyType, CellId, MetricsSample, Time, TimeUnit
from swarm_field.experiments import (
    EXPLORE,
    CustomPreset,
    Experiment,
    ExplorerPreset,
    MapConfig,
    RobotPreset,
    Run,
    RunResult,
    SwarmScenario,
    arm_name,
    markdown_table,
    mean_and_error,
    river_and_rocks,
    runs_for,
)
from swarm_field.runner import run_experiment
from swarm_field.swarm import ExploreConfig

_REFERENCE = ExploreConfig()
"""The variant the others are compared with: the defaults, 8 x 8 blocks with the discount."""
RANDOM_WAYPOINT = CustomPreset("random-waypoint", BodyType.SCOUT, RANDOM_WAYPOINT_BRAIN)
"""The baseline: robots that walk to one random cell after the other."""


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
class ExploreRunSpec:
    """One variant on a field of ``size`` cells with ``robots`` robots, for ``hours``.

    Attributes:
        variant: The configuration of the swarm.
        size: The field's side in cells.
        robots: How many robots explore it.
        hours: How long a run lasts.
    """

    variant: Variant
    size: int
    robots: int
    hours: float

    def into_scenario(self, seed: int) -> SwarmScenario:
        """The run with ``seed``: idleness is measured over the second half of it."""
        variant = self.variant
        obstacles: tuple[CellId, ...] = ()
        if variant.obstacles:
            obstacles = river_and_rocks(self.size)
        end: Time = int(self.hours * TimeUnit.HOUR)
        field = (
            EXPLORE.without_robots()
            .with_map(MapConfig(size=self.size, obstacles=obstacles))
            .with_robot_config(replace(EXPLORE.robots, radio_range_m=variant.radio_range_m))
            .with_metrics(replace(EXPLORE.metrics, measure_from=end // 2))
        )
        preset: RobotPreset = RANDOM_WAYPOINT
        if variant.explore is not None:
            preset = ExplorerPreset(explore=variant.explore)
        return replace(field.add_robots(preset, self.robots), until=end, seed=seed)


@dataclass(frozen=True)
class Exploration:
    """What one run measured, from its metrics series sampled once a minute.

    Attributes:
        t90_minutes: When 90 % of the field had been sensed, or ``None`` if it never was.
        full_minutes: When all of the field had been sensed, or ``None`` if it never was.
        idleness_minutes: Mean idleness over the second half of the run.
        worst_idleness_minutes: Worst idleness over the second half of the run.
        redundancy: The share of visits to a cell sensed less than 30 minutes before.
    """

    t90_minutes: float | None
    full_minutes: float | None
    idleness_minutes: float
    worst_idleness_minutes: float
    redundancy: float


def measure_exploration(run: Run[ExploreRunSpec]) -> Exploration:
    """Make one run of the spec with the run's seed and measure it; a run function."""
    scenario = run.config.into_scenario(run.seed)
    series = run_experiment(scenario.into_experiment(run.arm))
    final = series[-1]
    return Exploration(
        t90_minutes=_time_to_cover(series, 0.9),
        full_minutes=_time_to_cover(series, 1.0),
        idleness_minutes=final.mean_idleness_ms / TimeUnit.MINUTE,
        worst_idleness_minutes=final.worst_idleness_ms / TimeUnit.MINUTE,
        redundancy=final.redundancy,
    )


def _time_to_cover(series: tuple[MetricsSample, ...], share: float) -> float | None:
    """The minute of the first sample at which ``share`` of the field had been sensed."""
    for sample in series:
        if sample.coverage >= share:
            return sample.time / TimeUnit.MINUTE
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


def table_row(name: str, results: list[RunResult[ExploreRunSpec, Exploration]]) -> list[str]:
    """One table row: the means over the seeds' results, ``-`` where a seed never got there."""
    measured = [result.result for result in results]
    return [
        name,
        mean_and_error([exploration.t90_minutes for exploration in measured]),
        mean_and_error([exploration.full_minutes for exploration in measured]),
        f"{mean(exploration.idleness_minutes for exploration in measured):.1f}",
        f"{mean(exploration.worst_idleness_minutes for exploration in measured):.1f}",
        f"{mean(exploration.redundancy for exploration in measured):.3f}",
        f"{mean(result.wall for result in results):.0f}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=32, help="field side in cells")
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--hours", type=float, default=3.0, help="simulated time per run")
    parser.add_argument("--seeds", type=int, default=4, help="runs per variant")
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    parser.add_argument(
        "--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS), help="tables to print"
    )
    args = parser.parse_args()

    specs: dict[str, ExploreRunSpec] = {}
    for name in args.sections:
        _, make_variants = SECTIONS[name]
        for variant in make_variants():
            specs[arm_name(name, variant.name)] = ExploreRunSpec(
                variant, args.size, args.robots, args.hours
            )
    experiment = Experiment("explore_variants")
    results = experiment.run(runs_for(specs, range(args.seeds)), measure_exploration, args.jobs)
    by_variant: dict[str, list[RunResult[ExploreRunSpec, Exploration]]] = {}
    for result in results:
        by_variant.setdefault(result.run.arm, []).append(result)

    text = (
        f"{args.robots} robots, {args.size} x {args.size} cells of 10 m, {args.hours} h, "
        f"mean over {args.seeds} seeds\n"
    )
    headers = ["variant", "t90", "full", "idle", "worst", "redund", "wall (s)"]
    for name in args.sections:
        title, make_variants = SECTIONS[name]
        rows: list[list[str]] = []
        for variant in make_variants():
            rows.append(table_row(variant.name, by_variant[arm_name(name, variant.name)]))
        text += f"\n### {title}\n\n{markdown_table(headers, rows)}"
    print(text)
    experiment.write_text("tables.md", text)


if __name__ == "__main__":
    main()
