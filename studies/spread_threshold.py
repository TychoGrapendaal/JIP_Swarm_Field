"""Study from which pest spread rate a swarm is needed, against a grid-search baseline (#74).

Question: up to which spread rate do utility robots that each sweep a strip of the field keep the
pests under control, and does the swarm hold out longer? Three systems, on the field, pests and
robots of the base demo of #67 (``SMOKE``):

- ``grid-10``: ten utility robots at 0.5 m/s that sweep their own strip and treat what they drive
  over, sharing nothing (``studies/baselines.py``);
- ``swarm-5+5``: the base demo itself: five explorers, five seeded killers;
- ``swarm-10+10``: the same swarm with twice the robots of each kind.

Every run starts with ``INITIAL_PESTS`` pests of severity ``INITIAL_SEVERITY`` on random cells
off the bottom row, drawn from the seed, so all systems face the same pests for the same seed. No
pest appears after that: the only way pests multiply is by spreading, so the spread rate is the
only variable.

Each system is run at every spread rate with the same seeds. A run is out of control when its
infested cells still grow at the end: their mean over the last hour is more than ``GROWTH_FACTOR``
times, and more than ``GROWTH_FLOOR`` cells above, their mean over the hour before. Hourly means
rather than single samples keep the ordinary ups and downs of a steady run from counting. A run
whose infestation has filled the field stops growing for want of healthy cells, so it is out of
control too when at least ``SATURATED_SHARE`` of the crop cells are infested over its last hour.
A system's threshold is the lowest spread rate at which most of its seeds are out of control.

The runs are measured from their metrics series and run in parallel by ``Experiment``, which
writes every run's result, the tables and a plot of the infested cells against the spread rate to
a new folder in ``output/``. Run from the repository root with:

    uv run python -m studies.spread_threshold --seeds 5 --hours 3 --jobs 8
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from statistics import mean, stdev

from studies.baselines import grid_sweep_brain
from swarm_field.contracts import MetricsSample, Time, TimeUnit
from swarm_field.experiments import (
    EXPLORER,
    SEEDED_KILLER,
    SMOKE,
    Experiment,
    Run,
    SwarmScenario,
    by_arm,
    runs_for,
)
from swarm_field.runner import ExperimentConfig, run_experiment

DEFAULT_SPREADS = (0.0, 0.005, 0.01, 0.02, 0.05)
INITIAL_PESTS = 10
INITIAL_SEVERITY = 0.1
"""The severity of a pest that has just appeared, as ``PestConfig.spawn_severity``."""
GROWTH_FACTOR = 1.5
GROWTH_FLOOR = 10
SATURATED_SHARE = 0.25
"""The share of the crop cells infested over the last hour from which a run is lost however
little it still grows, since a field that is mostly infested has few healthy cells left to
infest."""


def _without_new_pests(scenario: SwarmScenario, spread: float) -> SwarmScenario:
    """``scenario`` where no pest appears, so pests multiply only by spreading at ``spread``."""
    pests = replace(scenario.pests, pests_per_hour=0.0, spread_probability=spread)
    return scenario.with_pests(pests)


def _grid_sweep(robots: int) -> Callable[[SwarmScenario, float], ExperimentConfig]:
    def build(base: SwarmScenario, spread: float) -> ExperimentConfig:
        scenario = _without_new_pests(base, spread).without_robots()
        scenario = scenario.add_robots(SEEDED_KILLER, robots)
        # Utility robots share nothing, and each sweeps its strip with the grid-sweep brain.
        scenario = scenario.with_robot_config(replace(scenario.robots, radio_range_m=0.0))
        return replace(scenario.into_experiment(), brain=grid_sweep_brain(robots))

    return build


def _swarm(explorers: int, killers: int) -> Callable[[SwarmScenario, float], ExperimentConfig]:
    def build(base: SwarmScenario, spread: float) -> ExperimentConfig:
        scenario = _without_new_pests(base, spread).without_robots()
        scenario = scenario.add_robots(EXPLORER, explorers).add_robots(SEEDED_KILLER, killers)
        return scenario.into_experiment()

    return build


SYSTEMS: dict[str, Callable[[SwarmScenario, float], ExperimentConfig]] = {
    "grid-10": _grid_sweep(10),
    "swarm-5+5": _swarm(5, 5),
    "swarm-10+10": _swarm(10, 10),
}
"""Each system by name, building its run from ``SMOKE`` with a seed and horizon, at a spread."""


@dataclass(frozen=True)
class SpreadSpec:
    """One system at one spread rate, to run with every seed.

    Attributes:
        system: A name of ``SYSTEMS``.
        spread: The pests' spread probability.
        hours: The simulated time of every run.
    """

    system: str
    spread: float
    hours: float

    def into_experiment(self, seed: int) -> ExperimentConfig:
        """The run from ``seed``, with the initial pests, labelled ``SYSTEM/spread-SPREAD/SEED``."""
        base = replace(SMOKE, seed=seed, until=int(self.hours * TimeUnit.HOUR))
        config = with_initial_pests(SYSTEMS[self.system](base, self.spread), seed)
        label = f"{self.system}/spread-{self.spread:g}/{seed}"
        return replace(config, label=label, scenario_name="spread-threshold")


@dataclass(frozen=True)
class SpreadResult:
    """What one run measured, from its metrics series.

    Attributes:
        mean_infested: Infested crop cells, averaged over the samples.
        out_of_control: Whether its infested cells still grew at the end (``is_out_of_control``).
        clean_at: When the field was first free of pests, or ``None`` if it never was.
        pests_appeared: Pests that appeared over the run, the initial ones included.
        pests_cleared: Of those, the pests cleared.
    """

    mean_infested: float
    out_of_control: bool
    clean_at: Time | None
    pests_appeared: int
    pests_cleared: int


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


def measure_spread(run: Run[SpreadSpec]) -> SpreadResult:
    """Run one system, spread rate and seed, and reduce its metrics series to its result."""
    config = run.config.into_experiment(run.seed)
    series = run_experiment(config)
    clean_at: Time | None = None
    for sample in series:
        if sample.targets_remaining == 0:
            clean_at = sample.time
            break
    return SpreadResult(
        mean_infested=mean(sample.targets_remaining for sample in series),
        out_of_control=is_out_of_control(series, len(config.scenario.crops)),
        clean_at=clean_at,
        pests_appeared=series[-1].pests_appeared,
        pests_cleared=series[-1].pests_cleared,
    )


def is_out_of_control(series: Sequence[MetricsSample], crop_cells: int) -> bool:
    """Whether the run's infested cells still grew over its last hour, by their hourly means, or
    filled at least ``SATURATED_SHARE`` of its ``crop_cells`` then.

    A run shorter than two hours compares the second half of its samples with the first.
    """
    end = series[-1].time
    last: list[int] = []
    before: list[int] = []
    for sample in series:
        if sample.time > end - TimeUnit.HOUR:
            last.append(sample.targets_remaining)
        elif sample.time > end - 2 * TimeUnit.HOUR:
            before.append(sample.targets_remaining)
    if not before:
        half = len(series) // 2
        before = [sample.targets_remaining for sample in series[:half]]
        last = [sample.targets_remaining for sample in series[half:]]
    if not before or not last:
        return False
    if mean(last) >= SATURATED_SHARE * crop_cells:
        return True
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
        appeared: The pests that appeared, averaged over the seeds.
        cleared: Of those, the pests cleared, averaged over the seeds.
    """

    mean_infested: float
    spread_infested: float
    out_of_control: int
    seeds: int
    clean: int
    clean_after: float | None
    appeared: float
    cleared: float

    @classmethod
    def from_results(cls, results: list[SpreadResult]) -> Point:
        infested = [result.mean_infested for result in results]
        deviation = 0.0
        if len(infested) > 1:
            deviation = stdev(infested)
        lost = sum(1 for result in results if result.out_of_control)
        clean_times: list[Time] = []
        for result in results:
            if result.clean_at is not None:
                clean_times.append(result.clean_at)
        clean_after = None
        if clean_times:
            clean_after = mean(clean_times)
        return cls(
            mean_infested=mean(infested),
            spread_infested=deviation,
            out_of_control=lost,
            seeds=len(results),
            clean=len(clean_times),
            clean_after=clean_after,
            appeared=mean(result.pests_appeared for result in results),
            cleared=mean(result.pests_cleared for result in results),
        )

    def is_lost(self) -> bool:
        """Whether most seeds ended out of control."""
        return 2 * self.out_of_control > self.seeds


def collect_points(
    specs: dict[str, SpreadSpec], results: dict[str, list[SpreadResult]]
) -> dict[tuple[str, float], Point]:
    """Every system and spread rate's point, from the results of its runs."""
    points: dict[tuple[str, float], Point] = {}
    for arm, spec in specs.items():
        points[(spec.system, spec.spread)] = Point.from_results(results[arm])
    return points


def format_tables(
    systems: list[str], spreads: list[float], points: dict[tuple[str, float], Point]
) -> str:
    """Mean infested cells, clean fields and thresholds per system and spread rate, in Markdown."""
    header = "| Spread | " + " | ".join(systems) + " |"
    rule = "| --- |" + " --- |" * len(systems)
    lines = [
        "### Mean infested cells (standard deviation over the seeds; seeds out of control)",
        "",
        header,
        rule,
    ]
    for spread in spreads:
        cells: list[str] = []
        for system in systems:
            point = points[(system, spread)]
            cells.append(
                f"{point.mean_infested:.1f} (sd {point.spread_infested:.1f}; "
                f"{point.out_of_control}/{point.seeds} lost)"
            )
        lines.append(f"| {spread:g} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "### Field clean: seeds that cleared every pest, after how many minutes, and the "
            "pests cleared of those that appeared",
            "",
            header,
            rule,
        ]
    )
    for spread in spreads:
        cells = []
        for system in systems:
            point = points[(system, spread)]
            after = "-"
            if point.clean_after is not None:
                after = f"{point.clean_after / TimeUnit.MINUTE:.0f} min"
            cells.append(
                f"{point.clean}/{point.seeds}, {after}; {point.cleared:.1f} of {point.appeared:.1f}"
            )
        lines.append(f"| {spread:g} | " + " | ".join(cells) + " |")
    lines.extend(
        ["", "### Threshold: the lowest spread rate at which most seeds are out of control", ""]
    )
    for system in systems:
        threshold = "none in the range tested"
        for spread in spreads:
            if points[(system, spread)].is_lost():
                threshold = f"{spread:g}"
                break
        lines.append(f"- {system}: {threshold}")
    return "\n".join(lines)


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
    specs: dict[str, SpreadSpec] = {}
    for system in args.systems:
        for spread in args.spreads:
            specs[f"{system}/spread-{spread:g}"] = SpreadSpec(system, spread, args.hours)
    experiment = Experiment("spread_threshold")
    runs = runs_for(specs, range(args.seeds))
    results = by_arm(experiment.run(runs, measure_spread, args.jobs))
    points = collect_points(specs, results)
    tables = format_tables(args.systems, args.spreads, points)
    print(tables)
    experiment.write_text("tables.md", tables)
    plot_infested(args.systems, args.spreads, points, experiment.path("infested.png"))


if __name__ == "__main__":
    main()
