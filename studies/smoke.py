"""Study the base demo: five explorers that only explore, five killers that only treat (#67).

Question: does the simplest swarm that does the whole job clear the pests its explorers find, and
which part of the pipeline limits it? The base run is ``SMOKE``; each comparison changes one part
of it:

- ``explorers-only``: no killers, the upper bound on infested cells;
- ``random-waypoint``: explorers drive to random cells instead of following the EXPLORE field;
- ``radio-50``, ``radio-150``, ``radio-unlimited``: where news of a pest stops reaching the
  killers (the base run has 300 m);
- ``killers-0.1``: killers at 0.1 m/s instead of 0.5, how much of the response is driving;
- ``spread``: pests spread to their neighbours, whether five killers keep up.

Every condition runs with the same seeds. The runs are measured from their metrics series and run
in parallel by ``Experiment``, which writes every run's result and the tables, the means over the
seeds in Markdown, to a new folder in ``output/``. Run from the repository root with:

    uv run python -m studies.smoke --seeds 5 --hours 3 --jobs 8
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from statistics import mean

from studies.baselines import RANDOM_WAYPOINT_BRAIN
from swarm_field.contracts import BodyType, TimeUnit
from swarm_field.experiments import SMOKE, Experiment, Run, SwarmScenario, by_arm, runs_for
from swarm_field.runner import ExperimentConfig, run_experiment

SPREAD_PROBABILITY = 0.05
"""The spread variant's chance per minute that a pest of severity 1 infests one neighbour."""


def _smoke(scenario: SwarmScenario) -> ExperimentConfig:
    return scenario.into_experiment()


def _explorers_only(scenario: SwarmScenario) -> ExperimentConfig:
    return scenario.without_robots(BodyType.KILLER).into_experiment()


def _random_waypoint(scenario: SwarmScenario) -> ExperimentConfig:
    # The explorers come first and have the default brain; the killers have one of their own.
    return replace(scenario.into_experiment(), brain=RANDOM_WAYPOINT_BRAIN)


def _radio(radio_range_m: float) -> Callable[[SwarmScenario], ExperimentConfig]:
    def build(scenario: SwarmScenario) -> ExperimentConfig:
        robots = replace(scenario.robots, radio_range_m=radio_range_m)
        return scenario.with_robot_config(robots).into_experiment()

    return build


def _slow_killers(scenario: SwarmScenario) -> ExperimentConfig:
    robots = replace(scenario.robots, speed_mps_ground=0.1)
    return scenario.with_robot_config(robots).into_experiment()


def _spread(scenario: SwarmScenario) -> ExperimentConfig:
    pests = replace(scenario.pests, spread_probability=SPREAD_PROBABILITY)
    return scenario.with_pests(pests).into_experiment()


CONDITIONS: dict[str, Callable[[SwarmScenario], ExperimentConfig]] = {
    "smoke": _smoke,
    "explorers-only": _explorers_only,
    "random-waypoint": _random_waypoint,
    "radio-50": _radio(50.0),
    "radio-150": _radio(150.0),
    "radio-unlimited": _radio(float("inf")),
    "killers-0.1": _slow_killers,
    "spread": _spread,
}
"""Each condition by name, building its run from ``SMOKE`` with the run's seed and horizon."""


@dataclass(frozen=True)
class SmokeSpec:
    """One condition of the study, to run with every seed.

    Attributes:
        condition: A name of ``CONDITIONS``.
        hours: The simulated time of every run.
    """

    condition: str
    hours: int

    def into_experiment(self, seed: int) -> ExperimentConfig:
        """The condition's run from ``seed``, labelled ``CONDITION/SEED``."""
        scenario = replace(SMOKE, seed=seed, until=self.hours * TimeUnit.HOUR)
        config = CONDITIONS[self.condition](scenario)
        return replace(config, label=f"{self.condition}/{seed}", scenario_name="smoke")


@dataclass(frozen=True)
class SmokeResult:
    """What one run of the study measured, from its metrics series.

    Attributes:
        killers: How many killers the run had.
        mean_infested: Infested crop cells, averaged over the samples.
        peak_infested: The most infested crop cells at any sample.
        pests_appeared: Pests that appeared over the run, including those present at the start.
        pests_cleared: Of those, the pests cleared.
        detection_ms: Mean time from a pest appearing to its detection, over the samples.
        knowledge_ms: Mean time from a pest's detection to a killer knowing of it.
        response_ms: Mean time from a pest's detection to its clearing.
        driving: The killers' share of decisions driving to a pest.
        treating: Their share treating.
        idle: Their share with nothing to do.
        duplicate_trips: Of their trip decisions, the share another killer duplicated.
        coverage: Fraction of the crop cells sensed by the end.
        mean_idleness_ms: Mean age of a crop cell, after the warm-up.
        worst_idleness_ms: The largest age of a crop cell, after the warm-up.
        redundancy: The share of visits to a cell sensed shortly before.
        messages_received: Times a message reached a robot in range.
    """

    killers: int
    mean_infested: float
    peak_infested: int
    pests_appeared: int
    pests_cleared: int
    detection_ms: float | None
    knowledge_ms: float | None
    response_ms: float | None
    driving: float
    treating: float
    idle: float
    duplicate_trips: float
    coverage: float
    mean_idleness_ms: float
    worst_idleness_ms: float
    redundancy: float
    messages_received: int


def measure_smoke(run: Run[SmokeSpec]) -> SmokeResult:
    """Run one condition and seed, and read its measures off the metrics series."""
    config = run.config.into_experiment(run.seed)
    series = run_experiment(config)
    final = series[-1]
    detections: list[float] = []
    for sample in series:
        if sample.detection_latency_ms is not None:
            detections.append(sample.detection_latency_ms)
    detection_ms: float | None = None
    if detections:
        detection_ms = mean(detections)
    return SmokeResult(
        killers=sum(1 for robot in config.scenario.robots if robot.body is BodyType.KILLER),
        mean_infested=mean(sample.targets_remaining for sample in series),
        peak_infested=max(sample.targets_remaining for sample in series),
        pests_appeared=final.pests_appeared,
        pests_cleared=final.pests_cleared,
        detection_ms=detection_ms,
        knowledge_ms=final.knowledge_latency_ms,
        response_ms=final.response_time_ms,
        driving=final.killer_driving_share,
        treating=final.killer_treating_share,
        idle=final.killer_idle_share,
        duplicate_trips=final.duplicate_trip_share,
        coverage=final.coverage,
        mean_idleness_ms=final.mean_idleness_ms,
        worst_idleness_ms=final.worst_idleness_ms,
        redundancy=final.redundancy,
        messages_received=final.messages_received,
    )


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
    parser.add_argument("--hours", type=int, default=3, help="simulated hours per run")
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    return parser.parse_args()


def _in_unit(values: Sequence[float | None], unit: int, digits: int) -> str:
    """The mean of the values present, in ``unit``, or ``-`` if none is."""
    present: list[float] = []
    for value in values:
        if value is not None:
            present.append(value)
    if not present:
        return "-"
    return f"{mean(present) / unit:.{digits}f}"


def _killer_columns(results: Sequence[SmokeResult]) -> str:
    """The killers' time split and duplicate trips, or dashes for a run without killers."""
    if sum(result.killers for result in results) == 0:
        return "- | -"
    return (
        f"{mean(r.driving for r in results):.0%} / "
        f"{mean(r.treating for r in results):.0%} / "
        f"{mean(r.idle for r in results):.0%} "
        f"| {mean(r.duplicate_trips for r in results):.0%}"
    )


def format_tables(results: dict[str, list[SmokeResult]]) -> str:
    """The pest response, and the monitoring and its cost, one row per condition."""
    lines = [
        "### Pest response",
        "",
        "| Condition | Infested (mean / peak) | Cleared / appeared | Detection (min) "
        "| News to killer (s) | Response (min) | Killers driving / treating / idle "
        "| Same pest |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for condition, runs in results.items():
        lines.append(
            f"| {condition} "
            f"| {mean(r.mean_infested for r in runs):.1f} / "
            f"{mean(r.peak_infested for r in runs):.1f} "
            f"| {mean(r.pests_cleared for r in runs):.1f} / "
            f"{mean(r.pests_appeared for r in runs):.1f} "
            f"| {_in_unit([r.detection_ms for r in runs], TimeUnit.MINUTE, 1)} "
            f"| {_in_unit([r.knowledge_ms for r in runs], TimeUnit.SECOND, 0)} "
            f"| {_in_unit([r.response_ms for r in runs], TimeUnit.MINUTE, 1)} "
            f"| {_killer_columns(runs)} |"
        )
    lines.extend(
        [
            "",
            "### Monitoring and cost",
            "",
            "| Condition | Coverage | Idleness (min, mean / worst) | Redundancy "
            "| Messages received |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for condition, runs in results.items():
        mean_idle = _in_unit([r.mean_idleness_ms for r in runs], TimeUnit.MINUTE, 1)
        worst_idle = _in_unit([r.worst_idleness_ms for r in runs], TimeUnit.MINUTE, 1)
        lines.append(
            f"| {condition} "
            f"| {mean(r.coverage for r in runs):.0%} "
            f"| {mean_idle} / {worst_idle} "
            f"| {mean(r.redundancy for r in runs):.0%} "
            f"| {mean(r.messages_received for r in runs):,.0f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    specs: dict[str, SmokeSpec] = {}
    for condition in args.conditions:
        specs[condition] = SmokeSpec(condition, args.hours)
    experiment = Experiment("smoke")
    results = by_arm(experiment.run(runs_for(specs, range(args.seeds)), measure_smoke, args.jobs))
    tables = format_tables(results)
    print(tables)
    experiment.write_text("tables.md", tables)


if __name__ == "__main__":
    main()
