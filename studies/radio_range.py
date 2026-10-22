"""Study how far the robots' radio reaches against how fast the swarm finds and treats pests.

Question: in the pest experiment (``PestScenario``), how do the detection latency, the pests
left untreated and the amount of communication change with the radio range? Every range is run
with the same seeds, so two rows differ only in the range.

Every run writes its metrics series, and the whole study one summary row per run, to ``--out``
(``runs/radio_range`` by default); each row also records how its run was set up. The table
printed at the end is the mean over the seeds. Run from the repository root with:

    uv run python -m studies.radio_range --ranges 50 150 300 --seeds 3 --hours 1
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from statistics import mean

from swarm_field.contracts import TimeUnit
from swarm_field.experiments import PestScenario
from swarm_field.runner import (
    ExperimentConfig,
    RunSummary,
    export_summary_csv,
    export_summary_json,
    sweep,
)

DEFAULT_RANGES = (50.0, 150.0, 300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ranges", type=float, nargs="+", default=list(DEFAULT_RANGES), help="radio ranges in m"
    )
    parser.add_argument("--seeds", type=int, default=3, help="runs per range, seeds 0, 1, ...")
    parser.add_argument("--hours", type=float, default=1.0, help="simulated time per run")
    parser.add_argument(
        "--out", type=Path, default=Path("runs/radio_range"), help="where the results go"
    )
    return parser.parse_args()


def radio_range_config(radio_range_m: float, seed: int, hours: float) -> ExperimentConfig:
    """The pest experiment with one radio range and seed, labelled ``radio-RANGE/SEED``."""
    scenario = PestScenario(
        radio_range_m=radio_range_m, seed=seed, until=int(hours * TimeUnit.HOUR)
    )
    config = scenario.into_experiment(f"radio-{radio_range_m:g}/{seed}")
    return replace(config, scenario_name="pests")


def print_table(ranges: list[float], runs: tuple[RunSummary, ...], seeds: int) -> None:
    """One row per range: the mean over its seeds of the measures the question is about."""
    print(f"{'radio':>7}  {'latency':>11}  {'infested':>8}  {'coverage':>8}  {'received':>9}")
    for index, radio_range in enumerate(ranges):
        group = runs[index * seeds : (index + 1) * seeds]
        latencies: list[float] = []
        for run in group:
            if run.mean_detection_latency_ms is not None:
                latencies.append(run.mean_detection_latency_ms)
        latency = "-"
        if latencies:
            latency = f"{mean(latencies) / TimeUnit.MINUTE:.1f} min"
        infested = mean(run.targets_remaining for run in group)
        coverage = mean(run.coverage for run in group)
        received = mean(run.messages_received for run in group)
        print(
            f"{radio_range:>5g} m  {latency:>11}  {infested:>8.1f}  {coverage:>8.3f}  "
            f"{received:>9.0f}"
        )


def main() -> None:
    args = parse_args()
    configs: list[ExperimentConfig] = []
    for radio_range in args.ranges:
        for seed in range(args.seeds):
            configs.append(radio_range_config(radio_range, seed, args.hours))
    runs = sweep(configs)

    export_summary_csv(runs, args.out / "summary.csv")
    export_summary_json(runs, args.out / "summary.json")
    for run in runs:
        run.export(args.out)
    print_table(args.ranges, runs, args.seeds)
    print(f"\nWrote {len(runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
