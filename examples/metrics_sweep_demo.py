"""Run a seeded scenario sweep and export the metrics series and the summary table.

Run from the repository root with:
    uv run python examples/metrics_sweep_demo.py

Write the exports to a directory instead of only printing them:
    uv run python examples/metrics_sweep_demo.py --out runs/sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swarm_field.contracts import ScenarioId
from swarm_field.runner import RunSummary, export_summary_csv, export_summary_json, sweep_scenarios
from swarm_field.simulation import MetricsConfig

HOUR = 3_600_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory to write the summary table and each run's series to",
    )
    parser.add_argument("--seed", type=int, default=101, help="first seed of the sweep")
    parser.add_argument(
        "--hours", type=float, default=6.0, help="how many simulation hours each run lasts"
    )
    return parser.parse_args()


def print_summary_table(runs: tuple[RunSummary, ...]) -> None:
    """Print the sweep as a table, one row per run."""
    header = f"{'run':>10}  {'targets':>8}  {'coverage':>8}  {'latency':>9}  {'energy/m2':>10}"
    print(header)
    print("-" * len(header))
    for run in runs:
        latency = run.mean_detection_latency_ms
        shown = "n/a" if latency is None else f"{latency:,.0f}"
        print(
            f"{run.label:>10}  {run.targets_remaining:>8}  {run.coverage:>8.3f}  "
            f"{shown:>9}  {run.energy_per_area:>10.3e}"
        )


def print_series(run: RunSummary) -> None:
    """Print one run's whole metrics series, sample by sample."""
    print(f"\n{run.label} series ({len(run.series)} samples):")
    for sample in run.series:
        print(
            f"  t={sample.time:>9}  targets={sample.targets_remaining:>5}  "
            f"coverage={sample.coverage:.3f}  energy/m2={sample.energy_per_area:.3e}"
        )


def main() -> None:
    args = parse_args()
    runs = sweep_scenarios(
        [ScenarioId.S1, ScenarioId.S2, ScenarioId.S3],
        seeds=[args.seed, args.seed + 1],
        until=int(args.hours * HOUR),
        metrics=MetricsConfig(period=HOUR),
    )

    print_summary_table(runs)
    print_series(runs[0])

    if args.out is not None:
        export_summary_csv(runs, args.out / "summary.csv")
        export_summary_json(runs, args.out / "summary.json")
        for run in runs:
            run.export(args.out)
        print(f"\nWrote the summary table and {2 * len(runs)} series files to {args.out}")


if __name__ == "__main__":
    main()
