"""Measure how the swarm degrades as robots fail or mislocate themselves.

Run from the repository root with:
    uv run python examples/robustness_demo.py

Sweep every failure mode and write the table to a directory:
    uv run python examples/robustness_demo.py --out runs/robustness
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swarm_field.contracts import FailureMode, FailureSpec, PositionErrorSpec, ScenarioId
from swarm_field.runner import (
    RobustnessCase,
    RunSummary,
    export_summary_csv,
    export_summary_json,
    robustness_sweep,
)
from swarm_field.simulation import MetricsConfig

HOUR = 3_600_000
FAILURE_RATES = (0.001, 0.005)
POSITION_ERROR_RATES = (0.1, 0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=None, help="directory to write the results table to"
    )
    parser.add_argument("--seed", type=int, default=101, help="seed shared by every condition")
    parser.add_argument(
        "--scenario",
        type=ScenarioId,
        choices=list(ScenarioId),
        default=ScenarioId.S2,
        help="scenario to run every condition against",
    )
    parser.add_argument(
        "--hours", type=float, default=6.0, help="how many simulation hours each run lasts"
    )
    return parser.parse_args()


def all_cases() -> tuple[RobustnessCase, ...]:
    """A baseline, then every failure mode at two rates, then two position-error rates.

    One axis is perturbed at a time, so each row answers a single question: how much does the
    swarm lose to this fault, at this rate, on its own.
    """
    cases = [RobustnessCase(label="baseline")]
    for mode in FailureMode:
        for probability in FAILURE_RATES:
            label = f"failure-{mode.name.lower()}-{probability:g}"
            cases.append(
                RobustnessCase(
                    label=label,
                    failures=FailureSpec(period=60_000, probability=probability, mode=mode),
                )
            )
    for probability in POSITION_ERROR_RATES:
        label = f"position-error-{probability:g}"
        cases.append(
            RobustnessCase(
                label=label,
                position_error=PositionErrorSpec(probability=probability),
            )
        )
    return tuple(cases)


def print_table(runs: tuple[RunSummary, ...]) -> None:
    """Print the conditions as a table, with each row's coverage difference from the baseline."""
    baseline = runs[0]
    header = (
        f"{'condition':<32} {'active':>6} {'targets':>8} {'coverage':>8} "
        f"{'vs base':>8} {'latency':>9} {'energy/m2':>10}"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        latency = run.mean_detection_latency_ms
        shown = "n/a" if latency is None else f"{latency:,.0f}"
        print(
            f"{run.label:<32} {run.active_robots:>6} {run.targets_remaining:>8} "
            f"{run.coverage:>8.3f} {run.coverage - baseline.coverage:>+8.3f} "
            f"{shown:>9} {run.energy_per_area:>10.3e}"
        )


def main() -> None:
    args = parse_args()
    runs = robustness_sweep(
        args.scenario,
        all_cases(),
        seed=args.seed,
        until=int(args.hours * HOUR),
        metrics=MetricsConfig(period=HOUR),
    )

    print_table(runs)

    if args.out is not None:
        export_summary_csv(runs, args.out / "robustness.csv")
        export_summary_json(runs, args.out / "robustness.json")
        for run in runs:
            run.export(args.out)
        print(f"\nWrote the table and {2 * len(runs)} series files to {args.out}")


if __name__ == "__main__":
    main()
