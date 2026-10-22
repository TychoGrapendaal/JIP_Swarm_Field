"""Measure how the swarm degrades as robots fail or mislocate themselves.

Run from the repository root with:
    uv run python examples/robustness_demo.py

Sweep every failure mode and write the table to a directory:
    uv run python examples/robustness_demo.py --out runs/robustness

Run the conditions against any scenario of the catalogue, e.g. the pest experiment:
    uv run python examples/robustness_demo.py --scenario pests --hours 1

Watch one condition as PNG frames, one per metric tick, with lost robots drawn hollow:
    uv run --with matplotlib python examples/robustness_demo.py \
        --render failure-offline-0.005
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from swarm_field.contracts import FailureMode, FailureSpec, PositionErrorSpec, TimeUnit
from swarm_field.experiments import build_named_experiment, scenario_names
from swarm_field.runner import (
    ExperimentConfig,
    RobustnessCase,
    RunSummary,
    export_summary_csv,
    export_summary_json,
    robustness_case_config,
    robustness_sweep,
    run_experiment,
)
from swarm_field.visualization import FrameWriter, MatplotlibRenderer

HOUR = TimeUnit.HOUR
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
        choices=scenario_names(),
        default="S2",
        help="scenario to run every condition against, by name (default: S2)",
    )
    parser.add_argument(
        "--hours", type=float, default=6.0, help="how many simulation hours each run lasts"
    )
    parser.add_argument(
        "--render",
        type=str,
        default=None,
        metavar="CONDITION",
        help=(
            "write PNG frames of one condition to --render-out, one per metric tick. "
            "Defaults to every condition when no name is given."
        ),
    )
    parser.add_argument(
        "--render-out",
        type=Path,
        default=Path("runs/frames"),
        help="directory the rendered frames are written to",
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


def select_case(cases: tuple[RobustnessCase, ...], label: str) -> RobustnessCase:
    """Find one condition by name, listing the available ones if it does not exist."""
    for case in cases:
        if case.label == label:
            return case
    names = ", ".join(case.label for case in cases)
    raise SystemExit(f"no condition named {label!r}; choose one of: {names}")


def render_condition(case: RobustnessCase, base: ExperimentConfig, directory: Path) -> int:
    """Re-run one condition with a renderer and write its frames. Returns the frame count.

    The sweep above already ran this exact case, and a run is a function of its seed, so drawing
    it a second time reproduces the row in the table rather than a different experiment. That is
    why the frames can be trusted against the numbers next to them.
    """
    config = robustness_case_config(base, case)
    writer = FrameWriter(MatplotlibRenderer(show_targets=False), directory, case.label)
    run_experiment(config, writer)
    return writer.frames_written


def main() -> None:
    args = parse_args()
    until = int(args.hours * HOUR)
    cases = all_cases()
    base = build_named_experiment(args.scenario, args.seed, until)
    base = replace(base, metrics=replace(base.metrics, period=HOUR))

    runs = robustness_sweep(base, cases)

    print_table(runs)

    if args.out is not None:
        export_summary_csv(runs, args.out / "robustness.csv")
        export_summary_json(runs, args.out / "robustness.json")
        for run in runs:
            run.export(args.out)
        print(f"\nWrote the table and {2 * len(runs)} series files to {args.out}")

    if args.render is not None:
        case = select_case(cases, args.render)
        written = render_condition(case, base, args.render_out)
        print(f"Wrote {written} frames of {case.label} to {args.render_out}")


if __name__ == "__main__":
    main()
