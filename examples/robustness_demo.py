"""Measure how the swarm degrades as robots fail or mislocate themselves.

Every condition (``RobustnessCase``) runs against one scenario of the catalogue with the same
seeds, in parallel through ``Experiment``, which writes the runs and the table to a new folder in
``output/``. A row is the mean over the seeds of the run's last metrics sample.

Run from the repository root with:
    uv run python examples/robustness_demo.py

Run the conditions against any scenario of the catalogue, e.g. the pest experiment:
    uv run python examples/robustness_demo.py --scenario pests --hours 1

Watch one condition as PNG frames, one per metric tick, with lost robots drawn hollow:
    uv run --with matplotlib python examples/robustness_demo.py \\
        --render failure-offline-0.005
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from swarm_field.contracts import FailureMode, MetricsSample, TimeUnit
from swarm_field.experiments import (
    Experiment,
    RobustnessCase,
    RobustnessRunSpec,
    by_arm,
    default_robustness_cases,
    markdown_table,
    measure_robustness,
    runs_for,
    scenario_names,
)
from swarm_field.runner import run_experiment
from swarm_field.visualization import FrameWriter, MatplotlibRenderer

FAILURE_RATES = (0.001, 0.005)
POSITION_ERROR_RATES = (0.1, 0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=1, help="runs per condition, from --seed")
    parser.add_argument("--seed", type=int, default=101, help="the first seed of every condition")
    parser.add_argument(
        "--scenario",
        choices=scenario_names(),
        default="S2",
        help="scenario to run every condition against, by name (default: S2)",
    )
    parser.add_argument(
        "--hours", type=float, default=6.0, help="how many simulation hours each run lasts"
    )
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    parser.add_argument(
        "--render",
        type=str,
        default=None,
        metavar="CONDITION",
        help="write PNG frames of one condition to --render-out, one per metric tick",
    )
    parser.add_argument(
        "--render-out",
        type=Path,
        default=Path("runs/frames"),
        help="directory the rendered frames are written to",
    )
    return parser.parse_args()


def table(results: dict[str, list[MetricsSample]]) -> str:
    """The conditions as a Markdown table, with each row's coverage difference from the first."""
    headers = ["condition", "active", "targets", "coverage", "vs first", "latency (min)"]
    first = mean(final.coverage for final in next(iter(results.values())))
    rows: list[list[str]] = []
    for condition, finals in results.items():
        coverage = mean(final.coverage for final in finals)
        latencies: list[float] = []
        for final in finals:
            if final.detection_latency_ms is not None:
                latencies.append(final.detection_latency_ms / TimeUnit.MINUTE)
        latency = "-"
        if latencies:
            latency = f"{mean(latencies):.1f}"
        rows.append(
            [
                condition,
                f"{mean(final.active_robots for final in finals):.1f}",
                f"{mean(final.targets_remaining for final in finals):.1f}",
                f"{coverage:.3f}",
                f"{coverage - first:+.3f}",
                latency,
            ]
        )
    return markdown_table(headers, rows)


def select_case(cases: tuple[RobustnessCase, ...], label: str) -> RobustnessCase:
    """Find one condition by name, listing the available ones if it does not exist."""
    for case in cases:
        if case.label == label:
            return case
    names = ", ".join(case.label for case in cases)
    raise SystemExit(f"no condition named {label!r}; choose one of: {names}")


def render_condition(spec: RobustnessRunSpec, seed: int, directory: Path) -> int:
    """Run one condition again with a renderer and write its frames; returns the frame count.

    A run is a function of its seed, so the frames show exactly the run of the table's row.
    """
    writer = FrameWriter(MatplotlibRenderer(show_targets=False), directory, spec.case.label)
    run_experiment(spec.into_experiment(seed), writer)
    return writer.frames_written


def main() -> None:
    args = parse_args()
    cases = default_robustness_cases(FAILURE_RATES, POSITION_ERROR_RATES, modes=tuple(FailureMode))
    specs: dict[str, RobustnessRunSpec] = {}
    for case in cases:
        specs[case.label] = RobustnessRunSpec(args.scenario, args.hours, case)

    if args.render is not None:
        case = select_case(cases, args.render)
        written = render_condition(specs[case.label], args.seed, args.render_out)
        print(f"Wrote {written} frames of {case.label} to {args.render_out}")
        return

    experiment = Experiment(f"robustness_{args.scenario}")
    seeds = range(args.seed, args.seed + args.seeds)
    results = by_arm(experiment.run(runs_for(specs, seeds), measure_robustness, args.jobs))
    text = table(results)
    print(text)
    experiment.write_text("tables.md", text)


if __name__ == "__main__":
    main()
