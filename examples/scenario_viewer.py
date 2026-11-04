"""Watch, record or export one real experiment (C.1 over B.5).

``experiments.build_named_experiment`` builds any scenario of the catalogue, the presets and the
hand-built experiments alike, and ``runner.build_run`` assembles its robots and brains, so the
run is the same object graph ``runner.run_experiment`` drives; the console only advances it in
slices, which is what makes pausing and stepping possible. The console itself (time controls,
injecting, inspecting, recording and replaying) is ``swarm_field.console``; see the README for
every flag.

Each scenario runs for its own horizon unless ``--hours`` sets one. The metrics cadence is at
most a minute, because the presets' default of one sample per hour would leave the figures in
the title static for the whole run. Sampling cadence only changes what is measured, never what
the world does: ``MetricsCollector`` holds no random source.

Run in a window:
    uv run python examples/scenario_viewer.py --scenario S3 --seed 7 --hours 6
    uv run python examples/scenario_viewer.py --scenario pests

Run a scenario with changed parameters, described in a JSON file of ``configs/`` (one condition
of it, the first unless ``--condition`` names another; its first seed unless ``--seed`` is given):
    uv run python examples/scenario_viewer.py --config configs/pests.json
    uv run python examples/scenario_viewer.py --config configs/radio_range.json --condition radio-50

Record the whole run without a window into a video, and export its metrics alongside it:
    uv run python examples/scenario_viewer.py --hours 6 --speed 16 --save runs/s2.mp4 --out runs/s2
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from swarm_field.console import (
    InteractiveConsole,
    add_session_arguments,
    format_time,
    save_session,
    show_window,
    validate_session_arguments,
    write_recording,
)
from swarm_field.contracts import CommandLog, Time, TimeUnit
from swarm_field.experiments import LiveRun, build_named_experiment, scenario_names
from swarm_field.experiments.from_json import ExperimentFile
from swarm_field.runner import ExperimentConfig
from swarm_field.simulation import write_metrics_csv, write_metrics_json
from swarm_field.visualization import DEFAULT_TRAIL_LENGTH, MatplotlibRenderer

DEFAULT_SCENARIO = "S2"
DEFAULT_SEED = 101
DEFAULT_SPEED = 4.0
METRICS_PERIOD_MS = TimeUnit.MINUTE


def parse_args() -> argparse.Namespace:
    """Read the command line: which scenario to run, for how long, and what to write."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        choices=scenario_names(),
        default=None,
        help=f"scenario to run, by name (default: {DEFAULT_SCENARIO})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON experiment file to run instead of a named scenario, as in configs/",
    )
    parser.add_argument(
        "--condition",
        default=None,
        help="condition of the --config file to run (default: its first)",
    )
    add_session_arguments(parser, default_speed=DEFAULT_SPEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory to write the run's metrics series and summary to when it ends",
    )
    args = parser.parse_args()
    validate_session_arguments(parser, args)
    if args.replay is not None and args.scenario is not None:
        parser.error("--scenario cannot be combined with --replay: the recording holds it")
    if args.config is not None:
        # A recording names a scenario of the catalogue, which a JSON file's run is not.
        for flag, value in (("--scenario", args.scenario), ("--replay", args.replay)):
            if value is not None:
                parser.error(f"--config cannot be combined with {flag}")
        if args.record is not None:
            parser.error("--config cannot be combined with --record: it could not be replayed")
    elif args.condition is not None:
        parser.error("--condition needs --config")
    return args


def build_file_experiment(
    path: Path, condition: str | None, seed: int | None, until: Time | None
) -> ExperimentConfig:
    """The run of one condition of the JSON experiment file at ``path``.

    Args:
        path: An experiment file, as ``from_json`` runs it.
        condition: The condition to run, or ``None`` for the file's first.
        seed: The seed of the run, or ``None`` for the file's first.
        until: How far to run, or ``None`` for the file's ``hours``.
    """
    experiment = ExperimentFile.from_path(path)
    scenarios = experiment.into_scenarios()
    if condition is None:
        condition = next(iter(scenarios))
    if condition not in scenarios:
        raise SystemExit(f"{path} has no condition {condition!r}; it has {sorted(scenarios)}")
    scenario = scenarios[condition]
    if seed is None:
        seed = experiment.seeds[0]
    scenario = scenario.with_seed(seed)
    if until is not None:
        scenario = replace(scenario, until=until)
    return scenario.into_experiment(f"{condition}/{seed}")


def export_run(run: LiveRun, out: Path) -> None:
    """Write the run's metrics series to ``out`` as ``metrics.csv`` and ``metrics.json``."""
    out.mkdir(parents=True, exist_ok=True)
    series = run.metrics.series()
    write_metrics_csv(series, out / "metrics.csv")
    write_metrics_json(series, out / "metrics.json")
    print(f"Wrote the metrics series ({len(series)} samples) to {out}")


def main() -> None:
    """Run one seeded scenario in a window or without one, then write what was asked for."""
    args = parse_args()
    log: CommandLog | None = None
    name = DEFAULT_SCENARIO
    seed = DEFAULT_SEED
    until: Time | None = None
    if args.replay is not None:
        log = CommandLog.from_json(args.replay.read_text(encoding="utf-8"))
        if log.scenario not in scenario_names():
            raise SystemExit(f"{args.replay} was not recorded from a scenario of the catalogue")
        name = log.scenario
        seed = log.seed
        until = log.until
    else:
        if args.scenario is not None:
            name = args.scenario
        if args.seed is not None:
            seed = args.seed
        if args.hours is not None:
            until = int(args.hours * TimeUnit.HOUR)

    if args.config is not None:
        config = build_file_experiment(args.config, args.condition, args.seed, until)
    else:
        config = build_named_experiment(name, seed, until)
    period = min(config.metrics.period, METRICS_PERIOD_MS)
    config = replace(config, metrics=replace(config.metrics, period=period))
    run = LiveRun.from_config(config, log)
    console = InteractiveConsole(
        run,
        MatplotlibRenderer(show_targets=False),
        speed=args.speed,
        trail_length=DEFAULT_TRAIL_LENGTH,
        show_targets=True,
    )

    if args.save is None:
        show_window(console)
    else:
        frames = save_session(console, args.save)
        print(f"Wrote {frames} frames up to {format_time(run.now)} to {args.save}")
    if args.record is not None:
        write_recording(run.command_log(), args.record)
        print(f"Recorded {len(run.recorded)} commands up to {format_time(run.now)}")
    if args.out is not None:
        export_run(run, args.out)


if __name__ == "__main__":
    main()
