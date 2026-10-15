"""Watch, record or export one real experiment (C.1 over B.5).

``runner.build_experiment`` resolves the scenario preset and ``runner.build_run`` assembles the
robot roster and brains, so the run is the same object graph ``runner.run_experiment`` drives;
the console only advances it in slices, which is what makes pausing and stepping possible. The
console itself (time controls, injecting, inspecting, recording and replaying) is
``swarm_field.console``; see the README for every flag.

A scenario preset carries no duration of its own, so ``--hours`` sets the horizon. The metrics
cadence is shortened to match the pest dynamics, because the default of one sample per hour
would leave the figures in the title static for the whole run. Sampling cadence only changes
what is measured, never what the world does: ``MetricsCollector`` holds no random source.

Run in a window:
    uv run python examples/scenario_viewer.py --scenario S3 --seed 7 --hours 6

Record the whole run without a window into a video, and export its metrics alongside it:
    uv run python examples/scenario_viewer.py --hours 6 --speed 16 --save runs/s2.mp4 --out runs/s2
"""

from __future__ import annotations

import argparse
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
from swarm_field.contracts import CommandLog, ScenarioId, TimeUnit
from swarm_field.runner import (
    RunSummary,
    build_experiment,
    build_run,
    export_summary_csv,
    export_summary_json,
)
from swarm_field.simulation import CommandReplay, MetricsCollector, MetricsConfig
from swarm_field.visualization import MatplotlibRenderer

DEFAULT_SCENARIO = ScenarioId.S2
DEFAULT_SEED = 101
DEFAULT_HOURS = 1.0
DEFAULT_SPEED = 4.0
METRICS_PERIOD_MS = TimeUnit.MINUTE


def parse_scenario(name: str) -> ScenarioId:
    """Resolve a scenario name such as ``S2`` to its preset, or exit with the valid names."""
    try:
        return ScenarioId[name]
    except KeyError:
        choices = ", ".join(member.name for member in ScenarioId)
        raise SystemExit(f"unknown scenario {name!r}: choose from {choices}") from None


def parse_args() -> argparse.Namespace:
    """Read the command line: which scenario to run, for how long, and what to write."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help=f"scenario preset to run, by name (default: {DEFAULT_SCENARIO.name})",
    )
    add_session_arguments(parser, default_speed=DEFAULT_SPEED, default_hours=DEFAULT_HOURS)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory to write the run's metrics series and summary to when it ends",
    )
    args = parser.parse_args()
    validate_session_arguments(parser, args, default_hours=DEFAULT_HOURS)
    if args.replay is not None and args.scenario is not None:
        parser.error("--scenario cannot be combined with --replay: the recording holds it")
    return args


def export_run(
    label: str, scenario: str, seed: int, collector: MetricsCollector, out: Path
) -> None:
    """Write the run's metrics series and a one-row summary of it to ``out``."""
    summary = RunSummary.from_series(label, scenario, seed, collector.series())
    series_csv, series_json = summary.export(out)
    export_summary_csv([summary], out / "summary.csv")
    export_summary_json([summary], out / "summary.json")
    print(f"Wrote the summary and the series ({series_csv.name}, {series_json.name}) to {out}")


def main() -> None:
    """Run one seeded scenario in a window or without one, then write what was asked for."""
    args = parse_args()
    if args.replay is not None:
        log = CommandLog.from_json(args.replay.read_text(encoding="utf-8"))
        if log.scenario is None:
            raise SystemExit(f"{args.replay} was not recorded from a scenario preset")
        scenario = parse_scenario(log.scenario)
    else:
        scenario = DEFAULT_SCENARIO
        if args.scenario is not None:
            scenario = parse_scenario(args.scenario)
        seed = DEFAULT_SEED
        if args.seed is not None:
            seed = args.seed
        until = int(args.hours * TimeUnit.HOUR)
        log = CommandLog(seed=seed, until=until, commands=(), scenario=scenario.name)

    config = build_experiment(
        label=f"{scenario.name}/{log.seed}",
        scenario=scenario,
        seed=log.seed,
        until=log.until,
        metrics=MetricsConfig(period=METRICS_PERIOD_MS),
    )
    # Built without a renderer on purpose: the console draws on its own clock, and a renderer
    # handed to the run would draw on every metric tick as well.
    ctx, collector = build_run(config)
    console = InteractiveConsole(
        CommandReplay(ctx, log),
        MatplotlibRenderer(),
        label=config.label,
        horizon=config.until,
        replaying=args.replay is not None,
        speed=args.speed,
        collector=collector,
    )

    if args.save is None:
        show_window(console)
    else:
        frames = save_session(console, args.save)
        print(f"Wrote {frames} frames up to {format_time(console.time)} to {args.save}")
    if args.record is not None:
        write_recording(console, args.record)
        print(f"Recorded {len(console.recorded)} commands up to {format_time(console.time)}")
    if args.out is not None:
        export_run(config.label, scenario.name, log.seed, collector, args.out)


if __name__ == "__main__":
    main()
