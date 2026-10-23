"""A swarm exploring and monitoring a field with the EXPLORE seeder, on the interactive console.

Five scouts start side by side in a corner of a field of 10 m cells. Each is a real
``ComposedAgent`` following the EXPLORE field (``ExploreSeeder``), assembled and run by
``swarm_field.runner`` from an ``ExploreScenario``.

The overlay shades what each cell is worth to the EXPLORE rule in ground truth, from the age
since any robot last stood on it: dark is fresh (worth nothing until the revisit age), bright is
stale. Trails show where the robots went, dashed lines the target each robot is heading for, and
the panel the measures of ``MetricsCollector``. The console's controls work as in
``scenario_viewer.py``: pause, step, change the speed, inject a pest, inspect a robot.

Run from the repository root with:
    uv run python examples/explore_demo.py

Record a video instead (mp4 through ffmpeg, or a gif):
    uv run python examples/explore_demo.py --save explore.mp4

Options: ``--no-discount`` turns off the peer discount, ``--block-side 16`` uses 16 x 16 blocks,
``--obstacles`` adds a river with one bridge and two rocks.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from swarm_field.console import (
    InteractiveConsole,
    add_session_arguments,
    format_time,
    save_session,
    show_window,
)
from swarm_field.contracts import CellId, Time, TimeUnit, WorldSnapshot
from swarm_field.experiments import (
    EXPLORE,
    ExplorerPreset,
    LiveRun,
    MapConfig,
    SwarmScenario,
    river_and_rocks,
)
from swarm_field.simulation import MetricsConfig
from swarm_field.swarm import ExploreConfig
from swarm_field.swarm.explore import revisit_worths
from swarm_field.visualization import MatplotlibRenderer

DEFAULT_SPEED = 40.0
"""Twenty simulated seconds per frame."""
TRAIL_LENGTH = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=32, help="field side in cells")
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--no-discount", action="store_true", help="turn the peer discount off")
    parser.add_argument("--block-side", type=int, default=8, help="block side in cells")
    parser.add_argument("--obstacles", action="store_true", help="a river and two rocks")
    add_session_arguments(parser, default_speed=DEFAULT_SPEED, recording=False)
    return parser.parse_args()


def build_scenario(args: argparse.Namespace, config: ExploreConfig) -> SwarmScenario:
    """The scenario of the command line; idleness is measured over the second half of the run."""
    obstacles: tuple[CellId, ...] = ()
    if args.obstacles:
        obstacles = river_and_rocks(args.size)
    scenario = (
        EXPLORE.without_robots()
        .with_map(MapConfig(size=args.size, obstacles=obstacles))
        .add_robots(ExplorerPreset(explore=config), args.robots)
    )
    if args.seed is not None:
        scenario = replace(scenario, seed=args.seed)
    if args.hours is not None:
        scenario = replace(scenario, until=int(args.hours * TimeUnit.HOUR))
    metrics = MetricsConfig(period=20 * TimeUnit.SECOND, measure_from=scenario.until // 2)
    return scenario.with_metrics(metrics)


def main() -> None:
    args = parse_args()
    config = ExploreConfig(spread_peers=not args.no_discount, block_side=args.block_side)
    run = LiveRun.from_config(build_scenario(args, config).into_experiment())
    monitorable = np.ones((args.size, args.size), dtype=np.bool_)

    def staleness(snapshot: WorldSnapshot) -> NDArray[np.float64]:
        visits = np.full((args.size, args.size), -np.inf)
        for cell, sensed_at in run.metrics.last_sensed_times().items():
            visits[cell] = sensed_at
        values = revisit_worths(
            np.maximum(visits, 0.0), monitorable, snapshot.time, config.revisit_age
        )
        for cell, state in snapshot.cells.items():
            if state.obstacle:
                values[cell] = np.nan
        return values

    def panel(snapshot: WorldSnapshot) -> list[str]:
        return panel_lines(run, snapshot.time, config)

    console = InteractiveConsole(
        run,
        MatplotlibRenderer((12.0, 8.0), show_targets=False),
        speed=args.speed,
        overlay=staleness,
        overlay_label="worth of a visit, 0 until the revisit age",
        panel=panel,
        trail_length=TRAIL_LENGTH,
        show_targets=True,
    )
    if args.save is None:
        show_window(console)
        return
    frames = save_session(console, args.save)
    print(f"Wrote {frames} frames up to {format_time(run.now)} to {args.save}")


def panel_lines(run: LiveRun, now: Time, config: ExploreConfig) -> list[str]:
    """The run's settings and its measures so far."""
    discount = "off"
    if config.spread_peers:
        discount = "on"
    lines = [
        f"time         {now / TimeUnit.HOUR:5.2f} h",
        f"robots       {len(run.agents())}",
        f"blocks       {config.block_side} x {config.block_side}",
        f"discount     {discount}",
        f"revisit age  {config.revisit_age / TimeUnit.MINUTE:.0f} min",
        "",
    ]
    latest = run.metrics.latest()
    if latest is None:
        return lines
    lines.extend(
        [
            f"covered      {latest.coverage:6.1%}",
            f"90 % at      {_minutes(_time_to_cover(run, 0.9))}",
            f"100 % at     {_minutes(_time_to_cover(run, 1.0))}",
            f"redundancy   {latest.redundancy:6.1%}",
            f"idleness     {latest.mean_idleness_ms / TimeUnit.MINUTE:5.1f} min (mean)",
            f"             {latest.worst_idleness_ms / TimeUnit.MINUTE:5.1f} min (worst)",
            "  measured from half the run on",
        ]
    )
    return lines


def _time_to_cover(run: LiveRun, share: float) -> Time | None:
    """The first sample at which ``share`` of the field had been sensed, or ``None``."""
    for sample in run.metrics.series():
        if sample.coverage >= share:
            return sample.time
    return None


def _minutes(time: Time | None) -> str:
    if time is None:
        return "-"
    return f"{time / TimeUnit.MINUTE:5.1f} min"


if __name__ == "__main__":
    main()
