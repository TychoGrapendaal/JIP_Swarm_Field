"""A swarm exploring and monitoring a field with the EXPLORE seeder, drawn by ``SimulationViewer``.

Five scouts start side by side in a corner of a field of 10 m cells. Each is a real
``ComposedAgent`` following the EXPLORE field (``ExploreSeeder``) on the real simulation loop;
see ``swarm_field.experiments`` for how the run is assembled.

The overlay shades what each cell is worth to the EXPLORE rule in ground truth, from the age
since any robot last stood on it: dark is fresh (worth nothing until the revisit age), bright is
stale. Trails show where the robots
went, dashed lines the target each robot is heading for, and the panel the coverage measures of
``CoverageTracker``.

Run from the repository root with:
    uv run python examples/explore_demo.py

Record a video instead (mp4 through ffmpeg, or a gif):
    uv run python examples/explore_demo.py --save explore.mp4

Options: ``--no-discount`` turns off the peer discount, ``--blocks 16`` uses 16 x 16 blocks and
``--blocks 8 16`` makes them grow with the distance, ``--obstacles`` adds a river with one bridge
and two rocks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import CellId, RobotId, Time, TimeUnit, WorldSnapshot
from swarm_field.experiments import (
    ExploreRun,
    ExploreScenario,
    build_explore_run,
    river_and_rocks,
)
from swarm_field.swarm import ExploreConfig, ExploreSeeder, FieldSeeder
from swarm_field.swarm.explore import cell_values
from swarm_field.visualization import SimulationViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=32, help="field side in cells")
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--hours", type=float, default=2.0, help="simulated time")
    parser.add_argument("--frame-seconds", type=float, default=20.0, help="simulated per frame")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no-discount", action="store_true", help="turn the peer discount off")
    parser.add_argument("--blocks", type=int, nargs="+", default=[8], help="block sides")
    parser.add_argument("--obstacles", action="store_true", help="a river and two rocks")
    parser.add_argument("--save", type=Path, help="write a .mp4 or .gif instead of a window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExploreConfig(spread_peers=not args.no_discount, block_sides=tuple(args.blocks))
    obstacles: tuple[CellId, ...] = ()
    if args.obstacles:
        obstacles = river_and_rocks(args.size)
    scenario = ExploreScenario(
        width=args.size,
        height=args.size,
        robots=args.robots,
        obstacles=obstacles,
        measure_from=int(args.hours * TimeUnit.HOUR) // 2,
    )

    def seeder_for(robot: RobotId) -> FieldSeeder:
        return ExploreSeeder(config)

    run = build_explore_run(scenario, seeder_for)
    monitorable = np.ones((args.size, args.size), dtype=np.bool_)

    def staleness(snapshot: WorldSnapshot) -> NDArray[np.float64]:
        visits = np.maximum(run.tracker.last_visits(), 0.0)
        values = cell_values(visits, monitorable, snapshot.time, config.revisit_age)
        for cell, state in snapshot.cells.items():
            if state.obstacle:
                values[cell] = np.nan
        return values

    def panel(snapshot: WorldSnapshot) -> list[str]:
        return panel_lines(run, snapshot.time, config)

    viewer = SimulationViewer(
        run.snapshot_at,
        overlay=staleness,
        overlay_label="worth of a visit, 0 until the revisit age",
        goals=run.targets,
        panel=panel,
    )
    step = int(args.frame_seconds * TimeUnit.SECOND)
    frames = list(range(0, int(args.hours * TimeUnit.HOUR) + 1, step))
    viewer.animate(frames, fps=args.fps, save=args.save)


def panel_lines(run: ExploreRun, now: Time, config: ExploreConfig) -> list[str]:
    """The run's settings and its coverage measures so far."""
    summary = run.summary()
    discount = "off"
    if config.spread_peers:
        discount = "on"
    return [
        f"time         {now / TimeUnit.HOUR:5.2f} h",
        f"robots       {len(run.agents)}",
        f"blocks       {config.block_sides}",
        f"discount     {discount}",
        f"revisit age  {config.revisit_age / TimeUnit.MINUTE:.0f} min",
        "",
        f"covered      {summary.covered:6.1%}",
        f"90 % at      {_minutes(summary.time_to_90)}",
        f"100 % at     {_minutes(summary.time_to_full)}",
        f"redundancy   {summary.redundancy:6.1%}",
        f"idleness     {summary.mean_idleness / TimeUnit.MINUTE:5.1f} min (mean)",
        f"             {summary.worst_idleness / TimeUnit.MINUTE:5.1f} min (worst)",
        "  measured from half the run on",
    ]


def _minutes(time: Time | None) -> str:
    if time is None:
        return "-"
    return f"{time / TimeUnit.MINUTE:5.1f} min"


if __name__ == "__main__":
    main()
