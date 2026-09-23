"""Animated demonstration of the discrete-time pest dynamics.

Run from the repository root with:
    uv run --with matplotlib python examples/pest_dynamics_demo.py

Headless frame generation:
    uv run --with matplotlib python examples/pest_dynamics_demo.py --frames 80 --save frames/pests
"""

from __future__ import annotations

import argparse
from pathlib import Path
from random import Random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from swarm_field.contracts import BodyType, FieldGeometry
from swarm_field.simulation import (
    PestDynamicsConfig,
    RobotState,
    SimulationContext,
    SimulationLoop,
    WorldState,
    WorldTick,
)
from swarm_field.visualization import MatplotlibRenderer

FIELD_WIDTH = 24
FIELD_HEIGHT = 16
CELL_SIZE_M = 2.0
FRAME_TIME = 1_000


def build_context(seed: int) -> SimulationContext:
    """Build a reproducible field with an initial pest outbreak."""
    field = FieldGeometry(
        width=FIELD_WIDTH,
        height=FIELD_HEIGHT,
        cell_size=CELL_SIZE_M,
        row_sp=3.0,
        in_row_sp=1.0,
        headland=2.0,
        connectivity=4,
    )
    robots = [
        RobotState(
            robot_id=0,
            position=(2, 2),
            body=BodyType.SCOUT,
            speed_mps=1.0,
        ),
    ]
    world = WorldState(field, robots=robots)

    # Start with one coarse-grained outbreak.
    for cell, severity in {
        (11, 7): 0.9,
        (12, 7): 0.6,
        (11, 8): 0.5,
    }.items():
        world.cell(cell).infestation = severity

    # Add a small obstacle block so the spread boundary is visible.
    for cell in ((16, 4), (16, 5), (17, 4), (17, 5)):
        world.cell(cell).obstacle = True

    ctx = SimulationContext(
        now=0,
        rng=Random(seed),
        agents={},
        world=world,
    )
    ctx.schedule(
        WorldTick(
            time=0,
            config=PestDynamicsConfig(
                period=FRAME_TIME,
                spawn_probability=0.0008,
                growth_rate=0.03,
                spread_probability=0.28,
                spawn_severity=0.10,
            ),
        )
    )
    return ctx


def render_frame(
    renderer: MatplotlibRenderer,
    ctx: SimulationContext,
    time: int,
) -> None:
    """Advance the event simulation and render the ground-truth world."""
    SimulationLoop().run(ctx, until=time)
    renderer.render(ctx.world.snapshot(time))

    infested = sum(state.infestation > 0.0 for state in ctx.world.cells.values())
    mean_severity = sum(state.infestation for state in ctx.world.cells.values()) / len(
        ctx.world.cells
    )
    renderer.axes.set_title(
        f"Pest dynamics | t={time} | infested cells={infested} | mean severity={mean_severity:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120, help="number of frames")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="directory to save PNG frames instead of opening a window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = build_context(args.seed)
    renderer = MatplotlibRenderer(figsize=(10, 7), show_targets=False)

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for frame in range(args.frames):
            time = frame * FRAME_TIME
            render_frame(renderer, ctx, time)
            renderer.figure.savefig(args.save / f"frame_{frame:04d}.png", dpi=120)
        plt.close(renderer.figure)
        print(f"Wrote {args.frames} frames to {args.save}")
        return

    def update(frame: int) -> None:
        render_frame(renderer, ctx, frame * FRAME_TIME)

    animation = FuncAnimation(
        renderer.figure,
        update,
        frames=args.frames,
        interval=100,
        repeat=False,
    )
    _ = animation
    plt.show()


if __name__ == "__main__":
    main()
