"""Running-scenario demonstration of the matplotlib ``Renderer``.

Builds a fully cropped field with an all-scout roster, steps a local rule-based controller, and
draws each ``WorldState.snapshot`` through the ``Renderer`` port.

Run:
    uv run python examples/renderer_demo.py

Headless (write frames as PNGs instead of opening a window):
    uv run python examples/renderer_demo.py --frames 60 --save frames
"""

from __future__ import annotations

import argparse
from pathlib import Path
from random import Random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from swarm_field.contracts import BodyType, FieldGeometry, Move
from swarm_field.simulation import RobotState, WorldState
from swarm_field.visualization import MatplotlibRenderer, Renderer

FIELD_WIDTH = 30
FIELD_HEIGHT = 20
CELL_SIZE_M = 5.0
FRAME_TIME_MS = 200
SPEED_BY_BODY = {
    BodyType.SCOUT: 2.0,
    BodyType.KILLER: 1.2,
    BodyType.BATTERY: 0.8,
}


class RandomWalkController:
    """A tiny local controller: every idle robot wanders to a random walkable neighbour."""

    def __init__(self, world: WorldState, rng: Random) -> None:
        self.world = world
        self.rng = rng

    def step(self, time: int) -> None:
        """Advance the world to ``time`` and move every idle robot."""
        self.world.advance_to(time)
        for robot_id, robot in self.world.robots.items():
            if robot.move_target is not None:
                continue
            walkable = [
                neighbour
                for neighbour in self.world.neighbours(robot.position)
                if self.world.is_walkable(neighbour)
            ]
            if walkable:
                self.world.apply_action(robot_id, Move(to=self.rng.choice(walkable)), time=time)


def build_world(seed: int) -> WorldState:
    """Create a fully cropped field and an all-scout robot roster."""
    field = FieldGeometry(
        width=FIELD_WIDTH,
        height=FIELD_HEIGHT,
        cell_size=CELL_SIZE_M,
        row_sp=15.0,
        in_row_sp=1.5,
        headland=10.0,
        connectivity=8,
    )
    bodies = [BodyType.SCOUT] * 4
    robots = [
        RobotState(
            robot_id=robot_id,
            position=(1, 1 + robot_id),
            body=body,
            speed_mps=SPEED_BY_BODY[body],
        )
        for robot_id, body in enumerate(bodies)
    ]
    world = WorldState(field, robots=robots)

    for y in range(FIELD_HEIGHT):
        for x in range(FIELD_WIDTH):
            world.cell((x, y)).growth_stage = 0.3 + 0.6 * ((x + y) % 4) / 3

    return world


def draw(renderer: Renderer, world: WorldState, time: int) -> None:
    """Draw one frame through the ``Renderer`` port (no ground truth crosses this call)."""
    renderer.render(world.snapshot(time))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300, help="number of frames")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="directory to write PNG frames into instead of showing a window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world = build_world(args.seed)
    controller = RandomWalkController(world, Random(args.seed + 1))
    renderer = MatplotlibRenderer()

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for index in range(args.frames):
            time = index * FRAME_TIME_MS
            controller.step(time)
            draw(renderer, world, time)
            renderer.figure.savefig(args.save / f"frame_{index:04d}.png")
        print(f"Wrote {args.frames} frames to {args.save}")
        return

    frame_time = {"value": 0}

    def update(_frame: int) -> None:
        frame_time["value"] += FRAME_TIME_MS
        controller.step(frame_time["value"])
        draw(renderer, world, frame_time["value"])

    animation = FuncAnimation(
        renderer.figure, update, frames=args.frames, interval=40, repeat=False
    )
    _ = animation
    plt.show()


if __name__ == "__main__":
    main()
