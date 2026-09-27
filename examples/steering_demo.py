"""Animated demo of the A.3 ``Steerer`` walking a land robot around an obstacle wall.

A utility (KILLER) robot starts east of a vertical crop-wall; an infested cell lies west of
it. The wall's only gap is at the field's top edge, so the Steerer's committed right-hand
wall-following climbs the wall's east face up to that gap, rounds it, and descends to the
infested cell — then the demo treats it. Each decision point yields one direct
``Move(to=CellId)`` applied by ``WorldState``; the ``Sensing`` probe reads ground-truth
walkability as a stand-in for future perception. This is the reactive layer the ``Navigator``
falls back on when a potential field cannot guide a robot; the potential field itself is
demonstrated in ``potential_field_demo.py``.

Run from the repository root with:
    uv run python examples/steering_demo.py

Headless (write frames as PNGs instead of opening a window):
    uv run python examples/steering_demo.py --frames 300 --save frames/steering
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from swarm_field.contracts import (
    BodyType,
    CellId,
    Direction,
    FieldGeometry,
    Idle,
    Move,
    Treat,
)
from swarm_field.simulation import RobotState, WorldState
from swarm_field.swarm import Steerer
from swarm_field.visualization import MatplotlibRenderer

FIELD_WIDTH = 12
FIELD_HEIGHT = 9
CELL_SIZE_M = 2.0
TICK_MS = 200

ROBOT_ID = 0
START = (10, 3)
GOAL = (2, 3)
WALL_COLUMN = 6
WALL_BOTTOM = 0
WALL_TOP = 7


class WorldSensing:
    """Per-heading walkability probe backed by the world's ground truth.

    The probe answers the Steerer's ``clear(origin, direction)`` question straight from
    ``WorldState`` — the same ground-truth-backed stand-in the steering tests use, where
    ``QuadtreeSensing`` would answer it from believed terrain in a real brain. A real
    perception layer is future work; the Steerer itself stays position-agnostic about the field
    layout.
    """

    def __init__(self, world: WorldState) -> None:
        self._world: WorldState = world

    def clear(self, origin: CellId, direction: Direction) -> bool:
        """Whether the neighbour one ``direction`` step from ``origin`` is walkable."""
        dx, dy = direction.offset
        return self._world.is_walkable((origin[0] + dx, origin[1] + dy))


class SteeringController:
    """One land robot driven by the A.3 Steerer toward an absolute goal.

    Makes exactly one decision per parked cell: relative goal and current cell go in, a direct
    ``Move`` comes out and is applied to the world. The visited trace and arrival/treatment
    bookkeeping live here so the demo can draw them.
    """

    def __init__(self, world: WorldState, steerer: Steerer) -> None:
        self.world: WorldState = world
        self.steerer: Steerer = steerer
        self.visited: list[CellId] = [world.robots[ROBOT_ID].position]
        self.arrived: bool = False

    @property
    def robot(self) -> RobotState:
        """The demo's single ground-truth robot."""
        return self.world.robots[ROBOT_ID]

    def step(self, time: int) -> None:
        """Advance the world to ``time`` and act if the robot is parked and not done."""
        self.world.advance_to(time)
        if self.robot.move_target is not None:
            return

        if self.robot.position == GOAL and self.world.cell(GOAL).infestation > 0:
            self.world.apply_action(ROBOT_ID, Treat(cell=GOAL), time=time)
            self.arrived = True
            return

        dx = float(GOAL[0] - self.robot.position[0])
        dy = float(GOAL[1] - self.robot.position[1])
        action = self.steerer.steer((dx, dy), self.robot.position)

        if isinstance(action, Idle):
            self.arrived = True
            return
        if isinstance(action, Move):
            self.world.apply_action(ROBOT_ID, action, time=time)
            self.visited.append(action.to)


def build_world() -> WorldState:
    """A dressed-up field: growing crop everywhere, one crop-wall, one infested goal cell."""
    field = FieldGeometry(
        width=FIELD_WIDTH,
        height=FIELD_HEIGHT,
        cell_size=CELL_SIZE_M,
        row_sp=3.0,
        in_row_sp=1.0,
        headland=2.0,
        connectivity=8,
    )
    world = WorldState(
        field,
        robots=[
            RobotState(
                robot_id=ROBOT_ID,
                position=START,
                body=BodyType.KILLER,
                speed_mps=2.0,
            )
        ],
    )

    for y in range(FIELD_HEIGHT):
        for x in range(FIELD_WIDTH):
            world.cell((x, y)).growth_stage = 0.7

    for y in range(WALL_BOTTOM, WALL_TOP + 1):
        world.cell((WALL_COLUMN, y)).obstacle = True

    world.cell(GOAL).infestation = 1.0
    return world


def draw_overlay(renderer: MatplotlibRenderer, controller: SteeringController, time: int) -> None:
    """Draw the visited trace and goal marker on top of the rendered snapshot."""
    trace_x = [cell[0] + 0.5 for cell in controller.visited]
    trace_y = [cell[1] + 0.5 for cell in controller.visited]
    renderer.axes.plot(trace_x, trace_y, color="tab:blue", linewidth=1.2, alpha=0.85)
    renderer.axes.scatter(
        [GOAL[0] + 0.5],
        [GOAL[1] + 0.5],
        marker="*",
        s=220,
        color="gold",
        edgecolors="black",
        linewidths=0.6,
    )
    status = "reached & treated" if controller.arrived else f"{len(controller.visited)} cells"
    renderer.axes.set_title(
        f"Steering demo | t={time // 1000}.{time % 1000 // 100}s | {status} | "
        f"stalled={controller.steerer.is_stalled}"
    )
    renderer.axes.scatter([], [], color="tab:blue", label="trace")
    renderer.axes.legend(loc="lower left", markerscale=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=500, help="number of frames")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="directory to write PNG frames into instead of showing a window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world = build_world()
    controller = SteeringController(world, Steerer(WorldSensing(world)))
    renderer = MatplotlibRenderer(figsize=(10, 7), show_targets=False)

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for frame in range(args.frames):
            time = frame * TICK_MS
            controller.step(time)
            renderer.render(world.snapshot(time))
            draw_overlay(renderer, controller, time)
            renderer.figure.savefig(args.save / f"frame_{frame:04d}.png", dpi=120)
            if controller.arrived:
                break
        reached = world.cell(GOAL).infestation == 0.0
        print(
            f"Robot reached the goal cell in {(frame + 1) * TICK_MS / 1000:.1f}s after "
            f"{len(controller.visited)} moves; treated={reached}"
        )
        return

    def update(frame: int) -> None:
        time = frame * TICK_MS
        controller.step(time)
        renderer.render(world.snapshot(time))
        draw_overlay(renderer, controller, time)

    animation = FuncAnimation(
        renderer.figure, update, frames=args.frames, interval=40, repeat=False
    )
    _ = animation
    plt.show()


if __name__ == "__main__":
    main()
