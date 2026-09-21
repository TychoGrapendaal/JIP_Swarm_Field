"""Simple animated demonstration of the high-level field-cell model.

Run from the repository root with:
    uv run python examples/simple_world_sim.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from swarm_field.contracts import BodyType, FieldGeometry, Move, Treat
from swarm_field.simulation import RobotState, WorldState


@dataclass
class DemoController:
    targets: dict[int, tuple[int, int] | None] = field(default_factory=dict)

    def step(self, world: WorldState, time: int) -> None:
        world.advance_to(time)
        for robot_id, robot in world.robots.items():
            if robot.move_target is not None:
                continue

            if robot.body is BodyType.KILLER and world.cell(robot.position).infestation > 0:
                world.apply_action(robot_id, Treat(cell=robot.position), time=time)
                continue

            visible = world.observe_for(robot_id, time=time)
            infestations = [
                observation.cell
                for observation in visible
                if getattr(observation, "severity", 0.0) > 0
            ]
            if infestations:
                target = infestations[0]
            else:
                candidates = [
                    cell for cell in world.neighbours(robot.position) if world.is_walkable(cell)
                ]
                target = (
                candidates[(robot_id + time // 500) % len(candidates)]
                if candidates
                else None
            )

            if target is not None:
                world.apply_action(robot_id, Move(to=target), time=time)


def make_world() -> WorldState:
    field = FieldGeometry(
        width=20,
        height=12,
        cell_size=5.0,
        row_sp=2.0,
        in_row_sp=1.0,
        headland=5.0,
        connectivity=8,
    )
    robots = [
        RobotState(1, (2, 2), BodyType.SCOUT, speed_mps=2.0),
        RobotState(2, (2, 2), BodyType.KILLER, speed_mps=1.0),
        RobotState(3, (2, 2), BodyType.BATTERY, speed_mps=0.75),
    ]
    world = WorldState(field, robots=robots)

    for y in range(2, 10):
        for x in (8, 9, 10):
            world.cell((x, y)).obstacle = True

    for cell in ((15, 9), (16, 8), (14, 8)):
        world.cell(cell).infestation = 1.0

    return world


def main() -> None:
    world = make_world()
    controller = DemoController()
    current_time = 0

    fig, ax = plt.subplots(figsize=(11, 7))

    def update(frame: int) -> None:
        nonlocal current_time
        current_time += 200
        controller.step(world, current_time)

        ax.clear()
        grid = [
            [0 if not world.cell((x, y)).obstacle else 2 for x in range(world.field.width)]
            for y in range(world.field.height)
        ]
        ax.imshow(
            grid,
            origin="lower",
            interpolation="none",
            extent=(0, world.field.width, 0, world.field.height),
            cmap="Greens",
            vmin=0,
            vmax=2,
            alpha=0.35,
        )

        for cell, state in world.cells.items():
            if state.infestation <= 0:
                continue
            x, y = cell
            ax.scatter(x + 0.5, y + 0.5, marker="s", s=250, color="tomato", alpha=0.8)

        body_color = {
            BodyType.SCOUT: "royalblue",
            BodyType.KILLER: "crimson",
            BodyType.BATTERY: "goldenrod",
        }
        for body in BodyType:
            points = []
            for robot_id, robot in world.robots.items():
                if robot.body is body:
                    points.append(world.continuous_position(robot_id, current_time))
            if points:
                ax.scatter(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    s=55,
                    label=body.name,
                    color=body_color[body],
                    edgecolors="black",
                    linewidths=0.4,
                )

        ax.set_aspect("equal")
        ax.set_xlim(0, world.field.width)
        ax.set_ylim(0, world.field.height)
        ax.set_xlabel("Field cell X")
        ax.set_ylabel("Field cell Y")
        ax.set_title("High-level WorldState: direct-cell movement")
        ax.legend(loc="upper right")
        ax.text(
            0.01,
            1.01,
            f"time={current_time / 1000:.1f}s | robots={len(world.robots)} | "
            f"cell={world.field.cell_size:g} m",
            transform=ax.transAxes,
        )

    animation = FuncAnimation(fig, update, frames=300, interval=60, repeat=False)
    _ = animation
    plt.show()


if __name__ == "__main__":
    main()
