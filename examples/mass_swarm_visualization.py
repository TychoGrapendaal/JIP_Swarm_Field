"""Hundreds-of-robots visual proof of concept for the high-level swarm model.

The demo keeps the environment coarse: one cell is one field patch, a river can occupy many
cells, robots may share a cell, and movement is a direct target-cell command. The potential-field
controller uses local obstacle and infestation observations to choose among candidate cells.

Run:
    uv run python examples/mass_swarm_visualization.py

Optional:
    uv run python examples/mass_swarm_visualization.py --robots 500 --seed 7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from random import Random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from swarm_field.contracts import BodyType, FieldGeometry, Move, Treat
from swarm_field.simulation import RobotState, WorldState

FIELD_WIDTH = 50
FIELD_HEIGHT = 35
CELL_SIZE_M = 5.0
FRAME_TIME_MS = 200


@dataclass
class LocalPotentialMemory:
    """Per-robot local potential information."""

    obstacle_repulsion: dict[tuple[int, int], float] = field(default_factory=dict)
    infestation_attraction: dict[tuple[int, int], float] = field(default_factory=dict)


class PotentialFieldController:
    """Very small local-rule controller used only by the demonstration."""

    def __init__(self, world: WorldState, rng: Random) -> None:
        self.world = world
        self.rng = rng
        self.memory: dict[int, LocalPotentialMemory] = {
            robot_id: LocalPotentialMemory() for robot_id in world.robots
        }

    def step(self, time: int) -> None:
        self.world.advance_to(time)
        for robot_id, robot in self.world.robots.items():
            if robot.move_target is not None:
                continue

            observations = self.world.observe_for(robot_id, time)
            memory = self.memory[robot_id]
            self._update_memory(memory, observations)

            if robot.body is BodyType.KILLER and self.world.cell(robot.position).infestation > 0:
                self.world.apply_action(robot_id, Treat(cell=robot.position), time=time)
                continue

            target = self._choose_target(robot_id, memory)
            if target is not None:
                self.world.apply_action(robot_id, Move(to=target), time=time)

    @staticmethod
    def _update_memory(memory: LocalPotentialMemory, observations: list[object]) -> None:
        for observation in observations:
            cell = getattr(observation, "cell", None)
            if cell is None:
                continue
            if hasattr(observation, "is_obstacle") and observation.is_obstacle:
                memory.obstacle_repulsion[cell] = 100.0
            if hasattr(observation, "severity"):
                severity = float(observation.severity)
                if severity > 0:
                    memory.infestation_attraction[cell] = max(
                        severity,
                        memory.infestation_attraction.get(cell, 0.0),
                    )

    def _choose_target(
        self,
        robot_id: int,
        memory: LocalPotentialMemory,
    ) -> tuple[int, int] | None:
        robot = self.world.robots[robot_id]
        candidates = [
            cell for cell in self.world.neighbours(robot.position) if self.world.is_walkable(cell)
        ]
        if not candidates:
            return None

        best_score = float("inf")
        best_cells: list[tuple[int, int]] = []
        attraction_weight = {
            BodyType.SCOUT: 0.7,
            BodyType.KILLER: 1.5,
            BodyType.BATTERY: 0.25,
        }[robot.body]

        for cell in candidates:
            attraction = 0.0
            for target, severity in memory.infestation_attraction.items():
                distance = max(1.0, self._cell_distance(cell, target))
                attraction += attraction_weight * severity / distance

            repulsion = 0.0
            for obstacle, strength in memory.obstacle_repulsion.items():
                distance = self._cell_distance(cell, obstacle)
                repulsion += strength / max(1.0, distance * distance)

            # A small exploration term prevents identical robots from making exactly the same
            # choice when all nearby potentials are equal.
            exploration = self.rng.random() * 0.15
            score = repulsion - attraction + exploration

            if score < best_score - 1e-9:
                best_score = score
                best_cells = [cell]
            elif abs(score - best_score) <= 1e-9:
                best_cells.append(cell)

        return self.rng.choice(best_cells)

    @staticmethod
    def _cell_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5


def build_world(robot_count: int, seed: int) -> WorldState:
    rng = Random(seed)
    field = FieldGeometry(
        width=FIELD_WIDTH,
        height=FIELD_HEIGHT,
        cell_size=CELL_SIZE_M,
        row_sp=15.0,
        in_row_sp=1.5,
        headland=10.0,
        connectivity=8,
    )

    robots: list[RobotState] = []
    speed_by_body = {
        BodyType.SCOUT: 2.0,
        BodyType.KILLER: 1.2,
        BodyType.BATTERY: 0.8,
    }
    bodies = [
        BodyType.SCOUT,
        BodyType.SCOUT,
        BodyType.SCOUT,
        BodyType.KILLER,
        BodyType.BATTERY,
    ]

    for robot_id in range(robot_count):
        body = bodies[robot_id % len(bodies)]
        while True:
            position = (rng.randrange(FIELD_WIDTH), rng.randrange(FIELD_HEIGHT))
            if position[0] not in (24, 25, 26) or not 5 <= position[1] <= 29:
                break
        robots.append(
            RobotState(
                robot_id=robot_id,
                position=position,
                body=body,
                speed_mps=speed_by_body[body],
            )
        )

    world = WorldState(field, robots=robots)

    # A river occupying three whole cells across the middle of the field.
    for x in (24, 25, 26):
        for y in range(5, 30):
            world.cell((x, y)).obstacle = True

    # A few infestation patches on the two sides of the river.
    for cell, severity in {
        (11, 8): 1.0,
        (16, 13): 0.8,
        (36, 10): 1.0,
        (41, 22): 0.9,
        (33, 27): 0.7,
    }.items():
        world.cell(cell).infestation = severity

    return world


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", type=int, default=300, help="number of robots")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world = build_world(args.robots, args.seed)
    controller = PotentialFieldController(world, Random(args.seed + 1))
    current_time = 0

    fig, ax = plt.subplots(figsize=(13, 9))
    cmap = ListedColormap(["#dcefd2", "#ef6c5b", "#6d6d6d"])

    def update(_frame: int) -> None:
        nonlocal current_time
        current_time += FRAME_TIME_MS
        controller.step(current_time)

        grid: list[list[int]] = []
        for y in range(world.field.height):
            row: list[int] = []
            for x in range(world.field.width):
                state = world.cell((x, y))
                if state.obstacle:
                    row.append(2)
                elif state.infestation > 0:
                    row.append(1)
                else:
                    row.append(0)
            grid.append(row)

        ax.clear()
        ax.imshow(
            grid,
            origin="lower",
            interpolation="none",
            extent=(0, world.field.width, 0, world.field.height),
            cmap=cmap,
            vmin=0,
            vmax=2,
        )

        body_colors = {
            BodyType.SCOUT: "royalblue",
            BodyType.KILLER: "crimson",
            BodyType.BATTERY: "goldenrod",
        }
        for body in BodyType:
            points = [
                world.continuous_position(robot_id, current_time)
                for robot_id, robot in world.robots.items()
                if robot.body is body
            ]
            if not points:
                continue
            ax.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                s=13,
                alpha=0.75,
                color=body_colors[body],
                label=f"{body.name} ({len(points)})",
                linewidths=0,
            )

        ax.set_aspect("equal")
        ax.set_xlim(0, FIELD_WIDTH)
        ax.set_ylim(0, FIELD_HEIGHT)
        ax.set_xlabel("Field cell X")
        ax.set_ylabel("Field cell Y")
        ax.set_xticks(range(0, FIELD_WIDTH + 1, 5))
        ax.set_yticks(range(0, FIELD_HEIGHT + 1, 5))
        ax.grid(alpha=0.15, linewidth=0.5)
        ax.set_title(
            f"High-level swarm demo | {len(world.robots)} robots | "
            f"{CELL_SIZE_M:g} m cells | t={current_time / 1000:.1f} s"
        )
        ax.legend(loc="upper right", markerscale=2)

    animation = FuncAnimation(fig, update, frames=600, interval=40, repeat=False)
    _ = animation
    plt.show()


if __name__ == "__main__":
    main()
