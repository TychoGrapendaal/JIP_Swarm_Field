"""Interactive swarm-field console (C.1).

A running scenario you can steer from the window: robots act through the real B loop
(``RobotDecision`` events drive ``SwarmAgent.decide``), pests spread via ``WorldTick``, and
the frontend-injected interactions go through the typed ``Command`` port.

- Click an empty cell to inject an infestation (``InjectInfestation`` → ``apply_command``).
- Click a robot to inspect it (``SwarmAgent.inspect()`` → ``AgentInspection`` panel).

Time controls (``Pause`` / ``StepOnce`` / ``SetTimeScale`` commands) pace the render loop:

- space / p    pause / resume
- s            step once: advance by exactly the next simulation event
- + / =        double the simulation speed
- -            halve the simulation speed
- 0            reset to 1x

Run:
    uv run python examples/interactive_demo.py

Headless (write frames as PNGs instead of opening a window):
    uv run python examples/interactive_demo.py --frames 240 --save frames/interactive
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from random import Random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from swarm_field.contracts import (
    BodyType,
    Command,
    FieldGeometry,
    InjectInfestation,
    Pause,
    RobotId,
    SetTimeScale,
    StepOnce,
)
from swarm_field.simulation import (
    PestDynamicsConfig,
    RobotDecision,
    RobotState,
    SimulationContext,
    SimulationLoop,
    WorldState,
    WorldTick,
)
from swarm_field.swarm import ComposedAgent, Explore
from swarm_field.visualization import MatplotlibRenderer

FIELD_WIDTH = 24
FIELD_HEIGHT = 16
CELL_SIZE_M = 2.0
BASE_STEP_MS = 500
WORLD_TICK_PERIOD = 3_000
INJECT_SEVERITY = 0.8
SPEED_BY_BODY = {
    BodyType.SCOUT: 2.0,
    BodyType.KILLER: 1.2,
    BodyType.BATTERY: 0.8,
}


def build_context(seed: int) -> SimulationContext:
    """Build a seeded field with an initial pest patch, obstacles and one agent per robot."""
    field = FieldGeometry(
        width=FIELD_WIDTH,
        height=FIELD_HEIGHT,
        cell_size=CELL_SIZE_M,
        row_sp=3.0,
        in_row_sp=1.0,
        headland=2.0,
        connectivity=4,
    )
    bodies = [BodyType.SCOUT, BodyType.KILLER, BodyType.BATTERY]
    robots = [
        RobotState(
            robot_id=robot_id,
            position=start,
            body=body,
            speed_mps=SPEED_BY_BODY[body],
        )
        for robot_id, (body, start) in enumerate(zip(bodies, ((2, 2), (3, 3), (2, 4)), strict=True))
    ]
    world = WorldState(field, robots=robots)

    for cell, severity in {
        (14, 6): 0.6,
        (15, 7): 0.5,
        (18, 10): 0.4,
    }.items():
        world.cell(cell).infestation = severity

    for cell in ((20, 5), (20, 6), (20, 7), (21, 5), (21, 6), (21, 7)):
        world.cell(cell).obstacle = True

    rng = Random(seed)
    ctx = SimulationContext(
        now=0,
        rng=rng,
        agents={
            robot_id: ComposedAgent(robot_id=robot_id, behaviors=[Explore(rng)])
            for robot_id in range(len(robots))
        },
        world=world,
    )
    for robot_id in range(len(robots)):
        ctx.schedule(RobotDecision(time=0, robot=robot_id))
    ctx.schedule(
        WorldTick(
            time=0,
            config=PestDynamicsConfig(
                period=WORLD_TICK_PERIOD,
                spawn_probability=0.008,
                growth_rate=0.03,
                spread_probability=0.28,
                spawn_severity=0.10,
            ),
        )
    )
    return ctx


def apply_time_command(
    command: Command, paused: bool, speed: float, step_pending: bool
) -> tuple[bool, float, bool]:
    """Translate a time-pacing command into render-loop state (paused, speed, step).

    ``StepOnce`` steps exactly once and then freezes: after the single event the run stays
    paused until the user resumes.
    """
    if isinstance(command, Pause):
        return not paused, speed, False
    if isinstance(command, StepOnce):
        return True, speed, True
    if isinstance(command, SetTimeScale):
        return paused, command.factor, False
    return paused, speed, step_pending


def keys_to_command(key: str, speed: float) -> Command | None:
    """Map a pressed key to the ``Command`` it expresses, or ``None``."""
    if key in (" ", "p"):
        return Pause()
    if key == "s":
        return StepOnce()
    if key in ("+", "="):
        return SetTimeScale(speed=min(16.0, speed * 2.0))
    if key == "-":
        return SetTimeScale(speed=max(0.25, speed / 2.0))
    if key == "0":
        return SetTimeScale(speed=1.0)
    return None


class InteractiveConsole:
    """Zero-state-run controller: owns the context and marches the sim clock per render."""

    def __init__(self, ctx: SimulationContext, renderer: MatplotlibRenderer) -> None:
        self.ctx: SimulationContext = ctx
        self.renderer: MatplotlibRenderer = renderer
        self.loop: SimulationLoop = SimulationLoop()
        self.time: int = 0
        self.paused: bool = False
        self.speed: float = 1.0
        self.step_pending: bool = False
        self.selected: RobotId | None = None

    def advance(self, render: bool = False) -> None:
        """Advance one render frame: process pending events, then draw the world."""
        if self.step_pending:
            self.loop.step(self.ctx)
            self.step_pending = False
        elif not self.paused:
            until = self.time + int(BASE_STEP_MS * self.speed)
            self.loop.run(self.ctx, until=until)
        self.time = self.ctx.now

        if render:
            self.draw()

    def apply_click(self, x_data: float, y_data: float) -> None:
        """Inject on an empty cell, or select/deselect a robot occupying one."""
        cell = (math.floor(x_data), math.floor(y_data))
        if not self.ctx.world.is_inside(cell):
            return
        robots_here = self.ctx.world.robots_at(cell)
        if robots_here:
            robot = robots_here[0]
            self.selected = None if self.selected == robot else robot
            return
        if self.ctx.world.cell(cell).obstacle:
            return
        self.ctx.apply_command(InjectInfestation(cell=cell, severity=INJECT_SEVERITY))

    def on_click(self, event: object) -> None:
        click = getattr(event, "xdata", None)
        y_click = getattr(event, "ydata", None)
        if click is None or y_click is None:
            return
        self.apply_click(click, y_click)

    def draw(self) -> None:
        """Render a snapshot through the ``Renderer`` port and overlay the inspection panel."""
        self.renderer.render(self.ctx.world.snapshot(self.time))
        self._draw_panel()

    def _draw_panel(self) -> None:
        axes = self.renderer.axes
        if self.selected is not None:
            inspection = self.ctx.agents[self.selected].inspect()
            x, y = self.ctx.world.continuous_position(self.selected, self.time)
            axes.scatter([x], [y], s=240, facecolors="none", edgecolors="black", linewidths=2)
            lines = [
                f"robot {inspection.robot_id}",
                f"goal: {inspection.current_goal}",
                f"belief: {inspection.belief}",
            ]
            axes.text(
                0.02,
                0.985,
                "\n".join(lines),
                transform=axes.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                family="monospace",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
            )
        mode = "PAUSED" if self.paused else f"{self.speed:g}x"
        axes.set_title(
            f"Interactive console | t={self.time / 1000:.1f}s | {mode} | "
            f"click field: inject | click robot: inspect"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=240, help="number of frames to run")
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
    console = InteractiveConsole(build_context(args.seed), MatplotlibRenderer())
    renderer = console.renderer

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for frame in range(args.frames):
            console.advance()
            console.draw()
            renderer.figure.savefig(args.save / f"frame_{frame:04d}.png", dpi=120)
        plt.close(renderer.figure)
        print(f"Wrote {args.frames} frames to {args.save}")
        return

    def on_key(event: object) -> None:
        key = getattr(event, "key", None)
        if key is None:
            return
        command = keys_to_command(key, console.speed)
        if command is not None:
            console.paused, console.speed, console.step_pending = apply_time_command(
                command, console.paused, console.speed, console.step_pending
            )

    def update(_frame: int) -> None:
        console.advance(render=True)

    renderer.figure.canvas.mpl_connect("key_press_event", on_key)
    renderer.figure.canvas.mpl_connect("button_press_event", console.on_click)
    animation = FuncAnimation(
        renderer.figure,
        update,
        frames=args.frames,
        interval=60,
        repeat=True,
    )
    _ = animation
    plt.show()


if __name__ == "__main__":
    main()
