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

Record a session, then replay it exactly, in a window or as a video (``.gif``, or ``.mp4`` if
ffmpeg is installed). The recording holds the seed and every injected pest with its simulation
time, and is written when the window closes:
    uv run python examples/interactive_demo.py --record recordings/demo.json
    uv run python examples/interactive_demo.py --replay recordings/demo.json
    uv run python examples/interactive_demo.py --replay recordings/demo.json --save demo.mp4
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Hashable
from pathlib import Path
from random import Random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from swarm_field.contracts import (
    BodyType,
    CellId,
    Command,
    CommandLog,
    FieldGeometry,
    InjectInfestation,
    Pause,
    RobotId,
    SetTimeScale,
    StepOnce,
    SwarmAgent,
    Time,
    TimedCommand,
)
from swarm_field.simulation import (
    CommandReplay,
    PestDynamicsConfig,
    RobotDecision,
    RobotState,
    SimulationContext,
    WorldState,
    WorldTick,
)
from swarm_field.swarm import (
    ActionCandidate,
    AgentMemory,
    ComposedAgent,
    DecisionContext,
    FieldProvider,
    FollowField,
    MemoryView,
    Navigator,
)
from swarm_field.swarm.field import Seed
from swarm_field.visualization import MatplotlibRenderer, save_frames

FIELD_WIDTH = 24
FIELD_HEIGHT = 16
CELL_SIZE_M = 2.0
BASE_STEP_MS = 500
WORLD_TICK_PERIOD = 3_000
INJECT_SEVERITY = 0.8
DEFAULT_FRAMES = 240
FRAME_INTERVAL_MS = 60
SPEED_BY_BODY = {
    BodyType.SCOUT: 2.0,
    BodyType.KILLER: 1.2,
    BodyType.BATTERY: 0.8,
}


class _WanderSeeder:
    """Stand-in for the EXPLORE seeder (#41): a random walk over believed ground.

    It names one random neighbouring cell as the wander field's only target, and asks for a new
    one at every decision point by making the field key the decision time. The navigator then
    rebuilds the field per tick and the robot steps towards it. The target is clamped into the
    field, so it is never the unreachable outside cell a robot standing at the edge would pick.

    A real exploration algorithm would seed unvisited or stale cells instead of a random
    neighbour, and would keep them long enough to be worth walking to.
    """

    name: str = "wander"

    def __init__(self, field: FieldGeometry, rng: Random) -> None:
        self._field: FieldGeometry = field
        self._rng: Random = rng

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        return now

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        dx, dy = self._rng.choice(((1, 0), (0, 1), (-1, 0), (0, -1)))
        x = min(max(robot[0] + dx, 0), self._field.width - 1)
        y = min(max(robot[1] + dy, 0), self._field.height - 1)
        return (Seed((x, y), 0.0),)


class WanderBehavior:
    """A minimal exploration behavior: bid to follow the wander field.

    The behaviour never moves the robot itself; it names the seeder whose field to follow and
    the navigator takes the step. Standing on the field's target needs no handling here: the
    navigator returns ``Idle`` for that case.
    """

    name: str = "wander"

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        candidate = ActionCandidate(
            action=FollowField(_WanderSeeder.name),
            score=1.0,
            source=self.name,
            target=None,
        )
        return (candidate,)


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
    agents: dict[RobotId, SwarmAgent] = {}
    for robot_id in range(len(robots)):
        memory = AgentMemory(robot_id, field)
        fields = FieldProvider(
            memory,
            flies=False,
            # Its own random source, so a robot's walk does not depend on how many draws its
            # peers happened to make first.
            seeders=[_WanderSeeder(field, Random(seed + robot_id))],
        )
        agents[robot_id] = ComposedAgent(
            robot_id=robot_id,
            behaviors=[WanderBehavior()],
            memory=memory,
            fields=fields,
            navigator=Navigator(fields.quadtree),
        )
    ctx = SimulationContext(
        now=0,
        rng=rng,
        agents=agents,
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
        return SetTimeScale(factor=min(16.0, speed * 2.0))
    if key == "-":
        return SetTimeScale(factor=max(0.25, speed / 2.0))
    if key == "0":
        return SetTimeScale(factor=1.0)
    return None


class InteractiveConsole:
    """Owns a running context, marches its clock per render frame and records the commands.

    The clock always advances through a ``CommandReplay``: a live session replays an empty
    log, and a replayed session the recorded one, so both take exactly the same path.

    Attributes:
        replay: Advances the context, applying the replayed log's commands on the way.
        renderer: Draws the world.
        replaying: Whether the session plays back a recording. Injecting is then disabled, so
            the run stays the recorded one, and the clock stops where the recording ended.
        recorded: The commands applied in this session, for ``command_log``.
    """

    def __init__(
        self,
        replay: CommandReplay,
        renderer: MatplotlibRenderer,
        replaying: bool = False,
    ) -> None:
        self.replay: CommandReplay = replay
        self.ctx: SimulationContext = replay.ctx
        self.renderer: MatplotlibRenderer = renderer
        self.replaying: bool = replaying
        self.recorded: list[TimedCommand] = []
        self.time: int = 0
        self.paused: bool = False
        self.speed: float = 1.0
        self.step_pending: bool = False
        self.selected: RobotId | None = None

    def advance(self, render: bool = False) -> None:
        """Advance one render frame: process pending events, then draw the world."""
        if self.step_pending:
            self.replay.step()
            self.step_pending = False
        elif not self.paused:
            until = self.time + int(BASE_STEP_MS * self.speed)
            if self.replaying:
                until = min(until, self.replay.log.until)
            self.replay.advance_to(until)
        self.time = self.ctx.now

        if render:
            self.draw()

    def command_log(self) -> CommandLog:
        """The recording of this session so far: its seed and the commands applied in it."""
        return CommandLog(seed=self.replay.log.seed, until=self.time, commands=tuple(self.recorded))

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
        if self.replaying or self.ctx.world.cell(cell).obstacle:
            return
        command = InjectInfestation(cell=cell, severity=INJECT_SEVERITY)
        self.ctx.apply_command(command)
        self.recorded.append(TimedCommand(time=self.ctx.now, command=command))

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
        hint = "click field: inject | click robot: inspect"
        if self.replaying:
            hint = f"REPLAY until t={self.replay.log.until / 1000:.1f}s | click robot: inspect"
        axes.set_title(f"Interactive console | t={self.time / 1000:.1f}s | {mode} | {hint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help=f"number of frames to run (default {DEFAULT_FRAMES}, or a replay's full length)",
    )
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="save instead of opening a window: a .gif or .mp4 video, or a directory of PNGs",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="write the session's seed and injected pests to this JSON file at the end",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="play back a recording written by --record instead of a fresh session",
    )
    return parser.parse_args()


def save_session(console: InteractiveConsole, frames: int, path: Path) -> None:
    """Run ``frames`` frames without a window, writing a video or a directory of PNGs."""
    figure = console.renderer.figure

    def draw_frame(_index: int) -> None:
        console.advance(render=True)

    save_frames(figure, draw_frame, frames, path, fps=1000 // FRAME_INTERVAL_MS)
    plt.close(figure)
    print(f"Wrote {frames} frames to {path}")


def write_recording(console: InteractiveConsole, path: Path | None) -> None:
    """Write the session's ``CommandLog`` to ``path``, if a recording was asked for."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(console.command_log().to_json(), encoding="utf-8")
    print(f"Recorded {len(console.recorded)} commands up to t={console.time / 1000:.1f}s in {path}")


def main() -> None:
    args = parse_args()
    log = CommandLog(seed=args.seed, until=0, commands=())
    if args.replay is not None:
        log = CommandLog.from_json(args.replay.read_text(encoding="utf-8"))
    replaying = args.replay is not None
    console = InteractiveConsole(
        CommandReplay(build_context(log.seed), log), MatplotlibRenderer(), replaying=replaying
    )
    renderer = console.renderer

    frames = args.frames
    if frames is None:
        frames = DEFAULT_FRAMES
        if replaying:
            frames = math.ceil(log.until / BASE_STEP_MS) + 1

    if args.save is not None:
        save_session(console, frames, args.save)
        write_recording(console, args.record)
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
        frames=frames,
        interval=FRAME_INTERVAL_MS,
        repeat=True,
    )
    _ = animation
    plt.show()
    write_recording(console, args.record)


if __name__ == "__main__":
    main()
