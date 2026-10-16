"""The interactive console (C.1): steer, inspect, record and replay one run.

Like ``runner``, this is a composition-level module: it advances a run through the simulation
and draws it through the visualization, so it lives at the package root rather than in
``visualization``, which only ever reads snapshots. The console is the ``Command`` port's
frontend: key presses become the time-pacing commands, and clicks inject infestations or select
a robot to inspect. The run itself, and its recording, is an ``experiments.LiveRun``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from swarm_field.contracts import (
    AgentInspection,
    CellId,
    Command,
    CommandLog,
    InfestationObservation,
    InjectInfestation,
    Pause,
    RobotId,
    SetTimeScale,
    StepOnce,
    Time,
    TimeUnit,
)
from swarm_field.experiments import LiveRun
from swarm_field.visualization import MatplotlibRenderer, save_frames

BASE_STEP_MS: Time = 500
"""Simulated time one frame advances at speed 1."""
MIN_SPEED = 0.25
MAX_SPEED = 64.0
FRAME_INTERVAL_MS = 60
"""Time between frames in a window, which also sets a video's frame rate."""
INJECT_SEVERITY = 0.8
CONTROL_LINES = (
    "space/p pause  s step  +/- speed  0 -> 1x",
    "click field: inject  |  click robot: inspect",
)
REPLAY_CONTROL_LINES = (
    "space/p pause  s step  +/- speed  0 -> 1x",
    "replay: click robot to inspect",
)


def keys_to_command(key: str, speed: float) -> Command | None:
    """Map a pressed key to the time-pacing ``Command`` it expresses, or ``None``."""
    if key in (" ", "p"):
        return Pause()
    if key == "s":
        return StepOnce()
    if key in ("+", "="):
        return SetTimeScale(factor=min(MAX_SPEED, speed * 2.0))
    if key == "-":
        return SetTimeScale(factor=max(MIN_SPEED, speed / 2.0))
    if key == "0":
        return SetTimeScale(factor=1.0)
    return None


def summarise_belief(inspection: AgentInspection) -> str:
    """Summarise one robot's belief into a line short enough to read over the field.

    ``AgentInspection.belief`` is typed ``object`` and holds the raw tuple of observations, far
    too long to print once robots share knowledge. This counts the observations by kind and
    names the worst infestation the robot believes in, which is the part a reader looks for.
    """
    belief = inspection.belief
    if not isinstance(belief, tuple):
        return str(belief)
    counts: dict[str, int] = {}
    worst: tuple[float, CellId] | None = None
    for observation in belief:
        kind = type(observation).__name__.removesuffix("Observation").lower()
        counts[kind] = counts.get(kind, 0) + 1
        if isinstance(observation, InfestationObservation) and (
            worst is None or observation.severity > worst[0]
        ):
            worst = (observation.severity, observation.cell)
    parts = [f"{kind}={counts[kind]}" for kind in sorted(counts)]
    if worst is not None:
        severity, cell = worst
        parts.append(f"worst {cell}@{severity:.2f}")
    if not parts:
        return "empty"
    return " ".join(parts)


def format_time(time: Time) -> str:
    """A simulation time in the unit a reader expects: seconds below an hour, hours above."""
    if time < TimeUnit.HOUR:
        return f"{time / TimeUnit.SECOND:.1f}s"
    return f"{time / TimeUnit.HOUR:.2f}h"


class InteractiveConsole:
    """Advances one run a render frame at a time, and turns key presses and clicks into commands.

    The run stops at its horizon. In a replayed run injecting is disabled, so the run stays the
    recorded one.

    Attributes:
        run: The run being steered, which also records the commands applied to it.
        renderer: Draws the world.
        paused: Whether the clock stands still between frames.
        speed: Multiplier on ``BASE_STEP_MS`` of simulated time per frame.
        step_pending: Whether the next frame advances by exactly one event.
        selected: The robot whose inspection panel is shown, if any.
    """

    def __init__(self, run: LiveRun, renderer: MatplotlibRenderer, speed: float = 1.0) -> None:
        self.run: LiveRun = run
        self.renderer: MatplotlibRenderer = renderer
        self.paused: bool = False
        self.speed: float = speed
        self.step_pending: bool = False
        self.selected: RobotId | None = None

    @property
    def time(self) -> Time:
        """The simulation time of the frame on screen."""
        return self.run.now

    @property
    def finished(self) -> bool:
        """Whether the run has reached its horizon."""
        return self.time >= self.run.horizon

    def frames_to_horizon(self) -> int:
        """How many frames at the current speed carry the run from now to its horizon."""
        step = int(BASE_STEP_MS * self.speed)
        return math.ceil(max(self.run.horizon - self.time, 0) / step)

    def apply_time_command(self, command: Command) -> None:
        """Apply a time-pacing command to the render loop's state.

        ``StepOnce`` steps exactly once and then freezes: after the single event the run stays
        paused until the user resumes. ``InjectInfestation`` changes the world, not the clock,
        so it is ignored here.
        """
        if isinstance(command, Pause):
            self.paused = not self.paused
            self.step_pending = False
        elif isinstance(command, StepOnce):
            self.paused = True
            self.step_pending = True
        elif isinstance(command, SetTimeScale):
            self.speed = command.factor
            self.step_pending = False

    def advance(self, render: bool = False) -> None:
        """Advance one render frame: process the events it covers, then draw the world."""
        if self.step_pending:
            self.run.step()
            self.step_pending = False
        elif not self.paused and not self.finished:
            until = min(self.time + int(BASE_STEP_MS * self.speed), self.run.horizon)
            self.run.advance_to(until)
        if render:
            self.draw()

    def apply_click(self, x_data: float, y_data: float) -> None:
        """Select or deselect a robot on the clicked cell, or inject an infestation there."""
        world = self.run.context.world
        cell = (math.floor(x_data), math.floor(y_data))
        if not world.is_inside(cell):
            return
        robots_here = world.robots_at(cell)
        if robots_here:
            robot = robots_here[0]
            if self.selected == robot:
                self.selected = None
            else:
                self.selected = robot
            return
        if self.run.replaying or not world.is_walkable(cell):
            return
        self.run.apply_command(InjectInfestation(cell=cell, severity=INJECT_SEVERITY))

    def on_click(self, event: object) -> None:
        """Handle a matplotlib click, ignoring presses outside the axes."""
        x_data = getattr(event, "xdata", None)
        y_data = getattr(event, "ydata", None)
        if x_data is None or y_data is None:
            return
        self.apply_click(x_data, y_data)

    def on_key(self, event: object) -> None:
        """Handle a matplotlib key press as the time-pacing command it expresses."""
        key = getattr(event, "key", None)
        if key is None:
            return
        command = keys_to_command(key, self.speed)
        if command is not None:
            self.apply_time_command(command)

    def draw(self) -> None:
        """Render a snapshot through the ``Renderer`` port, then the inspection and status boxes.

        The title is left to the renderer, which writes the latest metrics sample into it.
        """
        self.renderer.render(self.run.snapshot())
        axes = self.renderer.axes
        if self.selected is not None:
            self._draw_inspection(self.selected)
        axes.text(
            0.02,
            0.015,
            "\n".join(self._status_lines()),
            transform=axes.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        )

    def _draw_inspection(self, robot: RobotId) -> None:
        """Ring the selected robot and show its ``AgentInspection``."""
        axes = self.renderer.axes
        inspection = self.run.context.agents[robot].inspect()
        x, y = self.run.context.world.continuous_position(robot, self.time)
        axes.scatter([x], [y], s=260, facecolors="none", edgecolors="black", linewidths=2.0)
        lines = [
            f"robot {inspection.robot_id}",
            f"goal: {inspection.current_goal}",
            f"belief: {summarise_belief(inspection)}",
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

    def _status_lines(self) -> list[str]:
        """The run, the time, the pacing mode and the controls."""
        if self.finished:
            mode = "finished"
        elif self.paused:
            mode = "paused"
        else:
            mode = f"{self.speed:g}x"
        clock = f"t={format_time(self.time)} of {format_time(self.run.horizon)}"
        controls = CONTROL_LINES
        if self.run.replaying:
            controls = REPLAY_CONTROL_LINES
        return [f"{self.run.label} | {clock} | {mode}", *controls]


def add_session_arguments(parser: argparse.ArgumentParser, default_speed: float) -> None:
    """Add the flags every console session shares, so each script offers them the same way.

    ``--seed`` and ``--hours`` default to ``None``, meaning the script's default seed and the
    scenario's own horizon; a replay takes both from its recording instead, which
    ``validate_session_arguments`` enforces.
    """
    parser.add_argument("--seed", type=int, default=None, help="random seed of a new session")
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="simulated hours the run lasts (default: the scenario's own horizon)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=default_speed,
        help=f"starting speed; one frame covers {BASE_STEP_MS / 1000:g} s of simulation times this",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="run to the horizon without a window, into a .gif or .mp4 video or a PNG directory",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="write the session's seed and injected pests to this JSON file when it ends",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="play back a recording written by --record instead of starting a new session",
    )


def validate_session_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse a replay that also asks for its own seed or length, which come from the recording.

    Exits through ``parser.error``, like any other invalid command line.
    """
    if args.replay is None:
        return
    if args.seed is not None:
        parser.error("--seed cannot be combined with --replay: the recording holds the seed")
    if args.hours is not None:
        parser.error("--hours cannot be combined with --replay: the recording holds the length")
    if args.record is not None:
        parser.error("--record cannot be combined with --replay: a replay adds no commands")


def save_session(console: InteractiveConsole, path: Path) -> int:
    """Run the console to its horizon without a window, into a video or PNG directory at ``path``.

    Returns:
        How many frames were written.
    """
    import matplotlib.pyplot as plt

    frames = console.frames_to_horizon()
    save_frames(
        console.renderer.figure,
        lambda _index: console.advance(render=True),
        frames,
        path,
        fps=1000 // FRAME_INTERVAL_MS,
    )
    plt.close(console.renderer.figure)
    return frames


def write_recording(log: CommandLog, path: Path) -> None:
    """Write ``log`` to ``path`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(log.to_json(), encoding="utf-8")


def show_window(console: InteractiveConsole) -> None:
    """Open a window that advances the console every frame until the window is closed."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    def draw_frame(_frame: int) -> None:
        console.advance(render=True)

    figure = console.renderer.figure
    figure.canvas.mpl_connect("key_press_event", console.on_key)
    figure.canvas.mpl_connect("button_press_event", console.on_click)
    animation = FuncAnimation(
        figure,
        draw_frame,
        interval=FRAME_INTERVAL_MS,
        cache_frame_data=False,
    )
    _ = animation
    plt.show()
