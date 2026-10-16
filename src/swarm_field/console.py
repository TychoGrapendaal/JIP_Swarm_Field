"""The interactive console (C.1): steer, inspect, record and replay one run.

Like ``runner``, this is a composition-level module: it advances a run through the simulation
and draws it through the visualization, so it lives at the package root rather than in
``visualization``, which only ever reads snapshots. The console is the ``Command`` port's
frontend: key presses become the time-pacing commands, and clicks inject infestations or select
a robot to inspect.

The clock always advances through a ``CommandReplay``: a live session replays an empty log and a
replayed session the recorded one, so both take exactly the same path through the run.
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
    TimedCommand,
    TimeUnit,
)
from swarm_field.simulation import CommandReplay, MetricsCollector, SimulationContext
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

    Attributes:
        replay: Advances the context, applying the replayed log's commands on the way.
        renderer: Draws the world.
        label: Names the run in the status box.
        horizon: Simulation time the run stops at, or ``None`` to run until the window closes.
        replaying: Whether the session plays back a recording. Injecting is then disabled, so
            the run stays the recorded one.
        collector: The run's metrics, shown in the title, or ``None`` for a run without them.
        recorded: The commands applied in this session, for ``command_log``.
        time: Simulation time of the frame on screen.
        paused: Whether the clock stands still between frames.
        speed: Multiplier on ``BASE_STEP_MS`` of simulated time per frame.
        step_pending: Whether the next frame advances by exactly one event.
        selected: The robot whose inspection panel is shown, if any.
    """

    def __init__(
        self,
        replay: CommandReplay,
        renderer: MatplotlibRenderer,
        label: str,
        horizon: Time | None = None,
        replaying: bool = False,
        speed: float = 1.0,
        collector: MetricsCollector | None = None,
    ) -> None:
        self.replay: CommandReplay = replay
        self.renderer: MatplotlibRenderer = renderer
        self.label: str = label
        self.horizon: Time | None = horizon
        self.replaying: bool = replaying
        self.collector: MetricsCollector | None = collector
        self.recorded: list[TimedCommand] = []
        self.time: Time = replay.ctx.now
        self.paused: bool = False
        self.speed: float = speed
        self.step_pending: bool = False
        self.selected: RobotId | None = None

    @property
    def ctx(self) -> SimulationContext:
        return self.replay.ctx

    @property
    def finished(self) -> bool:
        """Whether the run has reached its horizon."""
        return self.horizon is not None and self.time >= self.horizon

    def frames_to_horizon(self) -> int:
        """How many frames at the current speed carry the run from now to its horizon.

        Raises:
            ValueError: If the run has no horizon, so it would never end without a window.
        """
        if self.horizon is None:
            raise ValueError("a run without a horizon cannot be counted in frames")
        step = int(BASE_STEP_MS * self.speed)
        return math.ceil(max(self.horizon - self.time, 0) / step)

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
            self.replay.step()
            self.step_pending = False
        elif not self.paused and not self.finished:
            until = self.time + int(BASE_STEP_MS * self.speed)
            if self.horizon is not None:
                until = min(until, self.horizon)
            self.replay.advance_to(until)
        self.time = self.ctx.now
        if render:
            self.draw()

    def apply_click(self, x_data: float, y_data: float) -> None:
        """Select or deselect a robot on the clicked cell, or inject an infestation there."""
        world = self.ctx.world
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
        if self.replaying or not world.is_walkable(cell):
            return
        command = InjectInfestation(cell=cell, severity=INJECT_SEVERITY)
        self.ctx.apply_command(command)
        self.recorded.append(TimedCommand(time=self.ctx.now, command=command))

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

    def command_log(self) -> CommandLog:
        """The recording of this session so far: the replayed log's run and the new commands."""
        log = self.replay.log
        return CommandLog(
            seed=log.seed, until=self.time, commands=tuple(self.recorded), scenario=log.scenario
        )

    def draw(self) -> None:
        """Render a snapshot through the ``Renderer`` port, then the inspection and status boxes.

        The title is left to the renderer, which writes the latest metrics sample into it.
        """
        latest = None
        if self.collector is not None:
            latest = self.collector.latest()
        self.renderer.render(self.ctx.world.snapshot(self.time, latest))
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
        inspection = self.ctx.agents[robot].inspect()
        x, y = self.ctx.world.continuous_position(robot, self.time)
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
        clock = f"t={format_time(self.time)}"
        if self.horizon is not None:
            clock = f"{clock} of {format_time(self.horizon)}"
        controls = CONTROL_LINES
        if self.replaying:
            controls = REPLAY_CONTROL_LINES
        return [f"{self.label} | {clock} | {mode}", *controls]


def add_session_arguments(
    parser: argparse.ArgumentParser, default_speed: float, default_hours: float | None
) -> None:
    """Add the flags every console session shares, so each script offers them the same way.

    ``--seed`` and ``--hours`` default to ``None`` and ``default_hours``; a replay takes both
    from its recording instead, which ``validate_session_arguments`` enforces.
    """
    parser.add_argument("--seed", type=int, default=None, help="random seed of a new session")
    hours_default = "the scenario's own horizon, or until the window closes if it has none"
    if default_hours is not None:
        hours_default = f"{default_hours:g}"
    parser.add_argument(
        "--hours",
        type=float,
        default=default_hours,
        help=f"simulated hours the run lasts (default: {hours_default})",
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


def validate_session_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace, default_hours: float | None
) -> None:
    """Refuse a replay that also asks for its own seed or length, which come from the recording.

    Exits through ``parser.error``, like any other invalid command line.
    """
    if args.replay is None:
        return
    if args.seed is not None:
        parser.error("--seed cannot be combined with --replay: the recording holds the seed")
    if args.hours != default_hours:
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


def write_recording(console: InteractiveConsole, path: Path) -> None:
    """Write the session's ``CommandLog`` to ``path`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(console.command_log().to_json(), encoding="utf-8")


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
