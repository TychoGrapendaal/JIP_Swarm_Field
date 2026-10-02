"""Watch one real experiment in a steerable window (C.1 over B.5).

``interactive_demo.py`` drives a hand-built three-robot world, so it can exercise the ``Command``
port on its own. This demo drives a *real* experiment instead: ``runner.build_experiment`` resolves
the scenario preset and ``runner.build_run`` assembles the robot roster and brains, so what you see
is the same object graph ``runner.run_experiment`` would drive. The only difference is that the
render loop advances ``SimulationLoop`` in slices, which is what makes pausing and stepping
possible.

- Click an empty cell to inject an infestation (``InjectInfestation`` -> ``apply_command``).
- Click a robot to inspect it (``SwarmAgent.inspect()`` -> ``AgentInspection`` panel).

Time controls (``Pause`` / ``StepOnce`` / ``SetTimeScale``) pace the render loop:

- space / p    pause / resume
- s            step once: advance by exactly the next simulation event
- + / =        double the simulation speed
- -            halve the simulation speed
- 0            reset to 1x

A scenario preset carries no duration of its own, so ``--hours`` sets the horizon. It defaults to
the one hour ``runner.sweep_scenarios`` uses, so a viewer run and a sweep run cover the same span.

The metrics cadence is shortened to match the pest dynamics, because the default of one sample per
hour would leave the figures in the title static for the whole run. Sampling cadence only changes
what is measured, never what the world does: ``MetricsCollector`` holds no random source.

Run:
    uv run python examples/scenario_viewer.py

Headless (write frames as PNGs instead of opening a window):
    uv run python examples/scenario_viewer.py --frames 240 --save frames/viewer
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from swarm_field.contracts import (
    AgentInspection,
    CellId,
    Command,
    InfestationObservation,
    InjectInfestation,
    Pause,
    RobotId,
    ScenarioId,
    SetTimeScale,
    StepOnce,
    TimeUnit,
)
from swarm_field.runner import ExperimentConfig, build_experiment, build_run
from swarm_field.simulation import (
    MetricsCollector,
    MetricsConfig,
    SimulationContext,
    SimulationLoop,
)
from swarm_field.visualization import MatplotlibRenderer

HOUR_MS = TimeUnit.HOUR
DEFAULT_SCENARIO = ScenarioId.S2
DEFAULT_SEED = 101
DEFAULT_HOURS = 1.0
DEFAULT_FRAMES = 20_000
BASE_STEP_MS = 500
INITIAL_SPEED = 4.0
MIN_SPEED = 0.25
MAX_SPEED = 16.0
METRICS_PERIOD_MS = 60_000
INJECT_SEVERITY = 0.8
FRAME_INTERVAL_MS = 60
CHROME_LINES = (
    "space/p pause  s step  +/- speed  0 -> 1x",
    "click field: inject  |  click robot: inspect",
)


def parse_scenario(name: str) -> ScenarioId:
    """Resolve a scenario name such as ``S2`` to its preset, or exit with the valid names."""
    try:
        return ScenarioId[name]
    except KeyError:
        choices = ", ".join(member.name for member in ScenarioId)
        raise SystemExit(f"unknown scenario {name!r}: choose from {choices}") from None


def summarise_belief(inspection: AgentInspection) -> str:
    """Summarise one robot's belief into a line short enough to read over a forty-wide field.

    ``AgentInspection.belief`` is typed ``object`` and holds the raw tuple of observations, which is
    far too long to print verbatim once a fifteen-robot scenario is running. This counts the
    observations by kind and names the worst infestation the robot believes it is near, which is
    the part a reader is looking for.
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
        return SetTimeScale(factor=min(MAX_SPEED, speed * 2.0))
    if key == "-":
        return SetTimeScale(factor=max(MIN_SPEED, speed / 2.0))
    if key == "0":
        return SetTimeScale(factor=1.0)
    return None


class ScenarioViewer:
    """Drives one built experiment a render frame at a time.

    The context comes from ``runner.build_run``, so this class owns no simulation rules: it only
    decides how far to advance the loop between frames and what to draw. It builds that run without
    a renderer on purpose -- handing one to ``build_run`` would make the simulation draw on every
    metric tick as well, fighting the render clock.
    """

    def __init__(self, config: ExperimentConfig, renderer: MatplotlibRenderer) -> None:
        self.ctx: SimulationContext
        self.collector: MetricsCollector
        self.ctx, self.collector = build_run(config)
        self.renderer: MatplotlibRenderer = renderer
        self.loop: SimulationLoop = SimulationLoop()
        self.config: ExperimentConfig = config
        self.time: int = 0
        self.paused: bool = False
        self.speed: float = INITIAL_SPEED
        self.step_pending: bool = False
        self.selected: RobotId | None = None

    @property
    def finished(self) -> bool:
        """Whether the run has reached the horizon its configuration asked for."""
        return self.time >= self.config.until

    def advance(self, render: bool = False) -> None:
        """Advance one render frame: process events, then draw the world.

        Advancing in slices of simulated time is what the viewer adds over
        ``runner.run_experiment``; ``World.advance_to`` settles each move once and clears it, so
        slicing to a horizon measures exactly what jumping straight there would.
        """
        if self.step_pending:
            self.loop.step(self.ctx)
            self.step_pending = False
        elif not self.paused and not self.finished:
            target = min(self.time + int(BASE_STEP_MS * self.speed), self.config.until)
            self.loop.run(self.ctx, until=target)
        self.time = self.ctx.now
        if render:
            self.draw()

    def apply_click(self, x_data: float, y_data: float) -> None:
        """Inject on an empty cell, or select/deselect a robot occupying one."""
        world = self.ctx.world
        cell = (math.floor(x_data), math.floor(y_data))
        if not world.is_inside(cell):
            return
        robots_here = world.robots_at(cell)
        if robots_here:
            robot = robots_here[0]
            self.selected = None if self.selected == robot else robot
            return
        if not world.is_walkable(cell):
            return
        self.ctx.apply_command(InjectInfestation(cell=cell, severity=INJECT_SEVERITY))

    def on_click(self, event: object) -> None:
        """Handle a matplotlib click, ignoring presses outside the axes."""
        x_data = getattr(event, "xdata", None)
        y_data = getattr(event, "ydata", None)
        if x_data is None or y_data is None:
            return
        self.apply_click(x_data, y_data)

    def on_key(self, event: object) -> None:
        """Apply a key press as the time-pacing command it expresses."""
        key = getattr(event, "key", None)
        if key is None:
            return
        command = keys_to_command(key, self.speed)
        if command is None:
            return
        self.paused, self.speed, self.step_pending = apply_time_command(
            command, self.paused, self.speed, self.step_pending
        )

    def draw(self) -> None:
        """Render a snapshot through the ``Renderer`` port and overlay the inspection panel.

        The metrics sample is read from the collector rather than the world, because the collector
        is what the loop samples on each metric tick; passing it on lets the renderer keep the
        figures in its title.
        """
        self.renderer.render(self.ctx.world.snapshot(self.time, self.collector.latest()))
        self._draw_panel()

    def _draw_panel(self) -> None:
        """Ring the selected robot and annotate it, and add the frontend's own chrome.

        This deliberately leaves the title alone: the renderer owns it, and overwriting it would
        drop the live metrics it just wrote there.
        """
        axes = self.renderer.axes
        if self.selected is not None:
            inspection = self.ctx.agents[self.selected].inspect()
            x, y = self.ctx.world.continuous_position(self.selected, self.time)
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
        axes.text(
            0.02,
            0.015,
            self._status_line(),
            transform=axes.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            family="monospace",
            color="black",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        )

    def _status_line(self) -> str:
        """Describe the run and the current time-pacing mode for the viewer."""
        if self.finished:
            mode = "finished"
        elif self.paused:
            mode = "paused"
        else:
            mode = f"{self.speed:g}x"
        horizon = self.config.until / HOUR_MS
        lines = [
            f"{self.config.label} | t={self.time / HOUR_MS:.2f}h of {horizon:g}h | {mode}",
            *CHROME_LINES,
        ]
        return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Read the command line for which scenario to watch and for how long."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO.name,
        help="scenario preset to watch, by name",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_HOURS,
        help="how far to run the scenario, in simulated hours",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help="maximum number of frames to draw before the window stops",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="directory to save PNG frames instead of opening a window",
    )
    return parser.parse_args()


def main() -> None:
    """Watch one seeded scenario in a window, or write its frames headless."""
    args = parse_args()
    scenario = parse_scenario(args.scenario)
    config = build_experiment(
        label=f"{scenario.name}/{args.seed}",
        scenario=scenario,
        seed=args.seed,
        until=int(args.hours * HOUR_MS),
        metrics=MetricsConfig(period=METRICS_PERIOD_MS),
    )
    viewer = ScenarioViewer(config, MatplotlibRenderer())
    renderer = viewer.renderer

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        written = 0
        while written < args.frames and not viewer.finished:
            viewer.advance()
            viewer.draw()
            renderer.figure.savefig(args.save / f"frame_{written:04d}.png", dpi=120)
            written += 1
        plt.close(renderer.figure)
        state = "the run finished" if viewer.finished else f"only {args.frames} frames were drawn"
        print(f"Wrote {written} frames to {args.save}: {state}")
        return

    renderer.figure.canvas.mpl_connect("key_press_event", viewer.on_key)
    renderer.figure.canvas.mpl_connect("button_press_event", viewer.on_click)
    animation = FuncAnimation(
        renderer.figure,
        lambda _frame: viewer.advance(render=True),
        frames=args.frames,
        interval=FRAME_INTERVAL_MS,
        repeat=True,
    )
    _ = animation
    plt.show()


if __name__ == "__main__":
    main()
