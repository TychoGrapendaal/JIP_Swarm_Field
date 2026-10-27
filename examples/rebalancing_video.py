"""A video of role rebalancing (#64): fixed roles against rebalancing at the home base.

Four runs of the default small river field on the same seed, so with the same pests, side by
side: seeded killers on top, hop-based killers below, fixed roles on the left and rebalancing on
the right (the catalogue's ``rebalancing-fixed``, ``rebalancing``, ``rebalancing-hop-fixed`` and
``rebalancing-hop``). Explorers are circles and killers triangles; a purple ring marks a robot
driving to a dock to switch (``AgentObjective.SWITCH_ROLE``), and a yellow ring a robot that has
just switched. Below the fields, the killers over time for each method, with a cursor.

The four runs go in parallel, then the frames are drawn; the video needs ffmpeg. Run from the
repository root with:

    uv run python examples/rebalancing_video.py --save videos/rebalancing.mp4
    uv run python examples/rebalancing_video.py --seed 3 --hours 1 --save videos/seed3.mp4
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from swarm_field.contracts import AgentObjective, BodyType, TimeUnit
from swarm_field.experiments import REBALANCING, LiveRun, build_named_experiment

if TYPE_CHECKING:
    from matplotlib.axes import Axes

PANELS: tuple[tuple[str, str, str], ...] = (
    ("rebalancing-fixed", "Seeded killers", "Fixed roles"),
    ("rebalancing", "Seeded killers", "Rebalancing at the home base"),
    ("rebalancing-hop-fixed", "Hop-based killers", "Fixed roles"),
    ("rebalancing-hop", "Hop-based killers", "Rebalancing at the home base"),
)
"""The catalogue scenario of every field, by row and column: its method and its roles."""
SCOUT = "#1f5fa8"
KILLER = "#d6402b"
SWITCHED = "#f5b800"
HEADING = "#9b30d0"
PEST = "#8b2c00"
METHOD_LINE = {"Seeded killers": "#e07b28", "Hop-based killers": "#1f5fa8"}
FIXED_LINE = "#888888"
FRAME_STEP = 20 * TimeUnit.SECOND
"""Simulated time between two frames of the video."""


@dataclass(frozen=True)
class Robot:
    """A robot in one frame: where it is, its body, and whether it is heading for a dock."""

    robot_id: int
    x: float
    y: float
    body: BodyType
    to_dock: bool


@dataclass(frozen=True)
class Frame:
    """One frame of a run: its time in minutes, its robots and its pests (cell, infestation)."""

    minute: float
    robots: tuple[Robot, ...]
    pests: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True)
class Recording:
    """A run, frame by frame, with the killers per frame and every switch (minute, robot)."""

    frames: tuple[Frame, ...]
    killers: tuple[int, ...]
    switches: tuple[tuple[float, int], ...]


def record(name: str, seed: int, hours: float) -> Recording:
    """Run the catalogue's ``name`` on ``seed`` for ``hours`` and take a frame every 20 s."""
    logging.disable(logging.WARNING)
    until = int(hours * TimeUnit.HOUR)
    run = LiveRun.from_config(build_named_experiment(name, seed, until))
    world = run.context.world
    agents = run.context.agents
    bodies = {robot_id: robot.body for robot_id, robot in world.robots.items()}
    frames: list[Frame] = []
    killers: list[int] = []
    switches: list[tuple[float, int]] = []
    for time in range(0, until + 1, FRAME_STEP):
        run.advance_to(time)
        minute = time / TimeUnit.MINUTE
        robots: list[Robot] = []
        for robot_id, robot in world.robots.items():
            if robot.body is not bodies[robot_id]:
                switches.append((minute, robot_id))
                bodies[robot_id] = robot.body
            x, y = world.continuous_position(robot_id, time)
            objective = agents[robot_id].inspect().current_objective
            to_dock = objective is AgentObjective.SWITCH_ROLE
            robots.append(Robot(robot_id, x, y, robot.body, to_dock))
        pests: list[tuple[int, int, float]] = []
        for cell, state in world.cells.items():
            if state.infestation > 0:
                pests.append((cell[0], cell[1], state.infestation))
        frames.append(Frame(minute, tuple(robots), tuple(pests)))
        killers.append(sum(1 for robot in robots if robot.body is BodyType.KILLER))
    return Recording(tuple(frames), tuple(killers), tuple(switches))


def render(recordings: list[Recording], save: Path, fps: int) -> None:
    """Draw the four recordings side by side, a frame per frame, into ``save``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    figure = plt.figure(figsize=(12.0, 13.0), dpi=100)
    figure.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.98)
    grid = figure.add_gridspec(3, 2, height_ratios=[1, 1, 0.42], hspace=0.3, wspace=0.06)
    fields = [figure.add_subplot(grid[index // 2, index % 2]) for index in range(4)]
    timelines = [figure.add_subplot(grid[2, column]) for column in range(2)]
    count = min(len(recording.frames) for recording in recordings)
    end = recordings[0].frames[count - 1].minute
    cursors = []
    for row, axes in enumerate(timelines):
        method = PANELS[2 * row][1]
        for column, colour in enumerate((FIXED_LINE, METHOD_LINE[method])):
            recording = recordings[2 * row + column]
            minutes = [frame.minute for frame in recording.frames]
            axes.plot(
                minutes, recording.killers, color=colour, lw=1.5, label=PANELS[2 * row + column][2]
            )
        axes.set_xlim(0, end)
        axes.set_ylim(-0.5, 12.5)
        axes.set_xlabel("minute")
        axes.set_title(f"Killers over time, {method.lower()}", fontsize=10.5, loc="left")
        axes.grid(alpha=0.25)
        axes.spines[["top", "right"]].set_visible(False)
        axes.legend(fontsize=8, frameon=False, loc="upper right")
        cursors.append(axes.axvline(0, color="black", lw=1))
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    save.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(figure, str(save), dpi=100):
        for index in range(count):
            minute = recordings[0].frames[index].minute
            for axes, recording, (_, method, roles) in zip(fields, recordings, PANELS, strict=True):
                _draw_field(axes, recording, index, f"{method}: {roles}")
            for cursor in cursors:
                cursor.set_xdata([minute, minute])
            hours, minutes = divmod(int(minute), 60)
            figure.suptitle(
                f"Small river field, the same pests in all four   {hours}:{minutes:02d} h\n"
                "● explorer   ▲ killer   purple ring = driving to a dock to switch   "
                "yellow ring = just switched   black squares = docks   brown = pest",
                fontsize=10.5,
                x=0.06,
                ha="left",
                y=0.975,
            )
            writer.grab_frame()
    plt.close(figure)


def _draw_field(axes: Axes, recording: Recording, index: int, title: str) -> None:
    """One field at frame ``index``: rocks and rivers, docks, pests and robots."""
    from matplotlib.patches import Rectangle

    field = REBALANCING.map
    frame = recording.frames[index]
    axes.clear()
    axes.set_xlim(-1.3, field.size + 0.3)
    axes.set_ylim(-1.3, field.size + 0.3)
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_facecolor("#eef3e6")
    for x, y in field.blocked_cells():
        axes.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="#9aa7b0", lw=0))
    for x, y in field.docks:
        axes.add_patch(Rectangle((x - 1.0, y - 1.0), 2, 2, color="#2b2b2b", alpha=0.18, lw=0))
        axes.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ec="black", lw=1.6))
    for x, y, infestation in frame.pests:
        alpha = min(1.0, 0.35 + infestation)
        axes.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color=PEST, alpha=alpha, lw=0))
    recent = {robot for minute, robot in recording.switches if 0 <= frame.minute - minute <= 1.0}
    heading = 0
    for robot in frame.robots:
        if robot.body is BodyType.KILLER:
            axes.plot(robot.x, robot.y, "^", ms=8.5, mfc=KILLER, mec="white", mew=0.8)
        else:
            axes.plot(robot.x, robot.y, "o", ms=7, mfc=SCOUT, mec="white", mew=0.8)
        if robot.to_dock:
            heading += 1
            axes.plot(robot.x, robot.y, "o", ms=14, mfc="none", mec=HEADING, mew=1.8)
        if robot.robot_id in recent:
            axes.plot(robot.x, robot.y, "o", ms=17, mfc="none", mec=SWITCHED, mew=2.6)
    killers = recording.killers[index]
    explorers = len(frame.robots) - killers
    done = sum(1 for minute, _ in recording.switches if minute <= frame.minute)
    extra = f" · {heading} heading home" if heading else ""
    axes.set_title(
        f"{title}\n{explorers} explorers · {killers} killers · {len(frame.pests)} pests · "
        f"{done} switches{extra}",
        fontsize=9.5,
        loc="left",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=10, help="frames per second; a frame is 20 s")
    parser.add_argument("--save", type=Path, default=Path("videos/rebalancing.mp4"))
    args = parser.parse_args()
    names = [name for name, _, _ in PANELS]
    with ProcessPoolExecutor(max_workers=len(names)) as pool:
        futures = [pool.submit(record, name, args.seed, args.hours) for name in names]
        recordings = [future.result() for future in futures]
    render(recordings, args.save, args.fps)
    print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
