"""Film runs of a pest batch: run them again and render each into a video in its folder.

A run is a function of its arm and seed, so running it again gives exactly the run of the batch.
``record_videos`` takes runs from ``runs_for``, simulates and renders them in parallel, and writes
each to ``video/<arm>_seed<seed>.mp4`` in an experiment's folder, next to its ``results.json``::

    experiment = Experiment("seeded_vs_hop_small")
    results = experiment.run(runs_for(specs, seeds=range(12)), measure_pest_response)
    record_videos(experiment.directory, runs_for({"hop-tuned": specs["hop-tuned"]}, seeds=[0]))

A video steps its run as ``measure_pest_response`` does, one decision interval per frame, and
draws the ground truth with ``MatplotlibRenderer``: pests in red, from light at a new pest to dark
at a full one, killers as big red dots with their trails and targets, relays as small grey dots,
and a side panel with the time and the measures so far. An mp4 needs
ffmpeg on the ``PATH``.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import BodyType, Time, TimeUnit, WorldSnapshot
from swarm_field.experiments.batch import Run, run_in_parallel
from swarm_field.experiments.pest_response import MeasuredPestRun, PestResponse, PestRunSpec
from swarm_field.experiments.presets import EXPLORER, RELAY
from swarm_field.visualization import (
    MatplotlibRenderer,
    RobotStyle,
    RobotTrails,
    SidePanel,
    draw_cell_overlay,
    draw_targets,
    save_frames,
)

logger = logging.getLogger(__name__)

VIDEO_FOLDER = "video"
"""The folder of an experiment that holds its videos."""
FPS = 16
"""Frames per second: one frame is one decision interval, so 15 s of a run is 1/16 s of video."""
TRAIL_LENGTH = 20
"""How many past cells of each robot are drawn."""
RELAY_STYLE = RobotStyle("slategray", size=40.0, alpha=0.6, label="relay")
KILLER_STYLE = RobotStyle("red", size=160.0, label="killer")


def video_path(directory: Path, run: Run[Any]) -> Path:
    """Where the video of ``run`` goes in an experiment's folder: ``video/<arm>_seed<seed>.mp4``.

    The run's name, ``ARM/seedSEED``, with every ``/`` turned into ``_``.
    """
    return directory / VIDEO_FOLDER / f"{run.name.replace('/', '_')}.mp4"


def record_videos(
    directory: Path,
    runs: Sequence[Run[PestRunSpec]],
    processes: int | None = None,
    fps: int = FPS,
) -> list[Path]:
    """Run every run again and render it into a video in ``directory``, in parallel processes.

    Args:
        directory: The experiment's folder, e.g. ``Experiment.directory`` or a folder of
            ``output/`` made earlier.
        runs: The runs to film, e.g. from ``runs_for`` with the arms and seeds to look at.
        processes: How many processes, or ``None`` for one per CPU.
        fps: Frames per second of the videos.

    Returns:
        Where each run's video was written, in the order of ``runs``.

    Raises:
        ValueError: If two runs share a name.
    """
    logger.info("Recording %d videos into %s", len(runs), directory / VIDEO_FOLDER)
    results = run_in_parallel(runs, _VideoRecorder(directory, fps), processes)
    return [result.result for result in results]


def record_video(run: Run[PestRunSpec], path: Path, fps: int = FPS) -> PestResponse:
    """Run ``run``, render one frame per decision interval into ``path``, and measure it.

    A ``.gif`` is written with Pillow, an ``.mp4`` with ffmpeg, and any other path becomes a
    directory of PNG frames.

    Returns:
        What the run measured; the same as ``measure_pest_response`` gives, but for ``wall``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measured = MeasuredPestRun(run)
    styles = {BodyType.KILLER: KILLER_STYLE}
    scenario = run.config.scenario
    if scenario.count(RELAY.name) > 0 and scenario.count(EXPLORER.name) == 0:
        styles[BodyType.SCOUT] = RELAY_STYLE
    renderer = MatplotlibRenderer(
        (12.0, 8.0), show_targets=False, show_metrics=False, body_styles=styles
    )
    panel = SidePanel(renderer)
    trails = RobotTrails(TRAIL_LENGTH)

    def draw_frame(index: int) -> None:
        # The animation draws its first frame twice, so a frame advances only to its own time.
        if measured.live.now < (index + 1) * measured.step:
            measured.advance()
        snapshot = measured.live.snapshot()
        renderer.render(snapshot)
        draw_cell_overlay(renderer.axes, snapshot, pest_overlay(snapshot), "Reds")
        renderer.axes.set_title(f"{run.name}  |  {_clock(snapshot.time, measured.live.horizon)}")
        trails.update_and_draw(renderer.axes, snapshot)
        draw_targets(renderer.axes, snapshot, measured.live.targets())
        panel.show(panel_lines(measured, snapshot))

    save_frames(renderer.figure, draw_frame, measured.steps, path, fps=fps)
    plt.close(renderer.figure)
    logger.info("Wrote the video of %s to %s", run.name, path)
    return measured.response()


def pest_overlay(snapshot: WorldSnapshot) -> NDArray[np.float64]:
    """Every pest cell, from 0.6 at a new pest to 1 at a full one; other cells undrawn.

    A new pest barely tints its cell in the renderer, so this marks it clearly.
    """
    field = snapshot.field
    values = np.full((field.width, field.height), np.nan)
    for cell, state in snapshot.cells.items():
        if state.infestation > 0.0:
            values[cell] = 0.6 + 0.4 * state.infestation
    return values


def panel_lines(measured: MeasuredPestRun, snapshot: WorldSnapshot) -> list[str]:
    """The side panel: the run's setup, the pests now and so far, and what the killers do.

    The run's name is in the title, which is wider.
    """
    spec = measured.run.config
    scenario = spec.scenario
    tracker = measured.tracker
    lines = [f"time       {_clock(snapshot.time, measured.live.horizon)}", "robots"]
    for group in scenario.swarm:
        if group.count > 0:
            lines.append(f"  {group.count:3d}  {group.preset.name}")
    lines.extend(
        [
            f"radio      {scenario.robots.radio_range_m:g} m",
            f"pest rate  {spec.pests_per_hour:g} per hour",
            "",
            f"infested   {len(tracker.appeared)} cells now",
            f"treated    {len(tracker.responses)} pests so far",
        ]
    )
    if tracker.responses:
        minutes = sum(tracker.responses) / len(tracker.responses) / TimeUnit.MINUTE
        lines.append(f"response   {minutes:.1f} min (mean)")
    lines.append("")
    goals: Counter[str] = Counter()
    doses = scenario.pests.treatment_doses
    doses_used = 0.0
    agents = measured.live.agents()
    world = measured.live.context.world
    for robot_id in measured.killers:
        goals[agents[robot_id].inspect().current_goal or "idle"] += 1
        doses_used += (1.0 - world.robots[robot_id].consumable) * doses
    lines.append("killers by goal")
    for goal, count in sorted(goals.items()):
        lines.append(f"  {count:3d}  {goal}")
    lines.append(f"doses used {doses_used:.0f} of {doses * len(measured.killers)}")
    return lines


class _VideoRecorder:
    """Films one run into its video in an experiment's folder; it pickles, for the workers."""

    def __init__(self, directory: Path, fps: int) -> None:
        self._directory: Path = directory
        self._fps: int = fps

    def __call__(self, run: Run[PestRunSpec]) -> Path:
        """Record the video of ``run`` and return where it went."""
        path = video_path(self._directory, run)
        record_video(run, path, self._fps)
        return path


def _clock(time: Time, horizon: Time) -> str:
    """A run's time and its horizon as hours and minutes, e.g. ``1:05 of 3:00``."""
    return f"{_hours_and_minutes(time)} of {_hours_and_minutes(horizon)}"


def _hours_and_minutes(time: Time) -> str:
    minutes = time // TimeUnit.MINUTE
    return f"{minutes // 60}:{minutes % 60:02d}"
