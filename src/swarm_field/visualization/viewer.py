"""An animated view of any running simulation, built on ``MatplotlibRenderer``.

``SimulationViewer`` asks a snapshot source for the world at each frame's time and draws it with
``MatplotlibRenderer``. On top it adds what a study wants to see, each from a callable, so the
viewer knows nothing of the simulation or the algorithm:

- an overlay: one value from 0 to 1 per cell, e.g. how stale each cell is;
- trails: the last cells each robot stood on;
- goals: a dashed line from each robot to the cell it is heading for;
- a text panel: lines describing the run, e.g. its metrics.

The animation opens in a window, or is saved through ``save_frames``: a ``.gif``, an ``.mp4`` or a
directory of PNG frames.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import CellId, RobotId, Time, WorldSnapshot
from swarm_field.visualization.matplotlib_renderer import (
    BODY_COLORS,
    MatplotlibRenderer,
    save_frames,
)

if TYPE_CHECKING:
    from matplotlib.text import Text

SnapshotSource = Callable[[Time], WorldSnapshot]
"""The world at a time, e.g. after running the simulation up to it."""
CellOverlay = Callable[[WorldSnapshot], NDArray[np.float64]]
"""One value from 0 to 1 per cell, indexed ``[x, y]``; a cell holding NaN is left undrawn."""
GoalSource = Callable[[], Mapping[RobotId, CellId]]
"""The cell each robot is heading for."""
PanelSource = Callable[[WorldSnapshot], Sequence[str]]
"""The lines of the text panel."""


class SimulationViewer:
    """Draws frames of a running simulation, with optional overlay, trails, goals and panel.

    Attributes:
        renderer: Draws the snapshot itself; its figure and axes are the viewer's.
        trail_length: How many past cells of each robot are drawn.
    """

    def __init__(
        self,
        source: SnapshotSource,
        overlay: CellOverlay | None = None,
        overlay_label: str = "",
        overlay_colormap: str = "viridis",
        goals: GoalSource | None = None,
        panel: PanelSource | None = None,
        trail_length: int = 30,
        figsize: tuple[float, float] = (12.0, 8.0),
    ) -> None:
        self.renderer: MatplotlibRenderer = MatplotlibRenderer(figsize, show_targets=False)
        self.trail_length: int = trail_length
        self._source: SnapshotSource = source
        self._overlay: CellOverlay | None = overlay
        self._overlay_label: str = overlay_label
        self._overlay_colormap: str = overlay_colormap
        self._goals: GoalSource | None = goals
        self._panel: PanelSource | None = panel
        self._trails: dict[RobotId, deque[CellId]] = {}
        self.renderer.figure.subplots_adjust(right=0.68)
        self._panel_text: Text = self.renderer.figure.text(
            0.70, 0.95, "", va="top", ha="left", family="monospace", fontsize=9
        )

    def draw(self, time: Time) -> None:
        """Draw the frame of ``time``."""
        snapshot = self._source(time)
        self.renderer.render(snapshot)
        axes = self.renderer.axes
        field = snapshot.field
        if self._overlay is not None:
            values = self._overlay(snapshot)
            axes.imshow(
                values.T,
                origin="lower",
                interpolation="none",
                extent=(0, field.width, 0, field.height),
                cmap=self._overlay_colormap,
                vmin=0.0,
                vmax=1.0,
                alpha=0.6,
                zorder=0.5,
            )
            if self._overlay_label:
                axes.set_title(f"{axes.get_title()}\noverlay: {self._overlay_label}")
        self._draw_trails(snapshot)
        self._draw_goals(snapshot)
        if self._panel is not None:
            self._panel_text.set_text("\n".join(self._panel(snapshot)))

    def animate(self, frame_times: Sequence[Time], fps: int = 20, save: Path | None = None) -> None:
        """Play the frames in a window, or save them to ``save`` with ``save_frames``."""

        def draw_frame(index: int) -> None:
            self.draw(frame_times[index])

        if save is not None:
            save_frames(self.renderer.figure, draw_frame, len(frame_times), save, fps=fps)
            return
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        animation = FuncAnimation(
            self.renderer.figure,
            draw_frame,
            frames=len(frame_times),
            interval=1000 / fps,
            repeat=False,
        )
        _ = animation
        plt.show()

    def _draw_trails(self, snapshot: WorldSnapshot) -> None:
        """Remember each robot's cell and draw its last ``trail_length`` cells."""
        for robot in snapshot.robots:
            trail = self._trails.setdefault(robot.robot_id, deque(maxlen=self.trail_length))
            if not trail or trail[-1] != robot.position:
                trail.append(robot.position)
            xs = [cell[0] + 0.5 for cell in trail]
            ys = [cell[1] + 0.5 for cell in trail]
            self.renderer.axes.plot(xs, ys, color=BODY_COLORS[robot.body], alpha=0.5, linewidth=1)

    def _draw_goals(self, snapshot: WorldSnapshot) -> None:
        """Draw a dashed line from each robot to the cell it is heading for."""
        if self._goals is None:
            return
        goals = self._goals()
        for robot in snapshot.robots:
            goal = goals.get(robot.robot_id)
            if goal is None:
                continue
            self.renderer.axes.plot(
                [robot.position[0] + 0.5, goal[0] + 0.5],
                [robot.position[1] + 0.5, goal[1] + 0.5],
                color=BODY_COLORS[robot.body],
                linestyle="--",
                linewidth=0.8,
            )
            self.renderer.axes.scatter(
                [goal[0] + 0.5], [goal[1] + 0.5], marker="x", color=BODY_COLORS[robot.body]
            )
