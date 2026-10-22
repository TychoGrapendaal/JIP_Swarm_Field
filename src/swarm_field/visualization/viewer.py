"""An animated view of any running simulation, built on ``MatplotlibRenderer``.

``SimulationViewer`` asks a snapshot source for the world at each frame's time and draws it with
``MatplotlibRenderer``. On top it adds what a study wants to see, each from a callable, so the
viewer knows nothing of the simulation or the algorithm:

- an overlay: one value from 0 to 1 per cell, e.g. how stale each cell is;
- robot positions: where each robot is drawn, e.g. gliding between cells;
- trails: the last cells each robot stood on;
- goals: a dashed line from each robot to the cell it is heading for;
- lines: any lines through points of the field, e.g. a planned route;
- arrows: any arrows between points of the field, e.g. the pull on a robot;
- markers: labelled dots at points of the field, e.g. what a robot knows;
- a text panel: lines describing the run, e.g. its metrics.

The animation opens in a window, or is saved through ``save_frames``: a ``.gif``, an ``.mp4`` or a
directory of PNG frames.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import BodyType, CellId, RobotId, Time, WorldSnapshot
from swarm_field.visualization.matplotlib_renderer import (
    MatplotlibRenderer,
    RobotStyle,
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
TitleSource = Callable[[WorldSnapshot], str]
"""The title of a frame, e.g. with its time."""
PositionSource = Callable[[WorldSnapshot], Mapping[RobotId, tuple[float, float]]]
"""Where each robot is drawn, in cells, e.g. between two cells while it moves."""


@dataclass(frozen=True)
class Arrow:
    """An arrow between two points of the field, in cells; a cell's centre is ``x + 0.5``.

    Attributes:
        start: Where the arrow starts.
        end: Where its head is.
        color: Its colour.
        width: Its line width in points.
        alpha: Its opacity, from 0 to 1.
    """

    start: tuple[float, float]
    end: tuple[float, float]
    color: str
    width: float = 1.0
    alpha: float = 1.0


ArrowSource = Callable[[WorldSnapshot], Sequence[Arrow]]
"""The arrows to draw on a frame."""


@dataclass(frozen=True)
class Marker:
    """A dot at a point of the field, in cells, with an optional label next to it.

    Attributes:
        position: Where the dot is.
        color: Its colour.
        size: Its area in points squared.
        label: Text written next to it, or ``None``.
    """

    position: tuple[float, float]
    color: str
    size: float = 30.0
    label: str | None = None


MarkerSource = Callable[[WorldSnapshot], Sequence[Marker]]
"""The markers to draw on a frame."""


@dataclass(frozen=True)
class Polyline:
    """A line through points of the field, in cells, e.g. a planned route.

    Attributes:
        points: The points it passes through, in order.
        color: Its colour.
        width: Its line width in points.
        style: Its matplotlib line style, e.g. ``"--"`` for dashed.
    """

    points: tuple[tuple[float, float], ...]
    color: str
    width: float = 1.0
    style: str = "-"


PolylineSource = Callable[[WorldSnapshot], Sequence[Polyline]]
"""The lines to draw on a frame."""


class SimulationViewer:
    """Draws frames of a running simulation, with optional overlay, trails, goals and panel.

    A ``title`` is written above the renderer's own title, e.g. to name an experiment in a video;
    it can change per frame.

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
        arrows: ArrowSource | None = None,
        markers: MarkerSource | None = None,
        body_styles: Mapping[BodyType, RobotStyle] | None = None,
        title: str | TitleSource = "",
        positions: PositionSource | None = None,
        lines: PolylineSource | None = None,
    ) -> None:
        self.renderer: MatplotlibRenderer = MatplotlibRenderer(
            figsize, show_targets=False, body_styles=body_styles
        )
        self.trail_length: int = trail_length
        self._source: SnapshotSource = source
        self._overlay: CellOverlay | None = overlay
        self._overlay_label: str = overlay_label
        self._overlay_colormap: str = overlay_colormap
        self._goals: GoalSource | None = goals
        self._panel: PanelSource | None = panel
        self._arrows: ArrowSource | None = arrows
        self._markers: MarkerSource | None = markers
        self._title: str | TitleSource = title
        self._positions: PositionSource | None = positions
        self._lines: PolylineSource | None = lines
        self._trails: dict[RobotId, deque[CellId]] = {}
        self.renderer.figure.subplots_adjust(right=0.68)
        self._panel_text: Text = self.renderer.figure.text(
            0.70, 0.95, "", va="top", ha="left", family="monospace", fontsize=9
        )

    def draw(self, time: Time) -> None:
        """Draw the frame of ``time``."""
        snapshot = self._source(time)
        positions = None
        if self._positions is not None:
            positions = self._positions(snapshot)
        self.renderer.render(snapshot, positions)
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
        self._draw_lines(snapshot)
        self._draw_markers(snapshot)
        self._draw_arrows(snapshot)
        title = self._title
        if not isinstance(title, str):
            title = title(snapshot)
        if title:
            axes.set_title(f"{title}\n{axes.get_title()}")
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
        if self.trail_length == 0:
            return
        for robot in snapshot.robots:
            trail = self._trails.setdefault(robot.robot_id, deque(maxlen=self.trail_length))
            if not trail or trail[-1] != robot.position:
                trail.append(robot.position)
            xs = [cell[0] + 0.5 for cell in trail]
            ys = [cell[1] + 0.5 for cell in trail]
            color = self.renderer.body_styles[robot.body].color
            self.renderer.axes.plot(xs, ys, color=color, alpha=0.5, linewidth=1)

    def _draw_goals(self, snapshot: WorldSnapshot) -> None:
        """Draw a dashed line from each robot to the cell it is heading for."""
        if self._goals is None:
            return
        goals = self._goals()
        for robot in snapshot.robots:
            goal = goals.get(robot.robot_id)
            if goal is None:
                continue
            color = self.renderer.body_styles[robot.body].color
            self.renderer.axes.plot(
                [robot.position[0] + 0.5, goal[0] + 0.5],
                [robot.position[1] + 0.5, goal[1] + 0.5],
                color=color,
                linestyle="--",
                linewidth=0.8,
            )
            self.renderer.axes.scatter([goal[0] + 0.5], [goal[1] + 0.5], marker="x", color=color)

    def _draw_lines(self, snapshot: WorldSnapshot) -> None:
        """Draw the lines of the frame, under the markers and arrows."""
        if self._lines is None:
            return
        for line in self._lines(snapshot):
            self.renderer.axes.plot(
                [point[0] for point in line.points],
                [point[1] for point in line.points],
                color=line.color,
                linewidth=line.width,
                linestyle=line.style,
                zorder=2.2,
            )

    def _draw_markers(self, snapshot: WorldSnapshot) -> None:
        """Draw the markers of the frame, above the robots, each with its label."""
        if self._markers is None:
            return
        axes = self.renderer.axes
        for marker in self._markers(snapshot):
            x, y = marker.position
            axes.scatter(
                [x], [y], s=marker.size, color=marker.color, edgecolors="black", zorder=2.5
            )
            if marker.label is not None:
                axes.annotate(
                    marker.label,
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    zorder=2.5,
                )

    def _draw_arrows(self, snapshot: WorldSnapshot) -> None:
        """Draw the arrows of the frame, above the robots."""
        if self._arrows is None:
            return
        for arrow in self._arrows(snapshot):
            self.renderer.axes.annotate(
                "",
                xy=arrow.end,
                xytext=arrow.start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": arrow.color,
                    "lw": arrow.width,
                    "alpha": arrow.alpha,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=3,
            )
