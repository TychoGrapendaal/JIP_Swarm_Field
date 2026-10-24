"""What a study draws over a rendered snapshot: a cell overlay, trails, targets and a side panel.

Each piece draws into the axes or figure of a ``MatplotlibRenderer`` after it rendered a snapshot,
from data handed in by the caller, so none of them knows the simulation or the algorithm:

- a cell overlay: one value from 0 to 1 per cell, e.g. how stale each cell is;
- trails: the last cells each robot stood on, up to where it is now;
- targets: a dashed line from each robot to the cell it is heading for;
- lines, arrows and markers: any of them between points of the field, e.g. a planned route, the
  pull on a robot, or what a robot knows;
- a side panel: lines of text describing the run, e.g. its metrics.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import CellId, RobotId, WorldSnapshot
from swarm_field.visualization.matplotlib_renderer import BODY_COLORS, MatplotlibRenderer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.text import Text

CellOverlay = Callable[[WorldSnapshot], NDArray[np.float64]]
"""One value from 0 to 1 per cell, indexed ``[x, y]``; a cell holding NaN is left undrawn."""
PanelSource = Callable[[WorldSnapshot], Sequence[str]]
"""The lines of the side panel."""


def draw_cell_overlay(
    axes: Axes, snapshot: WorldSnapshot, values: NDArray[np.float64], colormap: str = "viridis"
) -> None:
    """Shade every cell by its value from 0 to 1, over the field and under the robots."""
    field = snapshot.field
    axes.imshow(
        values.T,
        origin="lower",
        interpolation="none",
        extent=(0, field.width, 0, field.height),
        cmap=colormap,
        vmin=0.0,
        vmax=1.0,
        alpha=0.6,
        zorder=0.5,
    )


def draw_targets(axes: Axes, snapshot: WorldSnapshot, targets: Mapping[RobotId, CellId]) -> None:
    """Draw a dashed line from each robot to the cell it is heading for, and mark that cell."""
    for robot in snapshot.robots:
        target = targets.get(robot.robot_id)
        if target is None:
            continue
        color = BODY_COLORS[robot.body]
        x, y = robot.drawn_at()
        axes.plot(
            [x, target[0] + 0.5],
            [y, target[1] + 0.5],
            color=color,
            linestyle="--",
            linewidth=0.8,
        )
        axes.scatter([target[0] + 0.5], [target[1] + 0.5], marker="x", color=color)


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


def draw_polylines(axes: Axes, lines: Sequence[Polyline]) -> None:
    """Draw the lines, above the robots and under the markers and arrows."""
    for line in lines:
        axes.plot(
            [point[0] for point in line.points],
            [point[1] for point in line.points],
            color=line.color,
            linewidth=line.width,
            linestyle=line.style,
            zorder=2.2,
        )


def draw_markers(axes: Axes, markers: Sequence[Marker]) -> None:
    """Draw the markers above the robots, each with its label."""
    for marker in markers:
        x, y = marker.position
        axes.scatter([x], [y], s=marker.size, color=marker.color, edgecolors="black", zorder=2.5)
        if marker.label is not None:
            axes.annotate(
                marker.label,
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                zorder=2.5,
            )


def draw_arrows(axes: Axes, arrows: Sequence[Arrow]) -> None:
    """Draw the arrows above everything else."""
    for arrow in arrows:
        axes.annotate(
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


DEFAULT_TRAIL_LENGTH = 3
"""How many past cells of each robot a trail keeps, unless told otherwise."""


class RobotTrails:
    """Remembers the last cells each robot stood on, and draws them as lines.

    Each line runs through the robot's last cells and ends where the robot is drawn, so it
    follows a robot gliding between two cells.

    Attributes:
        length: How many past cells of each robot are kept and drawn.
    """

    def __init__(self, length: int = DEFAULT_TRAIL_LENGTH) -> None:
        self.length: int = length
        self._trails: dict[RobotId, deque[CellId]] = {}

    def update_and_draw(self, axes: Axes, snapshot: WorldSnapshot) -> None:
        """Add each robot's cell to its trail, if it moved, and draw every trail."""
        for robot in snapshot.robots:
            trail = self._trails.setdefault(robot.robot_id, deque(maxlen=self.length))
            if not trail or trail[-1] != robot.position:
                trail.append(robot.position)
            xs = [cell[0] + 0.5 for cell in trail]
            ys = [cell[1] + 0.5 for cell in trail]
            x, y = robot.drawn_at()
            xs.append(x)
            ys.append(y)
            axes.plot(xs, ys, color=BODY_COLORS[robot.body], alpha=0.5, linewidth=1)


class SidePanel:
    """A column of monospaced text to the right of the field, rewritten on every frame."""

    def __init__(self, renderer: MatplotlibRenderer) -> None:
        figure = renderer.figure
        figure.subplots_adjust(right=0.68)
        self._text: Text = figure.text(
            0.70, 0.95, "", va="top", ha="left", family="monospace", fontsize=9
        )

    def show(self, lines: Sequence[str]) -> None:
        self._text.set_text("\n".join(lines))
