"""What a study draws over a rendered snapshot: a cell overlay, trails, targets and a side panel.

Each piece draws into the axes or figure of a ``MatplotlibRenderer`` after it rendered a snapshot,
from data handed in by the caller, so none of them knows the simulation or the algorithm:

- a cell overlay: one value from 0 to 1 per cell, e.g. how stale each cell is;
- trails: the last cells each robot stood on;
- targets: a dashed line from each robot to the cell it is heading for;
- a side panel: lines of text describing the run, e.g. its metrics.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
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
        axes.plot(
            [robot.position[0] + 0.5, target[0] + 0.5],
            [robot.position[1] + 0.5, target[1] + 0.5],
            color=color,
            linestyle="--",
            linewidth=0.8,
        )
        axes.scatter([target[0] + 0.5], [target[1] + 0.5], marker="x", color=color)


class RobotTrails:
    """Remembers the last cells each robot stood on, and draws them as lines.

    Attributes:
        length: How many past cells of each robot are kept and drawn.
    """

    def __init__(self, length: int) -> None:
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
