"""A matplotlib renderer behind the ``Renderer`` port.

Draws a read-only ``WorldSnapshot`` as a Python-native figure: one image layer for the field
cells (crop growth, infestation severity, obstacle) and a scatter layer for robots. matplotlib
is an optional technology kept behind the port and imported lazily, so importing this module (or
the ``visualization`` package) does not require matplotlib to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from swarm_field.contracts import BodyType, CellState, WorldSnapshot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

RGB = tuple[float, float, float]
RGBA = tuple[float, float, float, float]

BODY_COLORS: dict[BodyType, str] = {
    BodyType.SCOUT: "royalblue",
    BodyType.KILLER: "crimson",
    BodyType.BATTERY: "goldenrod",
}

_OBSTACLE_RGB: RGB = (0.35, 0.35, 0.35)
_BARE_SOIL_RGB: RGB = (0.85, 0.80, 0.68)
_CROP_RGB: RGB = (0.25, 0.60, 0.25)
_INFESTATION_RGB: RGB = (0.85, 0.10, 0.05)


def cell_rgba_image(snapshot: WorldSnapshot) -> list[list[RGBA]]:
    """Encode a snapshot's cells as an ``(height, width)`` RGBA image.

    Row ``0`` is ``y = 0`` and column ``0`` is ``x = 0``, matching matplotlib's ``origin="lower"``.
    Cells absent from ``snapshot.cells`` are drawn as bare soil, so a sparse or empty snapshot is
    still renderable.
    """
    image: list[list[RGBA]] = []
    for y in range(snapshot.field.height):
        row: list[RGBA] = []
        for x in range(snapshot.field.width):
            row.append(_cell_rgba(snapshot.cells.get((x, y))))
        image.append(row)
    return image


def _cell_rgba(state: CellState | None) -> RGBA:
    """Blend one cell: obstacle, else crop growth over soil, tinted by infestation severity.

    The crop layer is driven by ``growth_stage``. Species-specific colouring is deferred until
    scenarios define ``CropType`` members; until then a mature patch reads as a deeper green.
    """
    if state is not None and state.obstacle:
        return (*_OBSTACLE_RGB, 1.0)
    growth = 0.0 if state is None else state.growth_stage
    base = _lerp(_BARE_SOIL_RGB, _CROP_RGB, growth)
    infestation = 0.0 if state is None else state.infestation
    return (*_lerp(base, _INFESTATION_RGB, infestation), 1.0)


def _lerp(start: RGB, end: RGB, t: float) -> RGB:
    """Linearly interpolate between two colours, clamping ``t`` to ``0..1``."""
    clamped = min(1.0, max(0.0, t))
    return (
        start[0] + (end[0] - start[0]) * clamped,
        start[1] + (end[1] - start[1]) * clamped,
        start[2] + (end[2] - start[2]) * clamped,
    )


class MatplotlibRenderer:
    """Draws a ``WorldSnapshot`` onto a persistent matplotlib figure.

    The figure and axes are created once and redrawn on every ``render`` call; the caller chooses
    how to display them (``plt.show()``, an animation, or saving frames). The axes can also be
    cleared by the caller with ``axes.clear()`` and closed with ``figure``.

    Attributes:
        figure: The matplotlib figure the snapshot is drawn on.
        axes: The single axes the snapshot is drawn on.
        show_targets: Whether to draw a line from each robot to its movement target.
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (9.0, 7.0),
        show_targets: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt

        self.figure: Figure
        self.axes: Axes
        self.figure, self.axes = plt.subplots(figsize=figsize)
        self.show_targets: bool = show_targets

    def render(self, snapshot: WorldSnapshot) -> None:
        """Redraw the axes from ``snapshot``."""
        field = snapshot.field
        self.axes.clear()
        self.axes.imshow(
            cell_rgba_image(snapshot),
            origin="lower",
            interpolation="none",
            extent=(0, field.width, 0, field.height),
        )
        self._draw_robots(snapshot)
        self.axes.set_aspect("equal")
        self.axes.set_xlim(0, field.width)
        self.axes.set_ylim(0, field.height)
        self.axes.set_xlabel("Field cell X")
        self.axes.set_ylabel("Field cell Y")
        self.axes.set_title(
            f"t={snapshot.time} | {field.width}x{field.height} grid | "
            f"{len(snapshot.robots)} robot(s)"
        )

    def _draw_robots(self, snapshot: WorldSnapshot) -> None:
        """Scatter robots by body type and, optionally, link each one to its target."""
        for body in BodyType:
            robots = [robot for robot in snapshot.robots if robot.body is body]
            if not robots:
                continue
            xs = [robot.position[0] + 0.5 for robot in robots]
            ys = [robot.position[1] + 0.5 for robot in robots]
            color = BODY_COLORS[body]
            self.axes.scatter(xs, ys, s=26, color=color, label=f"{body.name} ({len(robots)})")
            if self.show_targets:
                for robot, x, y in zip(robots, xs, ys, strict=True):
                    if robot.target is not None:
                        self.axes.plot(
                            [x, robot.target[0] + 0.5],
                            [y, robot.target[1] + 0.5],
                            color=color,
                            linewidth=0.8,
                            alpha=0.6,
                        )
        if snapshot.robots:
            self.axes.legend(loc="upper right", markerscale=1.5)
