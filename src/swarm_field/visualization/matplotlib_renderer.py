"""A matplotlib renderer behind the ``Renderer`` port.

Draws a read-only ``WorldSnapshot`` as a Python-native figure: one image layer for the field
cells (crop growth, infestation severity, obstacle) and a scatter layer for robots. matplotlib
is an optional technology kept behind the port and imported lazily, so importing this module (or
the ``visualization`` package) does not require matplotlib to be installed.
"""

from __future__ import annotations

from pathlib import Path
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
_FAILED_COLOR = "dimgray"


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

    A robot that has failed is drawn hollow and grey instead of in its body colour, so a
    robustness run shows the swarm losing capacity rather than only reporting it. When the
    snapshot carries a metrics sample, the current figures are written into the title.

    Attributes:
        figure: The matplotlib figure the snapshot is drawn on.
        axes: The single axes the snapshot is drawn on.
        show_targets: Whether to draw a line from each robot to its movement target.
        show_metrics: Whether to write the snapshot's metrics into the title.
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (9.0, 7.0),
        show_targets: bool = True,
        show_metrics: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt

        self.figure: Figure
        self.axes: Axes
        self.figure, self.axes = plt.subplots(figsize=figsize)
        self.show_targets: bool = show_targets
        self.show_metrics: bool = show_metrics

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
        self.axes.set_title(self._title(snapshot))

    def _title(self, snapshot: WorldSnapshot) -> str:
        """The figure title: time, field size, live robot count and the current metrics."""
        field = snapshot.field
        failed = sum(robot.failed for robot in snapshot.robots)
        active = len(snapshot.robots) - failed
        census = f"{active}/{len(snapshot.robots)} active" if failed else f"{len(snapshot.robots)}"
        title = f"t={snapshot.time} | {field.width}x{field.height} grid | {census} robot(s)"
        if not self.show_metrics or snapshot.metrics is None:
            return title
        sample = snapshot.metrics
        return (
            f"{title} | targets={sample.targets_remaining} "
            f"coverage={sample.coverage:.3f} active={sample.active_robots}"
        )

    def _draw_robots(self, snapshot: WorldSnapshot) -> None:
        """Scatter live robots by body type, failed ones hollow, and link each to its target."""
        for body in BodyType:
            robots = [robot for robot in snapshot.robots if robot.body is body]
            live = [robot for robot in robots if not robot.failed]
            dead = [robot for robot in robots if robot.failed]
            for group, color, filled in (
                (live, BODY_COLORS[body], True),
                (dead, _FAILED_COLOR, False),
            ):
                if not group:
                    continue
                self.axes.scatter(
                    [robot.position[0] + 0.5 for robot in group],
                    [robot.position[1] + 0.5 for robot in group],
                    s=26,
                    color=color,
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    linewidths=1.4,
                    label=f"{body.name} ({len(group)})",
                    zorder=2 if filled else 1,
                )
            if self.show_targets:
                for robot in live:
                    if robot.target is not None:
                        self.axes.plot(
                            [
                                robot.position[0] + 0.5,
                                robot.target[0] + 0.5,
                            ],
                            [
                                robot.position[1] + 0.5,
                                robot.target[1] + 0.5,
                            ],
                            color=BODY_COLORS[body],
                            linewidth=0.8,
                            alpha=0.6,
                        )
        if snapshot.robots:
            self.axes.legend(loc="upper right", markerscale=1.5)


class FrameWriter:
    """A ``Renderer`` that writes every snapshot it is handed to a PNG.

    A sweep driven by the runner renders from inside the event loop, which is what makes frames
    rather than a live window the right output: by the time the run returns there is nothing left
    to animate. Wrapping a ``MatplotlibRenderer`` keeps the drawing in one place and lets a whole
    sweep's ticks be captured by passing this where a ``Renderer`` is expected.

    Attributes:
        renderer: The renderer that draws each frame before it is written.
        directory: Directory the frames are written to, created on first write.
        stem: Filename prefix, so several conditions can share one directory.
        frames_written: How many frames have been written so far.
    """

    def __init__(
        self,
        renderer: MatplotlibRenderer,
        directory: Path,
        stem: str,
        dpi: int = 120,
    ) -> None:
        self.renderer = renderer
        self.directory = directory
        self.stem = stem
        self.frames_written = 0
        self._dpi = dpi

    def render(self, snapshot: WorldSnapshot) -> None:
        """Draw the snapshot through the wrapped renderer and save it as the next frame."""
        self.renderer.render(snapshot)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self.stem}_{self.frames_written:04d}.png"
        self.renderer.figure.savefig(path, dpi=self._dpi)
        self.frames_written += 1
