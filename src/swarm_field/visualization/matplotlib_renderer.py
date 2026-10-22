"""A matplotlib renderer behind the ``Renderer`` port.

Draws a read-only ``WorldSnapshot`` as a Python-native figure: one image layer for the field
cells (crop growth, infestation severity, obstacle) and a scatter layer for robots. matplotlib
is an optional technology kept behind the port and imported lazily, so importing this module (or
the ``visualization`` package) does not require matplotlib to be installed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from swarm_field.contracts import BodyType, CellState, RobotId, RobotPublicState, WorldSnapshot

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


@dataclass(frozen=True)
class RobotStyle:
    """How the robots of one body are drawn.

    Attributes:
        color: The marker colour.
        size: The marker area in points squared.
        alpha: The marker opacity, from 0 to 1.
        label: The legend name, or ``None`` for the body's name.
    """

    color: str
    size: float = 26.0
    alpha: float = 1.0
    label: str | None = None


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
        body_styles: How each body's robots are drawn; a body without a style is drawn in its
            ``BODY_COLORS`` colour.
        show_grid: Whether to draw thin lines between the cells, so the grid is visible.
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (9.0, 7.0),
        show_targets: bool = True,
        show_metrics: bool = True,
        body_styles: Mapping[BodyType, RobotStyle] | None = None,
        show_grid: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt

        self.figure: Figure
        self.axes: Axes
        self.figure, self.axes = plt.subplots(figsize=figsize)
        self.show_targets: bool = show_targets
        self.show_metrics: bool = show_metrics
        self.body_styles: dict[BodyType, RobotStyle] = {}
        for body in BodyType:
            self.body_styles[body] = RobotStyle(BODY_COLORS[body])
        if body_styles is not None:
            self.body_styles.update(body_styles)
        self.show_grid: bool = show_grid

    def render(
        self,
        snapshot: WorldSnapshot,
        positions: Mapping[RobotId, tuple[float, float]] | None = None,
    ) -> None:
        """Redraw the axes from ``snapshot``.

        ``positions`` places robots at points of the field, in cells, instead of their cells'
        centres, e.g. so that a moving robot glides; a robot missing from it sits on its cell.
        """
        field = snapshot.field
        self.axes.clear()
        self.axes.imshow(
            cell_rgba_image(snapshot),
            origin="lower",
            interpolation="none",
            extent=(0, field.width, 0, field.height),
        )
        self._draw_robots(snapshot, positions or {})
        self.axes.set_aspect("equal")
        self.axes.set_xlim(0, field.width)
        self.axes.set_ylim(0, field.height)
        if self.show_grid:
            self._draw_grid(field.width, field.height)
        self.axes.set_xlabel("Field cell X")
        self.axes.set_ylabel("Field cell Y")
        self.axes.set_title(self._title(snapshot))

    def _draw_grid(self, width: int, height: int) -> None:
        """Draw thin lines on every cell border, over the field but under the robots.

        They are the axes' minor grid lines, one per cell border, so they need no artist per cell.
        Matplotlib drops a minor tick where a major one sits, which would leave out every border
        that carries a number, so that is switched off.
        """
        self.axes.xaxis.remove_overlapping_locs = False
        self.axes.yaxis.remove_overlapping_locs = False
        self.axes.set_xticks(range(width + 1), minor=True)
        self.axes.set_yticks(range(height + 1), minor=True)
        self.axes.tick_params(which="minor", length=0)
        self.axes.grid(which="minor", color="black", linewidth=0.3, alpha=0.35)
        self.axes.set_axisbelow(True)

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

    def _draw_robots(
        self, snapshot: WorldSnapshot, positions: Mapping[RobotId, tuple[float, float]]
    ) -> None:
        """Scatter live robots by body type, failed ones hollow, and link each to its target."""
        for body in BodyType:
            style = self.body_styles[body]
            robots = [robot for robot in snapshot.robots if robot.body is body]
            live = [robot for robot in robots if not robot.failed]
            dead = [robot for robot in robots if robot.failed]
            name = style.label if style.label is not None else body.name
            for group, color, filled in (
                (live, style.color, True),
                (dead, _FAILED_COLOR, False),
            ):
                if not group:
                    continue
                points = [_robot_point(robot, positions) for robot in group]
                self.axes.scatter(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    s=style.size,
                    color=color,
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    linewidths=1.4,
                    alpha=style.alpha,
                    label=f"{name} ({len(group)})",
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
                            color=style.color,
                            linewidth=0.8,
                            alpha=0.6,
                        )
        if snapshot.robots:
            self.axes.legend(loc="upper right", markerscale=1.5)


def _robot_point(
    robot: RobotPublicState, positions: Mapping[RobotId, tuple[float, float]]
) -> tuple[float, float]:
    """Where a robot is drawn: its given position, else its cell's centre."""
    point = positions.get(robot.robot_id)
    if point is None:
        return (robot.position[0] + 0.5, robot.position[1] + 0.5)
    return point


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


VIDEO_SUFFIXES: tuple[str, ...] = (".gif", ".mp4")
"""Suffixes ``save_frames`` writes as one video file rather than a directory of PNGs."""


def save_frames(
    figure: Figure,
    draw_frame: Callable[[int], None],
    frames: int,
    path: Path,
    fps: int = 16,
    dpi: int = 120,
) -> None:
    """Draw ``frames`` frames into ``figure`` without a window and write them to ``path``.

    A ``.gif`` path is written with Pillow and an ``.mp4`` path with ffmpeg, which must be on the
    ``PATH``; any other path is a directory that receives one PNG per frame.

    Args:
        figure: The figure ``draw_frame`` draws into.
        draw_frame: Draws frame ``index`` into the figure, advancing whatever it shows.
        frames: How many frames to draw.
        path: The video file, or the directory for the PNGs; parents are created.
        fps: Frames per second of a video.
        dpi: Resolution of the PNG frames.
    """
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    if path.suffix in VIDEO_SUFFIXES:
        path.parent.mkdir(parents=True, exist_ok=True)
        animation = FuncAnimation(figure, draw_frame, frames=frames, repeat=False)
        writer = PillowWriter(fps=fps)
        if path.suffix == ".mp4":
            writer = FFMpegWriter(fps=fps)
        animation.save(str(path), writer=writer)
        return
    path.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        draw_frame(index)
        figure.savefig(path / f"frame_{index:04d}.png", dpi=dpi)
