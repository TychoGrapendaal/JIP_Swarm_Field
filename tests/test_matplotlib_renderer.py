"""Tests for the matplotlib renderer behind the ``Renderer`` port."""

from pathlib import Path

import pytest

from swarm_field.contracts import BodyType, CellState, FieldGeometry, TimeUnit, Treat, WorldSnapshot
from swarm_field.simulation import RobotState, WorldState
from swarm_field.visualization import MatplotlibRenderer, save_frames
from swarm_field.visualization.matplotlib_renderer import cell_rgba_image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _field(width: int = 2, height: int = 2) -> FieldGeometry:
    return FieldGeometry(
        width=width,
        height=height,
        cell_size=5.0,
        row_sp=2.0,
        in_row_sp=1.0,
        headland=5.0,
    )


def test_cell_rgba_image_encodes_crop_infestation_and_obstacle() -> None:
    snapshot = WorldSnapshot(
        time=0,
        field=_field(),
        cells={
            (0, 0): CellState(crop=None, growth_stage=0.0, infestation=0.0, obstacle=True),
            (1, 0): CellState(crop=None, growth_stage=1.0, infestation=0.0, obstacle=False),
            (0, 1): CellState(crop=None, growth_stage=0.0, infestation=0.0, obstacle=False),
            (1, 1): CellState(crop=None, growth_stage=0.0, infestation=1.0, obstacle=False),
        },
        robots=[],
    )

    image = cell_rgba_image(snapshot)

    assert image[0][0] == pytest.approx((0.35, 0.35, 0.35, 1.0))
    assert image[0][1] == pytest.approx((0.47, 0.72, 0.47, 1.0))
    assert image[1][0] == pytest.approx((0.85, 0.80, 0.68, 1.0))
    assert image[1][1] == pytest.approx((0.85, 0.10, 0.05, 1.0))


def test_render_draws_the_grid_and_one_scatter_per_body() -> None:
    field = _field(width=6, height=4)
    robots = [
        RobotState(robot_id=1, position=(1, 1), body=BodyType.SCOUT, speed_mps=2.0),
        RobotState(robot_id=2, position=(4, 2), body=BodyType.KILLER, speed_mps=1.0),
    ]
    world = WorldState(field, robots=robots)
    world.cell((3, 1)).infestation = 1.0
    world.cell((5, 3)).obstacle = True

    renderer = MatplotlibRenderer()
    renderer.render(world.snapshot(0))

    images = renderer.axes.get_images()
    assert len(images) == 1
    assert images[0].get_array().shape[:2] == (field.height, field.width)
    assert len(renderer.axes.collections) == 2


def test_render_tracks_an_infestation_as_it_is_treated() -> None:
    field = _field(width=6, height=4)
    robots = [RobotState(robot_id=1, position=(3, 1), body=BodyType.KILLER, speed_mps=1.0)]
    world = WorldState(field, robots=robots)
    world.cell((3, 1)).infestation = 1.0
    renderer = MatplotlibRenderer()

    renderer.render(world.snapshot(0))
    before = renderer.axes.get_images()[0].get_array().copy()

    world.apply_action(1, Treat(cell=(3, 1)), time=100)
    renderer.render(world.snapshot(30 * TimeUnit.MINUTE + 100))
    after = renderer.axes.get_images()[0].get_array()

    assert not (before == after).all()
    assert world.snapshot(100).cells[(3, 1)].infestation < 1.0


@pytest.mark.parametrize("name", ["run.gif", "frames"])
def test_save_frames_draws_every_frame_into_a_video_or_png_directory(
    tmp_path: Path, name: str
) -> None:
    renderer = MatplotlibRenderer()
    world = WorldState(_field())
    drawn: list[int] = []

    def draw_frame(index: int) -> None:
        drawn.append(index)
        renderer.render(world.snapshot(index * 1_000))

    save_frames(renderer.figure, draw_frame, 3, tmp_path / name)

    assert len(drawn) >= 3
    if name == "frames":
        assert len(list((tmp_path / name).glob("*.png"))) == 3
    else:
        assert (tmp_path / name).stat().st_size > 0


def test_render_draws_a_line_on_every_cell_border() -> None:
    field = _field(width=3, height=2)
    renderer = MatplotlibRenderer()

    renderer.render(WorldState(field).snapshot(time=0))

    assert list(renderer.axes.xaxis.get_minorticklocs()) == [0, 1, 2, 3]
    assert list(renderer.axes.yaxis.get_minorticklocs()) == [0, 1, 2]
    assert all(tick.gridline.get_visible() for tick in renderer.axes.xaxis.get_minor_ticks())
