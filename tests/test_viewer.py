"""Tests for the simulation viewer: overlays, trails, goals and the panel on top of a snapshot."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from swarm_field.contracts import BodyType, CellId, FieldGeometry, RobotId, WorldSnapshot
from swarm_field.simulation import RobotState, WorldState
from swarm_field.visualization import SimulationViewer

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _world() -> WorldState:
    field = FieldGeometry(
        width=6, height=4, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0
    )
    robots = [
        RobotState(robot_id=1, position=(0, 0), body=BodyType.SCOUT, speed_mps=1.0),
        RobotState(robot_id=2, position=(5, 3), body=BodyType.KILLER, speed_mps=1.0),
    ]
    return WorldState(field, robots=robots)


def _viewer(world: WorldState) -> SimulationViewer:
    def overlay(snapshot: WorldSnapshot) -> NDArray[np.float64]:
        return np.full((snapshot.field.width, snapshot.field.height), 0.5)

    def goals() -> dict[RobotId, CellId]:
        return {1: (3, 2)}

    def panel(snapshot: WorldSnapshot) -> list[str]:
        return [f"t = {snapshot.time}"]

    return SimulationViewer(
        world.snapshot, overlay=overlay, overlay_label="half", goals=goals, panel=panel
    )


def test_a_frame_draws_the_trails_goals_and_panel() -> None:
    world = _world()
    viewer = _viewer(world)

    viewer.draw(0)
    world.robots[1].position = (1, 0)
    viewer.draw(10)

    assert viewer.renderer.figure.texts[0].get_text() == "t = 10"
    # A trail per robot, and one goal line.
    assert len(viewer.renderer.axes.lines) == 3
    assert "overlay: half" in viewer.renderer.axes.get_title()


def test_frames_are_saved_as_a_gif(tmp_path: Path) -> None:
    viewer = _viewer(_world())
    path = tmp_path / "run.gif"

    viewer.animate([0, 10], fps=5, save=path)

    assert path.stat().st_size > 0
