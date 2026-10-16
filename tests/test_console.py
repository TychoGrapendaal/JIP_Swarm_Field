"""Tests for the interactive console: time controls, clicks, the run's horizon and the extras."""

import numpy as np
import pytest
from numpy.typing import NDArray

from swarm_field.console import BASE_STEP_MS, InteractiveConsole, keys_to_command
from swarm_field.contracts import (
    CellId,
    CommandLog,
    Pause,
    SetTimeScale,
    StepOnce,
    TimeUnit,
    WorldSnapshot,
)
from swarm_field.experiments import LiveRun, build_named_experiment
from swarm_field.visualization import MatplotlibRenderer

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _console(until: int = TimeUnit.MINUTE, log: CommandLog | None = None) -> InteractiveConsole:
    """The S1 preset from seed 1, on a console."""
    run = LiveRun.from_config(build_named_experiment("S1", 1, until), log)
    return InteractiveConsole(run, MatplotlibRenderer())


def _empty_cell(console: InteractiveConsole) -> CellId:
    """A cell a robot could enter but none stands on."""
    world = console.run.context.world
    for cell in sorted(world.cells):
        if world.is_walkable(cell) and not world.robots_at(cell):
            return cell
    raise AssertionError("the field has no empty cell")


def _click(console: InteractiveConsole, cell: CellId) -> None:
    console.apply_click(cell[0] + 0.5, cell[1] + 0.5)


def test_keys_map_to_the_time_commands_within_the_speed_bounds() -> None:
    assert keys_to_command(" ", 1.0) == Pause()
    assert keys_to_command("s", 1.0) == StepOnce()
    assert keys_to_command("+", 2.0) == SetTimeScale(factor=4.0)
    assert keys_to_command("-", 0.25) == SetTimeScale(factor=0.25)
    assert keys_to_command("x", 1.0) is None


def test_pause_stops_the_clock_and_step_advances_exactly_one_event() -> None:
    console = _console()
    console.advance()
    assert console.time == BASE_STEP_MS

    console.apply_time_command(Pause())
    console.advance()
    assert console.time == BASE_STEP_MS

    next_event = console.run.context.queue.peek()
    assert next_event is not None
    console.apply_time_command(StepOnce())
    console.advance()
    assert console.time == next_event.time
    assert console.paused


def test_the_run_stops_at_its_horizon_and_counts_the_frames_to_it() -> None:
    console = _console(until=1_200)
    assert console.frames_to_horizon() == 3

    for _ in range(5):
        console.advance()

    assert console.time == 1_200
    assert console.finished


def test_a_click_on_an_empty_cell_injects_and_the_run_records_it() -> None:
    console = _console()
    console.advance()
    cell = _empty_cell(console)

    _click(console, cell)

    assert console.run.context.world.cell(cell).infestation > 0.0
    (recorded,) = console.run.recorded
    assert (recorded.time, recorded.command.cell) == (BASE_STEP_MS, cell)


def test_a_click_on_a_robot_toggles_its_inspection_and_draws_it() -> None:
    console = _console()
    robots = console.run.context.world.robots
    robot = robots[min(robots)]

    selections: list[int | None] = []
    for _ in range(2):
        _click(console, robot.position)
        console.draw()
        selections.append(console.selected)

    assert selections == [robot.robot_id, None]


def test_a_replay_refuses_new_injections() -> None:
    console = _console(log=CommandLog(seed=1, until=TimeUnit.MINUTE, commands=(), scenario="S1"))
    cell = _empty_cell(console)

    _click(console, cell)

    assert console.run.context.world.cell(cell).infestation == 0.0
    assert console.run.recorded == []


def test_a_frame_draws_the_overlay_trails_targets_and_side_panel() -> None:
    run = LiveRun.from_config(build_named_experiment("S1", 1, TimeUnit.MINUTE))

    def half(snapshot: WorldSnapshot) -> NDArray[np.float64]:
        return np.full((snapshot.field.width, snapshot.field.height), 0.5)

    def panel(snapshot: WorldSnapshot) -> list[str]:
        return [f"t = {snapshot.time}"]

    console = InteractiveConsole(
        run,
        MatplotlibRenderer(show_targets=False),
        overlay=half,
        overlay_label="half",
        panel=panel,
        trail_length=5,
        show_targets=True,
    )
    console.advance(render=True)
    console.advance(render=True)

    axes = console.renderer.axes
    assert console.renderer.figure.texts[0].get_text() == f"t = {2 * BASE_STEP_MS}"
    assert "overlay: half" in axes.get_title()
    # One trail per robot, and one dashed line per robot that has a target.
    trails = len(run.context.world.robots)
    assert len(axes.lines) == trails + len(run.targets())
