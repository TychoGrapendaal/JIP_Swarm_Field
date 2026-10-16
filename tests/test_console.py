"""Tests for the interactive console: time controls, clicks, recording and its horizon."""

from random import Random

import pytest

from swarm_field.console import BASE_STEP_MS, InteractiveConsole, keys_to_command
from swarm_field.contracts import (
    AgentInspection,
    BodyType,
    CommandLog,
    Decision,
    FieldGeometry,
    Idle,
    InjectInfestation,
    Pause,
    Perception,
    SetTimeScale,
    StepOnce,
    TimedCommand,
)
from swarm_field.simulation import (
    CommandReplay,
    RobotDecision,
    RobotState,
    SimulationContext,
    WorldState,
)
from swarm_field.visualization import MatplotlibRenderer

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


class _IdleAgent:
    def __init__(self, robot_id: int) -> None:
        self.robot_id: int = robot_id

    def decide(self, perception: Perception) -> Decision:
        return Decision(action=Idle())

    def inspect(self) -> AgentInspection:
        return AgentInspection(robot_id=self.robot_id, current_goal="idle", belief=())


def _console(
    horizon: int | None = None, log: CommandLog | None = None, replaying: bool = False
) -> InteractiveConsole:
    """One idle robot on (1, 1) of a 5 by 5 field, deciding every second."""
    field = FieldGeometry(width=5, height=5, cell_size=1.0, row_sp=1.0, in_row_sp=1.0, headland=0.0)
    robot = RobotState(robot_id=0, position=(1, 1), body=BodyType.SCOUT, speed_mps=1.0)
    ctx = SimulationContext(
        now=0, rng=Random(0), agents={0: _IdleAgent(0)}, world=WorldState(field, robots=[robot])
    )
    ctx.schedule(RobotDecision(time=0, robot=0, decision_interval=1_000))
    if log is None:
        log = CommandLog(seed=0, until=0, commands=(), scenario="S1")
    return InteractiveConsole(
        CommandReplay(ctx, log),
        MatplotlibRenderer(),
        label="test",
        horizon=horizon,
        replaying=replaying,
    )


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

    console.apply_time_command(StepOnce())
    console.advance()
    assert console.time == 1_000
    assert console.paused


def test_the_run_stops_at_its_horizon_and_counts_the_frames_to_it() -> None:
    console = _console(horizon=1_200)
    assert console.frames_to_horizon() == 3

    for _ in range(5):
        console.advance()

    assert console.time == 1_200
    assert console.finished


def test_a_click_injects_on_an_empty_cell_and_records_it_with_its_time() -> None:
    console = _console()
    console.advance()

    console.apply_click(3.5, 2.5)

    assert console.ctx.world.cell((3, 2)).infestation > 0.0
    command = InjectInfestation(cell=(3, 2), severity=console.recorded[0].command.severity)
    assert console.command_log() == CommandLog(
        seed=0,
        until=BASE_STEP_MS,
        commands=(TimedCommand(time=BASE_STEP_MS, command=command),),
        scenario="S1",
    )


def test_a_click_on_a_robot_toggles_its_inspection_and_draws_it() -> None:
    console = _console()

    selections: list[int | None] = []
    for _ in range(2):
        console.apply_click(1.5, 1.5)
        console.draw()
        selections.append(console.selected)

    assert selections == [0, None]


def test_a_replay_refuses_new_injections() -> None:
    console = _console(replaying=True)

    console.apply_click(3.5, 2.5)

    assert console.ctx.world.cell((3, 2)).infestation == 0.0
    assert console.recorded == []
