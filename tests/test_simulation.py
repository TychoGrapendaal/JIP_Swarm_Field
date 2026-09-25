"""Tests for the discrete-event simulation core and pest dynamics."""

from dataclasses import dataclass
from random import Random

import pytest

from swarm_field.contracts import (
    BodyType,
    FieldGeometry,
    InjectInfestation,
    Move,
    Pause,
    SetTimeScale,
    StepOnce,
    Time,
)
from swarm_field.simulation import (
    PestDynamicsConfig,
    RobotDecision,
    RobotState,
    SimulationContext,
    SimulationLoop,
    WorldState,
    WorldTick,
)
from swarm_field.swarm import ActionCandidate, ComposedAgent, DecisionContext


def _context() -> SimulationContext:
    field = FieldGeometry(
        width=5,
        height=5,
        cell_size=1.0,
        row_sp=1.0,
        in_row_sp=1.0,
        headland=0.0,
        connectivity=4,
    )
    return SimulationContext(
        now=0,
        rng=Random(0),
        agents={},
        world=WorldState(field),
    )


class _FixedMoveBehavior:
    """A minimal fake behavior that always proposes one direct move."""

    name: str = "fixed-move"

    def __init__(self, target: tuple[int, int]) -> None:
        self._target: tuple[int, int] = target

    def propose(self, ctx: DecisionContext) -> ActionCandidate:
        return ActionCandidate(action=Move(to=self._target), score=1.0, source=self.name)


@dataclass
class _RecordingEvent:
    time: Time
    log: list[Time]

    def handle(self, ctx: SimulationContext) -> None:
        self.log.append(self.time)


@dataclass
class _Recurring:
    time: Time
    period: Time
    fires: list[Time]

    def handle(self, ctx: SimulationContext) -> None:
        self.fires.append(self.time)
        nxt = _Recurring(time=self.time + self.period, period=self.period, fires=self.fires)
        ctx.schedule(nxt)


def test_events_processed_in_time_order() -> None:
    log: list[Time] = []
    ctx = _context()
    ctx.schedule(_RecordingEvent(time=30, log=log))
    ctx.schedule(_RecordingEvent(time=10, log=log))
    ctx.schedule(_RecordingEvent(time=20, log=log))

    SimulationLoop().run(ctx, until=100)

    assert log == [10, 20, 30]
    assert ctx.now == 100


def test_run_stops_at_until_and_keeps_later_events() -> None:
    log: list[Time] = []
    ctx = _context()
    ctx.schedule(_RecordingEvent(time=50, log=log))
    ctx.schedule(_RecordingEvent(time=150, log=log))

    SimulationLoop().run(ctx, until=100)

    assert log == [50]
    assert len(ctx.queue) == 1


def test_recurring_event_reschedules_itself() -> None:
    fires: list[Time] = []
    ctx = _context()
    ctx.schedule(_Recurring(time=0, period=25, fires=fires))

    SimulationLoop().run(ctx, until=100)

    assert fires == [0, 25, 50, 75, 100]


def test_world_tick_grows_infestation() -> None:
    ctx = _context()
    ctx.world.cell((2, 2)).infestation = 0.2

    event = WorldTick(
        time=0,
        config=PestDynamicsConfig(
            growth_rate=0.1,
            spread_probability=0.0,
            spawn_probability=0.0,
        ),
    )

    event.handle(ctx)

    assert ctx.world.cell((2, 2)).infestation == pytest.approx(0.3)
    next_event = ctx.queue.peek()
    assert isinstance(next_event, WorldTick)
    assert next_event.time == 60_000


def test_world_tick_spreads_without_cascading() -> None:
    ctx = _context()
    ctx.world.cell((2, 2)).infestation = 1.0

    event = WorldTick(
        time=0,
        config=PestDynamicsConfig(
            growth_rate=0.0,
            spread_probability=1.0,
            spawn_probability=0.0,
            spawn_severity=0.2,
        ),
    )

    event.handle(ctx)

    neighbours = set(ctx.world.neighbours((2, 2)))
    assert all(ctx.world.cell(cell).infestation == 0.2 for cell in neighbours)
    assert ctx.world.cell((0, 0)).infestation == 0.0


def test_world_tick_does_not_infest_obstacles() -> None:
    ctx = _context()
    ctx.world.cell((2, 2)).infestation = 1.0
    ctx.world.cell((2, 3)).obstacle = True

    event = WorldTick(
        time=0,
        config=PestDynamicsConfig(
            growth_rate=0.0,
            spread_probability=1.0,
            spawn_probability=1.0,
        ),
    )

    event.handle(ctx)

    assert ctx.world.cell((2, 3)).infestation == 0.0


def test_world_tick_is_deterministic_for_same_seed() -> None:
    def run(seed: int) -> dict[tuple[int, int], float]:
        ctx = _context()
        ctx.rng = Random(seed)
        ctx.world.cell((2, 2)).infestation = 1.0
        config = PestDynamicsConfig(
            growth_rate=0.05,
            spread_probability=0.25,
            spawn_probability=0.02,
            spawn_severity=0.1,
        )
        ctx.schedule(WorldTick(time=0, config=config))

        SimulationLoop().run(ctx, until=5 * config.period)

        return {cell: state.infestation for cell, state in sorted(ctx.world.cells.items())}

    assert run(seed=42) == run(seed=42)


def test_loop_step_processes_next_event_timestamp() -> None:
    log: list[Time] = []
    ctx = _context()
    ctx.schedule(_RecordingEvent(time=30, log=log))
    ctx.schedule(_RecordingEvent(time=10, log=log))
    ctx.schedule(_RecordingEvent(time=20, log=log))

    SimulationLoop().step(ctx)

    assert log == [10]
    assert ctx.now == 10
    assert len(ctx.queue) == 2


def test_apply_command_injects_infestation() -> None:
    ctx = _context()

    ctx.apply_command(InjectInfestation(cell=(2, 2), severity=0.7))

    assert ctx.world.cell((2, 2)).infestation == pytest.approx(0.7)
    assert ctx.world.cell((1, 1)).infestation == 0.0


def test_apply_command_ignores_time_pacing_commands() -> None:
    ctx = _context()
    for command in (Pause(), StepOnce(), SetTimeScale(2.0)):
        ctx.apply_command(command)


def test_robot_decision_drives_an_agent() -> None:
    robot = RobotState(robot_id=0, position=(2, 2), body=BodyType.SCOUT, speed_mps=2.0)
    ctx = SimulationContext(
        now=0,
        rng=Random(0),
        agents={0: ComposedAgent(robot_id=0, behaviors=[_FixedMoveBehavior((2, 3))])},
        world=WorldState(
            FieldGeometry(
                width=5,
                height=5,
                cell_size=1.0,
                row_sp=1.0,
                in_row_sp=1.0,
                headland=0.0,
            ),
            robots=[robot],
        ),
    )
    ctx.schedule(RobotDecision(time=0, robot=0))

    SimulationLoop().run(ctx, until=5_000)

    assert ctx.world.positions()[0] != (2, 2)
    next_event = ctx.queue.peek()
    assert isinstance(next_event, RobotDecision)
    assert next_event.robot == 0
