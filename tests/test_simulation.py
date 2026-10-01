"""Tests for the discrete-event simulation core and pest dynamics."""

from dataclasses import dataclass
from random import Random

import pytest

from swarm_field.contracts import (
    Action,
    AgentInspection,
    BodyType,
    Decision,
    FieldGeometry,
    Idle,
    Move,
    Perception,
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


class _ScriptedAgent:
    """Minimal agent that records each perception and returns scripted actions."""

    def __init__(self, robot_id: int, actions: tuple[Action, ...]) -> None:
        self.robot_id: int = robot_id
        self._actions: tuple[Action, ...] = actions
        self.perceptions: list[Perception] = []

    def decide(self, perception: Perception) -> Decision:
        self.perceptions.append(perception)
        index = min(len(self.perceptions) - 1, len(self._actions) - 1)
        return Decision(action=self._actions[index])

    def inspect(self) -> AgentInspection:
        return AgentInspection(robot_id=self.robot_id, current_goal=None, belief=None)


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


def test_robot_decision_perceives_acts_and_requeues() -> None:
    robot = RobotState(
        robot_id=1,
        position=(1, 1),
        body=BodyType.SCOUT,
        speed_mps=1.0,
        battery=0.75,
    )
    agent = _ScriptedAgent(1, (Move(to=(2, 1)), Idle()))
    field = FieldGeometry(
        width=4,
        height=4,
        cell_size=1.0,
        row_sp=1.0,
        in_row_sp=1.0,
        headland=0.0,
        connectivity=4,
    )
    world = WorldState(field, robots=[robot])
    ctx = SimulationContext(now=0, rng=Random(0), agents={1: agent}, world=world)
    ctx.schedule(RobotDecision(time=0, robot=1))

    SimulationLoop().run(ctx, until=1_000)

    assert len(agent.perceptions) == 2
    assert agent.perceptions[0].time == 0
    assert agent.perceptions[0].self_state.position == (1, 1)
    assert agent.perceptions[1].time == 1_000
    assert agent.perceptions[1].self_state.position == (2, 1)
    assert world.positions()[1] == (2, 1)

    next_event = ctx.queue.peek()
    assert isinstance(next_event, RobotDecision)
    assert next_event.time == 6_000
    assert next_event.robot == 1
