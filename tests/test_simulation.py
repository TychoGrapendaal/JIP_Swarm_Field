"""Tests for the discrete-event simulation core and pest dynamics."""

from collections.abc import Hashable
from dataclasses import dataclass
from random import Random
from typing import override

import pytest

from swarm_field.contracts import (
    Action,
    AgentInspection,
    Beacon,
    BeaconKind,
    BodyType,
    CellId,
    Decision,
    FieldGeometry,
    Idle,
    InfestationObservation,
    InjectInfestation,
    Message,
    MessageKind,
    Move,
    Pause,
    Perception,
    PestSourceSpec,
    SetTimeScale,
    StepOnce,
    Time,
)
from swarm_field.simulation import (
    PestDynamicsConfig,
    PestSourceBroadcast,
    RobotDecision,
    RobotState,
    SimulationContext,
    SimulationLoop,
    WorldState,
    WorldTick,
)
from swarm_field.swarm import (
    AgentMemory,
    ComposedAgent,
    FieldProvider,
    MemoryView,
    Navigator,
    TreatInfestation,
    UtilityArbiter,
)
from swarm_field.swarm.field import Seed


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


class _FixedTargetSeeder:
    """One fixed target, so a behaviour has a field to follow without any pest logic."""

    name: str = "fixed-target"
    window_blocks: int = 2

    def __init__(self, target: CellId) -> None:
        self._target: CellId = target

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        return None

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        return (Seed(self._target, 0.0),)


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


def test_robot_decision_perceives_acts_and_requeues_at_fixed_interval() -> None:
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
    ctx.schedule(RobotDecision(time=0, robot=1, decision_interval=5_000))

    SimulationLoop().run(ctx, until=5_000)

    assert len(agent.perceptions) == 2
    assert agent.perceptions[0].time == 0
    assert agent.perceptions[0].self_state.position == (1, 1)
    assert agent.perceptions[1].time == 5_000
    assert agent.perceptions[1].self_state.position == (2, 1)
    assert world.positions()[1] == (2, 1)

    next_event = ctx.queue.peek()
    assert isinstance(next_event, RobotDecision)
    assert next_event.time == 10_000
    assert next_event.robot == 1


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


def test_robot_decision_drives_a_real_brain() -> None:
    field = FieldGeometry(
        width=5,
        height=5,
        cell_size=1.0,
        row_sp=1.0,
        in_row_sp=1.0,
        headland=0.0,
    )
    robot = RobotState(robot_id=0, position=(2, 2), body=BodyType.SCOUT, speed_mps=2.0)
    memory = AgentMemory(0, field)
    # TreatInfestation bids only while memory knows a pest.
    memory.merge([InfestationObservation((2, 3), 1.0, 0, 1, severity=0.5, pest=None)])
    seeder = _FixedTargetSeeder((2, 3))
    fields = FieldProvider(memory, flies=False, seeders=[seeder])
    ctx = SimulationContext(
        now=0,
        rng=Random(0),
        agents={
            0: ComposedAgent(
                robot_id=0,
                behaviors=[TreatInfestation(seeder.name)],
                memory=memory,
                fields=fields,
                navigator=Navigator(fields.quadtree),
                arbiter=UtilityArbiter(),
            )
        },
        world=WorldState(field, robots=[robot]),
    )
    ctx.schedule(RobotDecision(time=0, robot=0))

    SimulationLoop().run(ctx, until=5_000)

    assert ctx.world.positions()[0] != (2, 2)
    next_event = ctx.queue.peek()
    assert isinstance(next_event, RobotDecision)
    assert next_event.robot == 0


def test_robot_decision_delivers_outbox_to_next_perception() -> None:
    from swarm_field.contracts import Message, MessageKind

    sender = RobotState(
        robot_id=1,
        position=(1, 1),
        body=BodyType.SCOUT,
        speed_mps=1.0,
        radio_range_m=5.0,
    )
    receiver = RobotState(
        robot_id=2,
        position=(2, 1),
        body=BodyType.SCOUT,
        speed_mps=1.0,
        radio_range_m=5.0,
    )

    message = Message(
        sender=1,
        sent_at=0,
        kind=MessageKind.HELLO,
        payload="hello",
    )

    class _SenderAgent(_ScriptedAgent):
        @override
        def decide(self, perception: Perception) -> Decision:
            self.perceptions.append(perception)
            return Decision(action=Idle(), outbox=[message])

    sender_agent = _SenderAgent(1, (Idle(),))
    receiver_agent = _ScriptedAgent(2, (Idle(),))
    world = WorldState(
        FieldGeometry(
            width=4,
            height=4,
            cell_size=1.0,
            row_sp=1.0,
            in_row_sp=1.0,
            headland=0.0,
            connectivity=4,
        ),
        robots=[sender, receiver],
    )
    ctx = SimulationContext(
        now=0,
        rng=Random(0),
        agents={1: sender_agent, 2: receiver_agent},
        world=world,
    )
    ctx.schedule(RobotDecision(time=0, robot=1, decision_interval=5_000))
    ctx.schedule(RobotDecision(time=5_000, robot=2, decision_interval=5_000))

    SimulationLoop().run(ctx, until=5_000)

    assert len(receiver_agent.perceptions) == 1
    inbox = receiver_agent.perceptions[0].inbox
    assert inbox == [message]


def _pest_source_world() -> WorldState:
    near = RobotState(robot_id=1, position=(1, 1), body=BodyType.KILLER, speed_mps=0.1)
    far = RobotState(robot_id=2, position=(8, 8), body=BodyType.KILLER, speed_mps=0.1)
    field = FieldGeometry(
        width=10, height=10, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0
    )
    return WorldState(field, robots=[near, far])


def _broadcast_severities(message_list: list[Message]) -> list[float]:
    severities: list[float] = []
    for message in message_list:
        if message.kind is not MessageKind.KNOWLEDGE:
            continue
        assert isinstance(message.payload, tuple)
        for fact in message.payload:
            assert isinstance(fact, InfestationObservation)
            assert fact.cell == (2, 1)
            severities.append(fact.severity)
    return severities


def test_a_pest_source_reaches_only_the_robots_in_range() -> None:
    ctx = SimulationContext(now=0, rng=Random(0), agents={}, world=_pest_source_world())
    ctx.world.cells[(2, 1)].infestation = 0.5

    PestSourceBroadcast(time=0, spec=PestSourceSpec(radio_range_m=15.0)).handle(ctx)

    inbox = ctx.channel.take_inbox(1)
    assert _broadcast_severities(inbox) == [0.5]
    beacons = [message.payload for message in inbox if message.kind is MessageKind.BEACON]
    assert beacons == [(Beacon(BeaconKind.TREAT, 0.0, (2, 1)),)]
    assert ctx.channel.take_inbox(2) == []
    assert ctx.channel.sent == 0


def test_a_cleared_pest_is_reported_clean_once_then_falls_silent() -> None:
    ctx = SimulationContext(now=0, rng=Random(0), agents={}, world=_pest_source_world())
    ctx.world.cells[(2, 1)].infestation = 0.5
    PestSourceBroadcast(time=0, spec=PestSourceSpec(radio_range_m=15.0)).handle(ctx)
    ctx.channel.take_inbox(1)
    ctx.world.cells[(2, 1)].infestation = 0.0

    severities: list[list[float]] = []
    kinds: list[set[MessageKind]] = []
    for _ in range(2):
        event = ctx.queue.pop()
        assert isinstance(event, PestSourceBroadcast)
        event.handle(ctx)
        inbox = ctx.channel.take_inbox(1)
        severities.append(_broadcast_severities(inbox))
        kinds.append({message.kind for message in inbox})

    assert severities == [[0.0], []]
    assert kinds == [{MessageKind.KNOWLEDGE}, set()]
