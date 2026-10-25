"""The event model: the ``Event`` port, the event queue, and simulation events.

The simulation is discrete-event: an ``EventQueue`` holds timestamped events and the loop
processes them in time order. An event's ``handle`` may schedule follow-up events through the
context. ``WorldTick`` models the recurring coarse-grained pest dynamics; other events (cell
arrival, message delivery, metric sampling, ...) are added as they are built.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from swarm_field.contracts import (
    Beacon,
    BeaconKind,
    CellId,
    FailureMode,
    FailureSpec,
    InfestationObservation,
    Message,
    MessageKind,
    Perception,
    PestDynamicsSpec,
    PestSourceSpec,
    RobotId,
    Time,
    TimeUnit,
)
from swarm_field.simulation.metrics import MetricsConfig

if TYPE_CHECKING:
    from swarm_field.simulation.context import SimulationContext


class Event(Protocol):
    """One thing that happens at a point in simulation time.

    Attributes:
        time: When the event occurs.
    """

    time: Time

    def handle(self, ctx: SimulationContext) -> None: ...


class EventQueue:
    """Min-priority queue of events, ordered by time then insertion order.

    Equal timestamps currently break by insertion order (FIFO). The full deterministic
    tie-break agreed for the simulation (world events before robot events, then robot id)
    is a refinement to layer onto the sort key when the loop is implemented.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[Time, int, Event]] = []
        self._counter: int = 0

    def schedule(self, event: Event) -> None:
        """Add an event to the queue."""
        heapq.heappush(self._heap, (event.time, self._counter, event))
        self._counter += 1

    def peek(self) -> Event | None:
        """The earliest event without removing it, or ``None`` if empty."""
        if not self._heap:
            return None
        return self._heap[0][2]

    def pop(self) -> Event | None:
        """Remove and return the earliest event, or ``None`` if empty."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[2]

    def __len__(self) -> int:
        return len(self._heap)


@dataclass
class RobotDecision:
    """Run one robot decision point and schedule the next one.

    A robot decides again as soon as the move it is making arrives, so it moves at its full
    speed, and every ``decision_interval`` while it stands still. A move cannot be changed
    once started, so a robot that is moving has nothing to decide before it arrives.

    Attributes:
        time: When the robot receives its next perception and decides.
        robot: Which robot acts.
        decision_interval: Time between this decision and the robot's next one while the robot
            stands still.
    """

    time: Time
    robot: RobotId
    decision_interval: Time = 5_000

    def __post_init__(self) -> None:
        """Require a positive decision interval so events always move forward in time."""
        if self.decision_interval <= 0:
            raise ValueError("decision_interval must be positive")

    def handle(self, ctx: SimulationContext) -> None:
        """Perceive, decide, act, deliver messages, and schedule the next decision."""
        ctx.world.advance_to(self.time)
        if self._has_failed(ctx):
            # A failed robot does not come back, so it stops scheduling decisions entirely and
            # the swarm carries on without it. Advancing the world before this check lets a move
            # that exhausts its battery be settled before the robot can make another decision.
            return
        perception = Perception(
            time=self.time,
            self_state=ctx.world.perceived_self(self.robot, self._believed_position(ctx)),
            sensed=ctx.world.observe_for(self.robot, self.time),
            inbox=ctx.channel.take_inbox(self.robot),
        )
        if ctx.metrics is not None:
            # Coverage and detection latency are per-sensing events, so they are recorded
            # here rather than at the next sample, which would only see the latest sensing.
            ctx.metrics.observe(perception.sensed, self.time)
        decision = ctx.agents[self.robot].decide(perception)
        arrival = ctx.world.apply_action(self.robot, decision.action, self.time)
        ctx.channel.broadcast(ctx.world, self.time, decision.outbox)
        next_time = self.time + self.decision_interval
        if arrival is not None and arrival > self.time:
            next_time = arrival
        ctx.schedule(
            RobotDecision(
                time=next_time,
                robot=self.robot,
                decision_interval=self.decision_interval,
            )
        )

    def _has_failed(self, ctx: SimulationContext) -> bool:
        return ctx.world.robots[self.robot].failed

    def _believed_position(self, ctx: SimulationContext) -> CellId | None:
        """Where the robot thinks it is, which is its true cell unless it draws sensor error.

        The error is redrawn at every decision and drawn from ``ctx.noise``, so a run with
        position error is reproducible and does not disturb the world's random stream.
        """
        spec = ctx.position_error
        if spec.probability <= 0.0 or ctx.noise.random() >= spec.probability:
            return None
        cell = ctx.world.robots[self.robot].position
        span = 2 * spec.max_offset + 1
        offset_x = ctx.noise.randrange(span) - spec.max_offset
        offset_y = ctx.noise.randrange(span) - spec.max_offset
        return (cell[0] + offset_x, cell[1] + offset_y)


@dataclass(frozen=True)
class FailureConfig:
    """How often the simulation checks whether a robot has failed, and what failing means.

    Attributes:
        period: Time between checks in simulation time units.
        probability: Chance that a given live robot fails on one check.
        mode: What the failure is when it happens.
    """

    period: Time = TimeUnit.MINUTE
    probability: float = 0.0
    mode: FailureMode = FailureMode.DEAD_BATTERY

    @classmethod
    def from_spec(cls, spec: FailureSpec) -> FailureConfig:
        """Build the event's configuration from the shared scenario contract."""
        return cls(period=spec.period, probability=spec.probability, mode=spec.mode)

    def __post_init__(self) -> None:
        """Validate the supported parameter ranges."""
        if self.period <= 0:
            raise ValueError("failure period must be positive")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("failure probability must be between 0 and 1")


@dataclass
class FailureCheck:
    """Fail some live robots, then reschedule the next check.

    Draws from ``ctx.noise`` rather than ``ctx.rng``, so a run with failures on is comparable
    with its baseline: the pest dynamics see exactly the same random stream either way.

    Attributes:
        time: Simulation time of this check.
        config: How often to check and what failing means.
    """

    time: Time
    config: FailureConfig = field(default_factory=FailureConfig)

    def handle(self, ctx: SimulationContext) -> None:
        """Fail each live robot that draws a fault, then schedule the next check."""
        for robot_id in ctx.world.live_robots():
            if ctx.noise.random() < self.config.probability:
                ctx.world.fail_robot(robot_id, self.config.mode, time=self.time)
        ctx.schedule(FailureCheck(time=self.time + self.config.period, config=self.config))


@dataclass
class PestSourceBroadcast:
    """Broadcast every pest from its own cell, as an explorer hovering above it would.

    Each infested cell sends, of the spec's ``kinds``, a ``KNOWLEDGE`` message with the
    infestation fact that own-cell sensing gives, and a ``BEACON`` message with a TREAT beacon of
    0 on the cell, from a synthetic sender of its own. A pest cleared since the last broadcast is
    reported once more, as clean and without a beacon, and then falls silent. Draws nothing from
    ``ctx.rng``.

    Attributes:
        time: Simulation time of this broadcast.
        spec: How often to broadcast and how far.
        infested: The cells that were infested at the previous broadcast.
    """

    time: Time
    spec: PestSourceSpec
    infested: frozenset[CellId] = frozenset()

    def handle(self, ctx: SimulationContext) -> None:
        """Broadcast the infested and the just-cleared cells, then schedule the next round."""
        infested_now: set[CellId] = set()
        for cell, state in ctx.world.cells.items():
            if state.infestation > 0.0:
                infested_now.add(cell)
        for cell in sorted(infested_now | self.infested):
            source = _pest_source_id(cell, ctx.world.field.width)
            severity = ctx.world.cells[cell].infestation
            for message in self._source_messages(cell, source, severity):
                ctx.channel.broadcast_from_cell(
                    ctx.world, self.time, cell, self.spec.radio_range_m, message
                )
        ctx.schedule(
            PestSourceBroadcast(
                time=self.time + self.spec.period, spec=self.spec, infested=frozenset(infested_now)
            )
        )

    def _source_messages(self, cell: CellId, source: RobotId, severity: float) -> list[Message]:
        """What the source on ``cell`` sends, of the spec's ``kinds``: the infestation fact, and
        a beacon while infested."""
        messages: list[Message] = []
        if MessageKind.KNOWLEDGE in self.spec.kinds:
            fact = InfestationObservation(
                cell=cell,
                confidence=1.0,
                observed_at=self.time,
                source=source,
                severity=severity,
                pest=None,
            )
            messages.append(
                Message(
                    sender=source, sent_at=self.time, kind=MessageKind.KNOWLEDGE, payload=(fact,)
                )
            )
        if MessageKind.BEACON in self.spec.kinds and severity > 0.0:
            beacon = Beacon(BeaconKind.TREAT, 0.0, cell)
            messages.append(
                Message(
                    sender=source, sent_at=self.time, kind=MessageKind.BEACON, payload=(beacon,)
                )
            )
        return messages


def _pest_source_id(cell: CellId, width: int) -> RobotId:
    """A sender id for the source on ``cell``: negative, so it never clashes with a robot's."""
    return -(1 + cell[0] + cell[1] * width)


@dataclass
class MetricSample:
    """Record one row of the run's metrics series, render the world, then reschedule itself.

    Runs on its own coarse cadence, independent of robot activity and of the pest dynamics.
    It draws nothing from ``ctx.rng``, so recording metrics leaves the rest of the run
    bit-for-bit identical.

    This is also the run's render tick, which is why the event is scheduled even for a run
    that records no metrics: a sweep and a live view share one cadence, so a frame always
    corresponds to a row of the series. Rendering reads only ground truth and draws nothing, so
    it cannot change the run either.

    Attributes:
        time: Simulation time this sample is taken at.
        config: The sampling cadence and the target threshold.
    """

    time: Time
    config: MetricsConfig = field(default_factory=MetricsConfig)

    def handle(self, ctx: SimulationContext) -> None:
        """Take one sample if the run records metrics, render if it has a renderer, reschedule."""
        sample = None
        if ctx.metrics is not None:
            sample = ctx.metrics.sample(
                ctx.world, self.time, ctx.channel.sent, ctx.channel.received
            )
        if ctx.renderer is not None:
            ctx.renderer.render(ctx.world.snapshot(self.time, metrics=sample))
        ctx.schedule(MetricSample(time=self.time + self.config.period, config=self.config))


@dataclass(frozen=True)
class PestDynamicsConfig:
    """Parameters controlling one coarse-grained pest population model.

    Attributes:
        period: Time between world ticks in simulation time units.
        spawn_probability: Probability that each healthy, non-obstacle cell spawns a new
            infestation during one tick.
        growth_rate: Additive severity growth applied to existing infestations per tick.
        spread_probability: Maximum probability that one infested cell infects one neighbour;
            the actual probability is scaled by the source severity.
        spawn_severity: Initial severity assigned to newly spawned or spread infestations.
    """

    period: Time = TimeUnit.MINUTE
    spawn_probability: float = 0.01
    growth_rate: float = 0.05
    spread_probability: float = 0.20
    spawn_severity: float = 0.10

    @classmethod
    def from_spec(cls, spec: PestDynamicsSpec) -> PestDynamicsConfig:
        """Build simulation dynamics from the shared scenario contract."""
        return cls(
            period=spec.period,
            spawn_probability=spec.spawn_probability,
            growth_rate=spec.growth_rate,
            spread_probability=spec.spread_probability,
            spawn_severity=spec.spawn_severity,
        )

    def __post_init__(self) -> None:
        """Validate supported pest parameter ranges."""
        if self.period <= 0:
            raise ValueError("pest dynamics period must be positive")
        for name, value in (
            ("spawn_probability", self.spawn_probability),
            ("growth_rate", self.growth_rate),
            ("spread_probability", self.spread_probability),
            ("spawn_severity", self.spawn_severity),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass
class WorldTick:
    """Advance pest dynamics one step, then reschedule itself at the next tick.

    Runs on its own coarse cadence (simulated minutes/hours), independent of robot activity.

    Attributes:
        time: Simulation time when this world update happens.
        config: Rates and timing used by the pest dynamics.
    """

    time: Time
    config: PestDynamicsConfig = field(default_factory=PestDynamicsConfig)

    def handle(self, ctx: SimulationContext) -> None:
        self._grow(ctx)
        new_infestations = self._spread(ctx)
        self._apply_new_infestations(ctx, new_infestations)
        self._spawn(ctx)
        if ctx.metrics is not None:
            ctx.metrics.track_infestation(ctx.world, self.time)

        ctx.schedule(
            WorldTick(
                time=self.time + self.config.period,
                config=self.config,
            )
        )

    def _grow(self, ctx: SimulationContext) -> None:
        """Increase every existing infestation without exceeding full severity.

        A cell a killer is treating does not grow: treatment removes a unit of severity per
        hour, which would otherwise lose to any growth faster than that, so no pest could be
        cleared.
        """
        if self.config.growth_rate == 0.0:
            return

        treated: set[CellId] = set()
        for robot in ctx.world.robots.values():
            if robot.treating_cell is not None:
                treated.add(robot.treating_cell)
        for cell, state in ctx.world.cells.items():
            if state.infestation <= 0.0 or cell in treated:
                continue
            state.infestation = min(1.0, state.infestation + self.config.growth_rate)

    def _spread(self, ctx: SimulationContext) -> set[CellId]:
        """Return newly infected cells selected from the current world state.

        The returned set is applied only after every source cell has been evaluated. This
        keeps one tick from cascading through multiple generations of neighbours.
        """
        if self.config.spread_probability == 0.0:
            return set()

        new_infestations: set[CellId] = set()
        for cell, state in sorted(ctx.world.cells.items()):
            if state.infestation <= 0.0:
                continue

            chance = state.infestation * self.config.spread_probability
            for neighbour in ctx.world.neighbours(cell):
                target = ctx.world.cell(neighbour)
                if target.obstacle or target.infestation > 0.0:
                    continue
                if ctx.rng.random() < chance:
                    new_infestations.add(neighbour)

        return new_infestations

    def _apply_new_infestations(
        self,
        ctx: SimulationContext,
        cells: set[CellId],
    ) -> None:
        """Apply one-tick spread results to the ground-truth world."""
        for cell in sorted(cells):
            ctx.world.cell(cell).infestation = self.config.spawn_severity

    def _spawn(self, ctx: SimulationContext) -> None:
        """Independently spawn new infestations on healthy, non-obstacle cells."""
        if self.config.spawn_probability == 0.0:
            return

        for _cell, state in sorted(ctx.world.cells.items()):
            if state.obstacle or state.infestation > 0.0:
                continue
            if ctx.rng.random() < self.config.spawn_probability:
                state.infestation = self.config.spawn_severity
