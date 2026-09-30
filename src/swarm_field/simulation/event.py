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

from swarm_field.contracts import CellId, Perception, PestDynamicsSpec, RobotId, Time

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
    """A robot perceives, calls ``decide``, commits to an action, and schedules its next event.

    While moving the follow-up is the next cell arrival; while stationary it is the next
    decision at a fixed interval. Actions are not interrupted, so no scheduled event is ever
    cancelled.

    Attributes:
        time: When the robot receives its next perception and decides.
        robot: Which robot acts.
        dwell: Stationary decision interval used when the chosen action does not start a move.
    """

    time: Time
    robot: RobotId
    dwell: Time = 5_000

    def handle(self, ctx: SimulationContext) -> None:
        """Build the robot's perception, apply its decision, and schedule the follow-up."""
        ctx.world.advance_to(self.time)
        perception = Perception(
            time=self.time,
            self_state=ctx.world.perceived_self(self.robot),
            sensed=ctx.world.observe_for(self.robot, self.time),
            inbox=ctx.channel.take_inbox(self.robot),
        )
        decision = ctx.agents[self.robot].decide(perception)
        ctx.channel.broadcast(ctx.world, self.time, decision.outbox)
        arrival = ctx.world.apply_action(self.robot, decision.action, self.time)
        follow_up = arrival if arrival is not None else self.time + self.dwell
        ctx.schedule(RobotDecision(time=follow_up, robot=self.robot, dwell=self.dwell))


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

    period: Time = 60_000
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

        ctx.schedule(
            WorldTick(
                time=self.time + self.config.period,
                config=self.config,
            )
        )

    def _grow(self, ctx: SimulationContext) -> None:
        """Increase every existing infestation without exceeding full severity."""
        if self.config.growth_rate == 0.0:
            return

        for state in ctx.world.cells.values():
            if state.infestation <= 0.0:
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
