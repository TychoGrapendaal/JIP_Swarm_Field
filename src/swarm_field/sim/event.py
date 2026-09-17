"""The event model: the ``Event`` port, the event queue, and a couple of example events.

The simulation is discrete-event: an ``EventQueue`` holds timestamped events and the loop
processes them in time order. An event's ``handle`` may schedule follow-up events through
the context. The two examples are still unimplemented: ``RobotDecision`` (a robot perceives,
decides and commits to an action) and ``WorldTick`` (recurring pest dynamics). Real events
(cell arrival, message delivery, metric sampling, ...) are added as they are built.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from swarm_field.contracts import RobotId, Time

if TYPE_CHECKING:
    from swarm_field.sim.context import SimContext


class Event(Protocol):
    """One thing that happens at a point in simulation time.

    Attributes:
        time: When the event occurs.
    """

    time: Time

    def handle(self, ctx: SimContext) -> None: ...


class EventQueue:
    """Min-priority queue of events, ordered by time then insertion order.

    Equal timestamps currently break by insertion order (FIFO). The full deterministic
    tie-break agreed for the sim (world events before robot events, then robot id) is a
    refinement to layer onto the sort key when the loop is implemented.
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
    """

    time: Time
    robot: RobotId

    def handle(self, ctx: SimContext) -> None:
        raise NotImplementedError


@dataclass
class WorldTick:
    """Advance pest dynamics one step, then reschedule itself at the next tick.

    Runs on its own coarse cadence (sim-minutes/hours), independent of robot activity.
    """

    time: Time

    def handle(self, ctx: SimContext) -> None:
        raise NotImplementedError
