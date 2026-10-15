"""Ground-truth pest response: when each pest appeared, when it was cleared, and how many are left.

It measures how well a swarm keeps a field clean, independently of what the robots believe.
``InfestationSample`` records the infested cells at a fixed period. A cell that turns infested
starts a pest; a cell that turns clean again ends it. Only a treatment clears a cell, so the
time between the two is the swarm's response time to that pest. A pest that spreads to a
neighbour starts a new pest there.

The measures:

- spawned, cleared and open pests;
- response time: from a pest appearing to its cell being clean, its mean and maximum;
- infested cells: how many cells were infested, averaged over the samples.
"""

from __future__ import annotations

from dataclasses import dataclass

from swarm_field.contracts import CellId, Time
from swarm_field.simulation.context import SimulationContext
from swarm_field.simulation.world import WorldState


@dataclass(frozen=True)
class ClearedPest:
    """A pest that appeared and was cleared.

    Attributes:
        cell: Where it was.
        appeared_at: The first sample in which its cell was infested.
        cleared_at: The first sample in which its cell was clean again.
    """

    cell: CellId
    appeared_at: Time
    cleared_at: Time

    @property
    def response_time(self) -> Time:
        return self.cleared_at - self.appeared_at


@dataclass(frozen=True)
class InfestationSummary:
    """The pest measures of a run so far.

    Attributes:
        spawned: How many pests appeared.
        cleared: How many of them were cleared.
        open: How many are still there.
        mean_response: The mean response time of the cleared pests, or ``None`` if none was.
        worst_response: The longest response time of a cleared pest, or of an open pest so
            far, if longer; ``None`` before the first pest.
        mean_infested: How many cells were infested, averaged over the samples.
    """

    spawned: int
    cleared: int
    open: int
    mean_response: float | None
    worst_response: Time | None
    mean_infested: float


class InfestationTracker:
    """Records when pests appear and are cleared, and the measures derived from them.

    Attributes:
        cleared: Every cleared pest, in the order cleared.
    """

    def __init__(self) -> None:
        self.cleared: list[ClearedPest] = []
        self._open: dict[CellId, Time] = {}
        self._spawned: int = 0
        self._now: Time = 0
        self._infested_sum: int = 0
        self._samples: int = 0

    def record(self, world: WorldState, now: Time) -> None:
        """Start a pest on every newly infested cell and end one on every newly clean cell."""
        infested: set[CellId] = set()
        for cell, state in world.cells.items():
            if state.infestation > 0:
                infested.add(cell)
        for cell in infested:
            if cell not in self._open:
                self._open[cell] = now
                self._spawned += 1
        for cell in list(self._open):
            if cell not in infested:
                self.cleared.append(ClearedPest(cell, self._open.pop(cell), now))
        self._now = now
        self._infested_sum += len(infested)
        self._samples += 1

    def summary(self) -> InfestationSummary:
        """The measures over everything recorded so far."""
        mean_response: float | None = None
        worst_response: Time | None = None
        if self.cleared:
            responses = [pest.response_time for pest in self.cleared]
            mean_response = sum(responses) / len(responses)
            worst_response = max(responses)
        for appeared_at in self._open.values():
            age = self._now - appeared_at
            if worst_response is None or age > worst_response:
                worst_response = age
        mean_infested = 0.0
        if self._samples > 0:
            mean_infested = self._infested_sum / self._samples
        return InfestationSummary(
            spawned=self._spawned,
            cleared=len(self.cleared),
            open=len(self._open),
            mean_response=mean_response,
            worst_response=worst_response,
            mean_infested=mean_infested,
        )


@dataclass
class InfestationSample:
    """Record the infested cells in an ``InfestationTracker``, then reschedule at the next period.

    Attributes:
        time: When this sample is taken.
        period: The time between samples.
        tracker: Where the samples go.
    """

    time: Time
    period: Time
    tracker: InfestationTracker

    def handle(self, ctx: SimulationContext) -> None:
        ctx.world.advance_to(self.time)
        self.tracker.record(ctx.world, self.time)
        ctx.schedule(InfestationSample(self.time + self.period, self.period, self.tracker))
