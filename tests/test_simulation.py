"""Tests for the discrete-event simulation core: the queue and the loop."""

from dataclasses import dataclass
from random import Random

from swarm_field.contracts import Time
from swarm_field.simulation import SimulationContext, SimulationLoop


def _context() -> SimulationContext:
    return SimulationContext(now=0, rng=Random(0), agents={})


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
