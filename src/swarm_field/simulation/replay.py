"""Replaying the recorded commands of an interactive run on a freshly built context."""

from swarm_field.contracts import CommandLog, Time, TimedCommand
from swarm_field.simulation.context import SimulationContext
from swarm_field.simulation.loop import SimulationLoop


class CommandReplay:
    """Advances a context and applies a ``CommandLog``'s commands at their recorded times.

    A live frontend applies a command between two ``SimulationLoop.run`` calls, after every
    event up to the current time was handled. The replay applies it at that same point, so a
    context built from the recorded seed reproduces the recorded run exactly. Scheduling the
    commands as events instead would not: equal timestamps break by insertion order, so a
    command scheduled up front would run before events the live run handled first.
    """

    def __init__(self, ctx: SimulationContext, log: CommandLog) -> None:
        self.ctx: SimulationContext = ctx
        self.log: CommandLog = log
        self._loop: SimulationLoop = SimulationLoop()
        self._pending: list[TimedCommand] = sorted(log.commands, key=lambda timed: timed.time)

    def advance_to(self, until: Time) -> None:
        """Run up to ``until``, applying every recorded command that falls within it."""
        while self._pending and self._pending[0].time <= until:
            timed = self._pending.pop(0)
            self._loop.run(self.ctx, until=timed.time)
            self.ctx.apply_command(timed.command)
        self._loop.run(self.ctx, until=until)

    def step(self) -> None:
        """Process the next pending event timestamp, applying any command recorded at it."""
        event = self.ctx.queue.peek()
        if event is None:
            return
        self.advance_to(event.time)
