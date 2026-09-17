"""The discrete-event loop.

Pop the earliest event, advance the clock to it, handle it (which may schedule more
events), and repeat until a target time or the queue empties.
"""

from swarm_field.contracts import Time
from swarm_field.simulation.context import SimulationContext


class SimulationLoop:
    def run(self, ctx: SimulationContext, until: Time) -> None:
        """Process events in time order until ``until`` (inclusive) or the queue empties."""
        while True:
            event = ctx.queue.peek()
            if event is None or event.time > until:
                break
            ctx.queue.pop()
            ctx.now = event.time
            event.handle(ctx)
        ctx.now = until
