"""B.2 — the staged simulation loop.

The loop is the active object: it knows nothing about what a tick contains, only that it
runs each stage in order and advances the clock. Build one by passing the stages directly,
e.g. ``SimLoop([Decide(), SpreadPests()])``.
"""

from swarm_field.contracts import Tick
from swarm_field.sim.context import SimContext
from swarm_field.sim.stage import SimStage


class SimLoop:
    def __init__(self, stages: list[SimStage]) -> None:
        self.stages: list[SimStage] = stages

    def step(self, ctx: SimContext) -> None:
        """Run every stage in order for one tick, then advance the clock."""
        for stage in self.stages:
            stage.run(ctx)
        ctx.tick += 1

    def run(self, ctx: SimContext, until: Tick) -> None:
        """Step until ``ctx.tick`` reaches ``until``."""
        while ctx.tick < until:
            self.step(ctx)
