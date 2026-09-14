"""Component B — the environment simulation.

Drives time as a pipeline of stages (``SimLoop`` over ``SimContext``). Kept minimal for
now: two example stages, assembled into a loop programmatically.
"""

from swarm_field.sim.context import SimContext
from swarm_field.sim.loop import SimLoop
from swarm_field.sim.stage import Decide, SimStage, SpreadPests

__all__ = [
    "Decide",
    "SimContext",
    "SimLoop",
    "SimStage",
    "SpreadPests",
]
