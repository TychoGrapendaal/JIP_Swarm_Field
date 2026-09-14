"""The shared per-tick context threaded through every simulation stage.

Stages communicate only through this context; they never call each other. Kept minimal for
now — fields such as a world and a message channel are added as the pieces that need them
are built.
"""

from dataclasses import dataclass, field
from random import Random

from swarm_field.contracts import Decision, RobotId, SwarmAgent, Tick


@dataclass
class SimContext:
    """Mutable state shared by all stages during one run.

    Attributes:
        tick: The current tick.
        rng: The single seeded random source; stages draw from it in a fixed order.
        agents: The robot brains, keyed by id (the boundary between the simulation and A).
        decisions: Per-tick scratch — each robot's decision, set by the decide stage.
    """

    tick: Tick
    rng: Random
    agents: dict[RobotId, SwarmAgent]
    decisions: dict[RobotId, Decision] = field(default_factory=dict)
