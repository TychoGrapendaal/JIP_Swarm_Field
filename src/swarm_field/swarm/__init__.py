"""Component A — the swarm algorithm (the robot brain).

A brain is a ``ComposedAgent`` assembled from a memory, a navigator and a list of behaviors,
each a small piece of decision logic proposing scored candidate actions. Behaviors that move
towards targets hand their objective's targets to the navigator, which follows them over a grid
potential field of the terrain the robot believes in, with a reactive ``Steerer`` to fall back
on.
"""

from swarm_field.contracts import Vector2
from swarm_field.swarm.behavior import (
    ActionCandidate,
    Behavior,
    DecisionContext,
    Explore,
    PotentialField,
    TreatInfestation,
)
from swarm_field.swarm.brain import ComposedAgent
from swarm_field.swarm.memory import AgentMemory, DecayConfig, MemoryView
from swarm_field.swarm.navigator import Navigator
from swarm_field.swarm.objective import Objective, TreatObjective
from swarm_field.swarm.steering import (
    QuadtreeSensing,
    Sensing,
    Steerer,
    SteeringConfig,
)

__all__ = [
    "ActionCandidate",
    "AgentMemory",
    "Behavior",
    "ComposedAgent",
    "DecayConfig",
    "DecisionContext",
    "Explore",
    "MemoryView",
    "Navigator",
    "Objective",
    "PotentialField",
    "QuadtreeSensing",
    "Sensing",
    "Steerer",
    "SteeringConfig",
    "TreatInfestation",
    "TreatObjective",
    "Vector2",
]
