"""Component A — the swarm algorithm (the robot brain).

A brain is a ``ComposedAgent`` assembled from a memory, a navigator and a list of behaviors,
each a small piece of decision logic proposing scored candidate actions. Behaviors that move
towards targets follow the grid potential field of an objective, built by the navigator.
"""

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
    "TreatInfestation",
    "TreatObjective",
]
