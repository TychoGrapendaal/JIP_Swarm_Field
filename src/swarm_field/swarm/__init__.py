"""Component A — the swarm algorithm (the robot brain).

A brain is a ``ComposedAgent`` assembled from a list of behaviors, each a small piece of
decision logic proposing scored candidate actions. Kept minimal for now: two example
behaviors, combined programmatically.
"""

from swarm_field.swarm.behavior import (
    ActionCandidate,
    Behavior,
    DecisionContext,
    Explore,
    TreatInfestation,
)
from swarm_field.swarm.brain import ComposedAgent
from swarm_field.swarm.memory import AgentMemory, DecayConfig, MemoryView

__all__ = [
    "ActionCandidate",
    "AgentMemory",
    "Behavior",
    "ComposedAgent",
    "DecayConfig",
    "DecisionContext",
    "Explore",
    "MemoryView",
    "TreatInfestation",
]
