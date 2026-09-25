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
    PotentialField,
    TreatInfestation,
)
from swarm_field.swarm.brain import ComposedAgent

__all__ = [
    "ActionCandidate",
    "Behavior",
    "ComposedAgent",
    "DecisionContext",
    "Explore",
    "PotentialField",
    "TreatInfestation",
]
