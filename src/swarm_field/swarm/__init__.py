"""Component A — the swarm algorithm (the robot brain).

A brain is a ``ComposedAgent`` assembled from a memory, a navigator and a list of behaviors,
each a small piece of decision logic proposing scored candidate actions. Behaviors that move
towards targets name a ``FieldSeeder`` registered in the navigator, which builds that seeder's
grid potential field over the terrain the robot believes in and follows it, with a reactive
``Steerer`` to fall back on.
"""

from swarm_field.contracts import Vector2
from swarm_field.swarm.behavior import (
    ActionCandidate,
    Behavior,
    DecisionContext,
    TreatInfestation,
)
from swarm_field.swarm.brain import ComposedAgent
from swarm_field.swarm.memory import AgentMemory, DecayConfig, MemoryView
from swarm_field.swarm.navigator import Navigator
from swarm_field.swarm.seeder import FieldSeeder, SeededField, TreatSeeder
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
    "FieldSeeder",
    "MemoryView",
    "Navigator",
    "QuadtreeSensing",
    "SeededField",
    "Sensing",
    "Steerer",
    "SteeringConfig",
    "TreatInfestation",
    "TreatSeeder",
    "Vector2",
]
