"""Component A — the swarm algorithm (the robot brain).

A brain is a ``ComposedAgent`` assembled from a memory, a navigator and a list of behaviors,
each a small piece of decision logic proposing scored candidates. Behaviors that move towards
targets propose ``FollowField`` with the name of a ``FieldSeeder`` registered in the navigator;
if that candidate wins, the navigator follows the seeder's grid potential field over the terrain
the robot believes in, with a reactive ``Steerer`` to fall back on.
"""

from swarm_field.contracts import Vector2
from swarm_field.swarm.behavior import (
    ActionCandidate,
    Behavior,
    DecisionContext,
    FollowField,
    Intent,
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
    "FollowField",
    "Intent",
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
