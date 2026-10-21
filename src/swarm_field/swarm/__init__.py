"""Component A — the swarm algorithm (the robot brain).

A brain is a ``ComposedAgent`` assembled from a memory, a field provider, a navigator and a
list of behaviors, each a small piece of decision logic proposing candidates scored on one shared
``[0, 1]`` scale, and an ``Arbiter`` that picks one. Behaviors that move towards targets propose
``FollowField`` with the name of a ``FieldSeeder`` registered in the ``FieldProvider``; if that
candidate wins, the navigator follows the seeder's grid potential field over the terrain the
robot believes in, with a reactive ``Steerer`` to fall back on.
"""

from swarm_field.contracts import Vector2
from swarm_field.swarm.arbiter import (
    Arbiter,
    ArbiterChoice,
    ArbiterConfig,
    Commitment,
    UtilityArbiter,
)
from swarm_field.swarm.behavior import (
    ActionCandidate,
    Behavior,
    DecisionContext,
    ExploreBehaviour,
    FollowField,
    FollowSeederField,
    Intent,
    TreatInfestation,
)
from swarm_field.swarm.brain import ComposedAgent
from swarm_field.swarm.explore import ExploreConfig, ExploreSeeder
from swarm_field.swarm.memory import AgentMemory, DecayConfig, KnowledgeSnapshot, MemoryView
from swarm_field.swarm.navigator import Navigator
from swarm_field.swarm.provider import FieldProvider
from swarm_field.swarm.seeder import (
    FieldSeeder,
    SeededField,
    TreatSeeder,
)
from swarm_field.swarm.sharing import KnowledgeExchange, SharingConfig, SharingStrategy
from swarm_field.swarm.steering import (
    QuadtreeSensing,
    Sensing,
    Steerer,
    SteeringConfig,
)

__all__ = [
    "ActionCandidate",
    "AgentMemory",
    "Arbiter",
    "ArbiterChoice",
    "ArbiterConfig",
    "Behavior",
    "Commitment",
    "ComposedAgent",
    "DecayConfig",
    "DecisionContext",
    "ExploreBehaviour",
    "ExploreConfig",
    "ExploreSeeder",
    "FieldProvider",
    "FieldSeeder",
    "FollowField",
    "FollowSeederField",
    "Intent",
    "KnowledgeExchange",
    "KnowledgeSnapshot",
    "MemoryView",
    "Navigator",
    "QuadtreeSensing",
    "SeededField",
    "Sensing",
    "SharingConfig",
    "SharingStrategy",
    "Steerer",
    "SteeringConfig",
    "TreatInfestation",
    "TreatSeeder",
    "UtilityArbiter",
    "Vector2",
]
