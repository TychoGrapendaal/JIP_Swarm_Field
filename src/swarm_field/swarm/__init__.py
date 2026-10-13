"""Component A — the swarm algorithm (the robot brain).

A brain is a ``ComposedAgent`` assembled from a memory, a field provider, a navigator and a
list of behaviors, each a small piece of decision logic proposing scored candidates. Behaviors
that move towards targets propose ``FollowField`` with the name of a ``FieldSeeder`` registered
in the ``FieldProvider``; if that candidate wins, the navigator follows the seeder's grid
potential field over the terrain the robot believes in, with a reactive ``Steerer`` to fall
back on.
"""

from swarm_field.contracts import Vector2
from swarm_field.swarm.assembly import build_agent
from swarm_field.swarm.behavior import (
    ActionCandidate,
    Behavior,
    DecisionContext,
    ExploreField,
    FollowField,
    Intent,
    ReturnToDock,
    SupplyBattery,
    TreatInfestation,
)
from swarm_field.swarm.brain import ComposedAgent
from swarm_field.swarm.capabilities import BODY_CAPABILITIES, CapabilitySet, capabilities_for
from swarm_field.swarm.memory import AgentMemory, DecayConfig, MemoryView
from swarm_field.swarm.navigator import Navigator
from swarm_field.swarm.provider import FieldProvider
from swarm_field.swarm.seeder import (
    BatteryRequestSeeder,
    DockSeeder,
    ExploreSeeder,
    FieldSeeder,
    SeededField,
    TreatSeeder,
)
from swarm_field.swarm.steering import (
    QuadtreeSensing,
    Sensing,
    Steerer,
    SteeringConfig,
)

__all__ = [
    "ActionCandidate",
    "AgentMemory",
    "BODY_CAPABILITIES",
    "BatteryRequestSeeder",
    "Behavior",
    "CapabilitySet",
    "ComposedAgent",
    "DecayConfig",
    "DecisionContext",
    "DockSeeder",
    "ExploreField",
    "ExploreSeeder",
    "FieldProvider",
    "FieldSeeder",
    "FollowField",
    "Intent",
    "MemoryView",
    "Navigator",
    "QuadtreeSensing",
    "ReturnToDock",
    "SeededField",
    "Sensing",
    "Steerer",
    "SteeringConfig",
    "SupplyBattery",
    "TreatInfestation",
    "TreatSeeder",
    "Vector2",
    "build_agent",
    "capabilities_for",
]
