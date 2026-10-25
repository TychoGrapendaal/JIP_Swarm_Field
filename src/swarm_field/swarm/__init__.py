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
    Intent,
    Regroup,
    ReturnToDock,
    SupplyBattery,
    TreatInfestation,
)
from swarm_field.swarm.brain import ComposedAgent
from swarm_field.swarm.capabilities import BODY_CAPABILITIES, CapabilitySet, capabilities_for
from swarm_field.swarm.explore import ExploreConfig, ExploreSeeder
from swarm_field.swarm.guidance import Guidance, Heading, ObjectiveField
from swarm_field.swarm.hop import (
    BeaconConfig,
    BeaconLink,
    BeaconMemory,
    BeaconView,
    EscapeConfig,
    HeardBeacon,
    HopField,
    PlacedBeacon,
    TreatByBeacon,
    treat_beacon,
)
from swarm_field.swarm.memory import AgentMemory, DecayConfig, KnowledgeSnapshot, MemoryView
from swarm_field.swarm.navigator import Navigator
from swarm_field.swarm.provider import FieldProvider
from swarm_field.swarm.recharge import RechargeConfig, ResupplyConfig
from swarm_field.swarm.resupply import Resupply
from swarm_field.swarm.seeder import (
    BatteryRequestSeeder,
    DockSeeder,
    FieldSeeder,
    RegroupSeeder,
    SeededField,
    TreatSeeder,
)
from swarm_field.swarm.sharing import (
    KnowledgeExchange,
    KnowledgeSharing,
    SharingConfig,
    SharingStrategy,
)
from swarm_field.swarm.steering import (
    QuadtreeSensing,
    SensedSurroundings,
    Sensing,
    Steerer,
    SteeringConfig,
)

__all__ = [
    "BODY_CAPABILITIES",
    "ActionCandidate",
    "AgentMemory",
    "Arbiter",
    "ArbiterChoice",
    "ArbiterConfig",
    "BatteryRequestSeeder",
    "BeaconConfig",
    "BeaconLink",
    "BeaconMemory",
    "BeaconView",
    "Behavior",
    "CapabilitySet",
    "Commitment",
    "ComposedAgent",
    "DecayConfig",
    "DecisionContext",
    "DockSeeder",
    "EscapeConfig",
    "ExploreBehaviour",
    "ExploreConfig",
    "ExploreSeeder",
    "FieldProvider",
    "FieldSeeder",
    "FollowField",
    "Guidance",
    "Heading",
    "HeardBeacon",
    "HopField",
    "Intent",
    "KnowledgeExchange",
    "KnowledgeSharing",
    "KnowledgeSnapshot",
    "MemoryView",
    "Navigator",
    "ObjectiveField",
    "PlacedBeacon",
    "QuadtreeSensing",
    "RechargeConfig",
    "Regroup",
    "RegroupSeeder",
    "Resupply",
    "ResupplyConfig",
    "ReturnToDock",
    "SeededField",
    "SensedSurroundings",
    "Sensing",
    "SharingConfig",
    "SharingStrategy",
    "Steerer",
    "SteeringConfig",
    "SupplyBattery",
    "TreatByBeacon",
    "TreatInfestation",
    "TreatSeeder",
    "UtilityArbiter",
    "Vector2",
    "capabilities_for",
    "treat_beacon",
]
