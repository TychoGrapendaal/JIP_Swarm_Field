"""Component A — the swarm algorithm (the robot brain).

A brain is a ``ComposedAgent`` assembled from a list of behaviors, each a small piece of
decision logic proposing scored candidate actions, plus a ``Navigator`` that turns a goal
(A.1) or a resolved hop field (A.4) into concrete movement. Kept minimal for now.
"""

from swarm_field.contracts import HopFieldEntry, NavGoal, NavMode, Vector2
from swarm_field.swarm.behavior import (
    ActionCandidate,
    Behavior,
    DecisionContext,
    Explore,
    TreatInfestation,
)
from swarm_field.swarm.brain import ComposedAgent
from swarm_field.swarm.navigator import (
    ChaseTarget,
    Navigator,
    Sensing,
    SteeringConfig,
    resolve_chase_target,
)

__all__ = [
    "ActionCandidate",
    "Behavior",
    "ChaseTarget",
    "ComposedAgent",
    "DecisionContext",
    "Explore",
    "HopFieldEntry",
    "NavGoal",
    "NavMode",
    "Navigator",
    "Sensing",
    "SteeringConfig",
    "TreatInfestation",
    "Vector2",
    "resolve_chase_target",
]
