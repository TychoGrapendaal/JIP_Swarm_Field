"""Component A — the swarm algorithm (the robot brain).

A brain is a ``ComposedAgent`` assembled from a list of behaviors, each a small piece of
decision logic proposing scored candidate actions. An ``ObjectiveArbiter`` reduces those
proposals to one objective per decision point and holds the robot to it.
"""

from swarm_field.swarm.arbiter import (
    ArbiterChoice,
    ArbiterConfig,
    Objective,
    ObjectiveArbiter,
    ObjectiveProposal,
)
from swarm_field.swarm.behavior import (
    ActionCandidate,
    Behavior,
    DecisionContext,
    Explore,
    TreatInfestation,
)
from swarm_field.swarm.brain import ComposedAgent

__all__ = [
    "ActionCandidate",
    "ArbiterChoice",
    "ArbiterConfig",
    "Behavior",
    "ComposedAgent",
    "DecisionContext",
    "Explore",
    "Objective",
    "ObjectiveArbiter",
    "ObjectiveProposal",
    "TreatInfestation",
]