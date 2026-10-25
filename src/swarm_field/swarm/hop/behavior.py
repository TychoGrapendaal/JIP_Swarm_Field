"""The pest response of the hop method: follow the hop field, and treat on arrival."""

from swarm_field.contracts import AgentObjective, InfestationObservation, Treat
from swarm_field.swarm.behavior import TREAT_SCORE, ActionCandidate, DecisionContext, FollowField
from swarm_field.swarm.hop.field import HopField
from swarm_field.swarm.hop.memory import BeaconView
from swarm_field.swarm.hop.pull import hop_pull
from swarm_field.swarm.hop.relay import BeaconConfig


class TreatByBeacon:
    """The pest-response behaviour on hop beacons: follow the hop field and treat on arrival.

    It knows of a pest only through the robot's beacons and its own sensing, never through
    remembered pest cells: it reads ``DecisionContext.sensed``, not ``DecisionContext.memory``.
    Its bid is the strength of the strongest attracting beacon of the ``hop_pull`` the hop field
    follows, so it never commits to a field with nothing to guide to, also when killers'
    beacons push.
    """

    name: str = "treat-hop"

    def __init__(
        self, config: BeaconConfig, beacons: BeaconView, field: str = HopField.name
    ) -> None:
        self._config: BeaconConfig = config
        self._beacons: BeaconView = beacons
        self._field: str = field

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        position = ctx.self_state.position
        for fact in ctx.sensed:
            if not isinstance(fact, InfestationObservation):
                continue
            if fact.cell == position and fact.severity > 0:
                treat = Treat(position)
                return (
                    ActionCandidate(treat, TREAT_SCORE, self.name, position, AgentObjective.TREAT),
                )

        pull = hop_pull(self._beacons, position, ctx.time, self._config)
        if pull is None:
            return ()
        follow = FollowField(self._field)
        return (ActionCandidate(follow, pull.strength, self.name, None, AgentObjective.TREAT),)
