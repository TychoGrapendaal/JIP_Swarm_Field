"""Restocking: a killer whose doses run low returns to a dock to restock them."""

from swarm_field.contracts import AgentObjective, CellId, Charge
from swarm_field.swarm.behavior import ActionCandidate, DecisionContext, FollowField
from swarm_field.swarm.capabilities import capabilities_for
from swarm_field.swarm.recharge import ResupplyConfig


class Resupply:
    """Return to a dock when the doses run low, and restock on it.

    At or below ``ResupplyConfig.low_doses`` doses left, the bid to follow the dock field rises
    linearly from ``low_priority`` to ``empty_priority`` with no dose left; a full tank never
    bids, however small. On a dock it bids
    ``Charge``, which the dock serves by refilling the doses at once. It reads only the robot's
    own state and the dock cells, so it serves any brain: it names the field to follow, the
    ``DockSeeder`` of a seeded killer or a map-free dock field.
    """

    name: str = "resupply"

    def __init__(
        self,
        seeder: str = "dock",
        *,
        docks: tuple[CellId, ...] = (),
        config: ResupplyConfig | None = None,
    ) -> None:
        self._seeder: str = seeder
        self._docks: frozenset[CellId] = frozenset(docks)
        self._config: ResupplyConfig = ResupplyConfig()
        if config is not None:
            self._config = config

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        profile = ctx.capabilities
        if profile is None:
            profile = capabilities_for(ctx.self_state.body)
        if not profile.can_treat or not self._docks:
            return ()
        doses = ctx.self_state.doses_left
        # A full tank is never low, also one no larger than the threshold, or a killer would
        # restock again and again on the dock.
        if doses > self._config.low_doses or ctx.self_state.consumable >= 1.0:
            return ()
        position = ctx.self_state.position
        if position in self._docks:
            charge = Charge()
            score = self._config.dock_priority
            return (ActionCandidate(charge, score, self.name, position, AgentObjective.RECHARGE),)
        shortage = 1.0
        if self._config.low_doses > 0:
            shortage = 1.0 - doses / self._config.low_doses
        span = self._config.empty_priority - self._config.low_priority
        score = self._config.low_priority + shortage * span
        follow = FollowField(self._seeder)
        return (ActionCandidate(follow, score, self.name, None, AgentObjective.RECHARGE),)
