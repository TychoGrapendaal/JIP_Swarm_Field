"""Recharging at docks: a seeder per dock, and the behaviour that sends the robot to one.

``ReturnToDock`` is the battery behaviour ``docs/research/arbiter.md`` anticipates. It is the one
behaviour that reaches every rule of ``UtilityArbiter``: its bid ramps up as the battery drains
(competing through the effort margin), crosses the override threshold when the robot must go
now, bids one target per dock (so changing dock would be a target switch within one objective),
and disappears once the robot is charged (finishing the objective).

Docks are fixed infrastructure, so every robot is assembled knowing where they are. Each dock has
its own ``DockSeeder``, so the arbiter's commitment to a dock is also the field the robot follows.
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from math import hypot

from swarm_field.contracts import AgentObjective, CellId, Charge, Time, TimeUnit
from swarm_field.swarm.behavior import ActionCandidate, DecisionContext, FollowField
from swarm_field.swarm.field import Seed
from swarm_field.swarm.memory import MemoryView

_RECHARGE = AgentObjective.RECHARGE


def dock_seeder_name(index: int) -> str:
    """The registry and field-provider name of the seeder of the dock at ``index``."""
    return f"dock-{index}"


class DockSeeder:
    """Seeds the field of one dock: draws the robot to that dock's cell and nowhere else.

    The seed never changes, so the field is rebuilt only when the terrain or the robot's block
    does.

    Attributes:
        name: The seeder's name, one per dock.
        dock: The dock's cell.
    """

    window_blocks: int = 2

    def __init__(self, name: str, dock: CellId) -> None:
        self.name: str = name
        self.dock: CellId = dock

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        return self.dock

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        return (Seed(self.dock, 0.0),)


@dataclass(frozen=True)
class RechargeConfig:
    """When a robot wants a dock, and how much.

    Attributes:
        low: Below this battery the robot starts bidding for a dock.
        critical: At or below this battery the robot bids ``override_score`` for its nearest
            dock, so it goes whatever else it was doing.
        max_ramp: The bid just above ``critical``; the ramp runs from 0 at ``low`` up to it.
            Kept below the arbiter's override threshold so the ramp competes through the margin.
        override_score: The bid of a robot that must go now, and of charging on a dock. Above
            ``TreatInfestation``'s 0.99, so the battery can interrupt a treatment.
        crowding_penalty: The fraction of its bid a dock loses per peer known to be heading for
            it, so robots spread over the docks.
        crowding_cap: At most this many peers count against a dock.
        peer_freshness: A peer's report of heading for a dock counts for this long.
    """

    low: float = 0.45
    critical: float = 0.2
    max_ramp: float = 0.85
    override_score: float = 1.0
    crowding_penalty: float = 0.12
    crowding_cap: int = 3
    peer_freshness: Time = 2 * TimeUnit.MINUTE


class ReturnToDock:
    """The battery behaviour: go to a dock when the battery runs low, and charge there.

    - On a dock with a low battery it bids ``Charge`` at ``override_score``.
    - At or below ``critical`` it bids ``override_score`` to follow its nearest dock's field.
    - Between ``critical`` and ``low`` it bids for every dock, on a ramp that grows as the
      battery drains, less for each peer known to be heading to the same dock.
    - Above ``low`` it bids nothing.
    """

    name: str = "return-to-dock"

    def __init__(
        self,
        docks: Sequence[CellId],
        seeders: Sequence[str],
        config: RechargeConfig | None = None,
    ) -> None:
        if len(docks) != len(seeders):
            raise ValueError("every dock needs exactly one seeder")
        if not docks:
            raise ValueError("return-to-dock needs at least one dock")
        self._docks: tuple[CellId, ...] = tuple(docks)
        self._seeders: tuple[str, ...] = tuple(seeders)
        self.config: RechargeConfig = RechargeConfig()
        if config is not None:
            self.config = config

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        config = self.config
        battery = ctx.self_state.battery
        position = ctx.self_state.position
        if battery >= config.low:
            return ()
        if position in self._docks:
            charge = ActionCandidate(
                Charge(), config.override_score, self.name, position, _RECHARGE
            )
            return (charge,)
        if battery <= config.critical:
            index = self._nearest_dock(position)
            follow = FollowField(self._seeders[index])
            dock = self._docks[index]
            return (ActionCandidate(follow, config.override_score, self.name, dock, _RECHARGE),)

        urgency = (config.low - battery) / (config.low - config.critical)
        base = config.max_ramp * urgency
        crowding = self._peers_heading_to_docks(ctx)
        bids: list[ActionCandidate] = []
        for dock, seeder in zip(self._docks, self._seeders, strict=True):
            crowd = min(crowding.get(dock, 0), config.crowding_cap)
            score = base * (1.0 - config.crowding_penalty * crowd)
            bids.append(ActionCandidate(FollowField(seeder), score, self.name, dock, _RECHARGE))
        return tuple(bids)

    def _nearest_dock(self, position: CellId) -> int:
        """The index of the dock nearest ``position`` in a straight line."""
        distances = [hypot(d[0] - position[0], d[1] - position[1]) for d in self._docks]
        return distances.index(min(distances))

    def _peers_heading_to_docks(self, ctx: DecisionContext) -> dict[CellId, int]:
        """How many other robots recently reported heading for each dock to recharge."""
        crowding: dict[CellId, int] = {}
        oldest = ctx.time - self.config.peer_freshness
        for peer in ctx.memory.peers():
            if peer.peer == ctx.self_state.robot_id or peer.observed_at < oldest:
                continue
            target = peer.target
            if peer.objective is not _RECHARGE or target is None or target not in self._docks:
                continue
            crowding[target] = crowding.get(target, 0) + 1
        return crowding
