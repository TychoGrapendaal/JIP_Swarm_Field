"""Role rebalancing (#64): when, and how many, robots switch between monitoring and killing.

Every robot runs ``RebalanceRoles`` next to its other behaviours. From its own memory it counts
the monitors (scouts) and killers it knows of, and the pests it knows of, and works out how many
killers the swarm needs. When the split is off for long enough, and only the robots best placed
to switch do so, it bids ``SwitchRole``; the simulation then changes the robot's body and speed.

There is no central controller, so every robot sees roughly the same imbalance. To stop all of
them switching at once, robots rank themselves: with two killers missing, only the two monitors
nearest a known pest switch. A minimum time in a role, and an imbalance that has to persist,
keep robots from flipping back and forth on a single burst of news.

With ``RebalanceConfig.switch_at_dock`` a role is a module swapped at a dock: a robot picked to
switch drives to the nearest dock first, reporting ``AgentObjective.SWITCH_ROLE``, and switches
there. Its peers count it in its new role from then on, so that nobody else sets off for the
same gap, and the robots nearest a dock are picked first.

``BodyGate`` lets one brain carry the behaviours of both roles: a gated behaviour bids only
while the robot has one of its bodies, which the robot reads from its perception. The brain's
capability gate cannot do this alone: it lets a scout follow the TREAT field, since following a
field is a move. ``experiments.RebalancingPreset`` assembles such a robot.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot

from swarm_field.contracts import (
    AgentObjective,
    BodyType,
    CellId,
    RobotId,
    SwitchRole,
    Time,
    TimeUnit,
)
from swarm_field.swarm.behavior import ActionCandidate, Behavior, DecisionContext, FollowField
from swarm_field.swarm.memory import MemoryView

MONITOR = BodyType.SCOUT
KILLER = BodyType.KILLER


@dataclass(frozen=True)
class RebalanceConfig:
    """How many killers a robot thinks the swarm needs, and when it switches role.

    Attributes:
        min_killers: Killers kept even when no pest is known, so a new pest is treated quickly.
        min_monitors: Monitors kept whatever the pests, so the field is still sensed. Roughly the
            cells to keep fresh divided by what one monitor senses in a revisit age.
        pests_per_killer: How many known pests one killer is expected to handle.
        persistence: How long an imbalance has to last before a robot acts on it.
        role_dwell: How long a robot keeps a new role before it may switch again.
        peer_horizon: Peers last heard of longer ago than this are not counted.
        switch_score: The bid to switch; at or above the arbiter's override threshold, so a
            switch is not held back by a long-running commitment.
        switch_at_dock: Whether a robot switches only on a dock: once it is picked to switch, it
            drives to the nearest dock and switches there.
    """

    min_killers: int = 1
    min_monitors: int = 6
    pests_per_killer: float = 2.0
    persistence: Time = 2 * TimeUnit.MINUTE
    role_dwell: Time = 15 * TimeUnit.MINUTE
    peer_horizon: Time = 20 * TimeUnit.MINUTE
    switch_score: float = 0.95
    switch_at_dock: bool = False


@dataclass(frozen=True)
class RoleCount:
    """The roles and pests one robot knows of, and the split it concludes the swarm needs.

    Attributes:
        monitors: The robots known to be monitoring, this one included if it monitors.
        killers: The robots known to be killing, this one included if it kills.
        pests: The infested cells known of.
        killers_needed: How many killers the swarm needs for those pests.
    """

    monitors: tuple["KnownRobot", ...]
    killers: tuple["KnownRobot", ...]
    pests: tuple[CellId, ...]
    killers_needed: int

    def killer_gap(self) -> int:
        """Killers missing (positive) or in excess (negative)."""
        return self.killers_needed - len(self.killers)


@dataclass(frozen=True)
class KnownRobot:
    """A robot as this robot knows it: who, where, and in which role.

    Attributes:
        robot_id: The robot.
        cell: Where it was last known to be.
        body: The role it was last known to have.
    """

    robot_id: RobotId
    cell: CellId
    body: BodyType


def count_roles(
    me: KnownRobot, memory: MemoryView, now: Time, config: RebalanceConfig
) -> RoleCount:
    """Count the roles and pests known to ``me``, and the number of killers needed.

    The killers needed are ``min_killers`` plus one per ``pests_per_killer`` known pests, but
    never so many that fewer than ``min_monitors`` monitors are left. A peer on its way to a dock
    to switch role counts in its new role already, and is no longer a candidate to switch.
    """
    robots = [me]
    oldest = now - config.peer_horizon
    for peer in memory.peers():
        if peer.peer is None or peer.peer == me.robot_id or peer.observed_at < oldest:
            continue
        body = peer.body
        if peer.objective is AgentObjective.SWITCH_ROLE:
            body = _other_role(body)
        robots.append(KnownRobot(peer.peer, peer.cell, body))
    monitors: list[KnownRobot] = []
    killers: list[KnownRobot] = []
    for robot in robots:
        if robot.body is MONITOR:
            monitors.append(robot)
        elif robot.body is KILLER:
            killers.append(robot)
    pests: list[CellId] = []
    for pest in memory.infestations():
        if pest.severity > 0:
            pests.append(pest.cell)

    wanted = config.min_killers + math.ceil(len(pests) / config.pests_per_killer)
    most = max(config.min_killers, len(monitors) + len(killers) - config.min_monitors)
    needed = min(wanted, most)
    return RoleCount(tuple(monitors), tuple(killers), tuple(pests), needed)


def _other_role(body: BodyType) -> BodyType:
    """The role a robot of ``body`` switches to: a killer for a monitor, else a monitor."""
    if body is MONITOR:
        return KILLER
    return MONITOR


def ranked_for_switch(
    candidates: Sequence[KnownRobot],
    pests: Sequence[CellId],
    to_killer: bool,
    docks: Sequence[CellId] = (),
) -> list[KnownRobot]:
    """The candidates in the order they should switch role, first one first.

    Monitors become killers nearest a known pest first; killers become monitors furthest from
    any known pest first. With ``docks``, where a switch has to happen, both go nearest a dock
    first. Ties, and the case with no known pest, go to the lowest robot id.
    """
    if docks:

        def to_dock(robot: KnownRobot) -> float:
            return min(hypot(robot.cell[0] - d[0], robot.cell[1] - d[1]) for d in docks)

        return sorted(candidates, key=lambda robot: (to_dock(robot), robot.robot_id))

    def distance(robot: KnownRobot) -> float:
        if not pests:
            return 0.0
        return min(hypot(robot.cell[0] - p[0], robot.cell[1] - p[1]) for p in pests)

    if to_killer:
        return sorted(candidates, key=lambda robot: (distance(robot), robot.robot_id))
    return sorted(candidates, key=lambda robot: (-distance(robot), robot.robot_id))


class RebalanceRoles:
    """Switches this robot between monitor and killer when the swarm's split is off.

    It bids only when all of these hold: the robot is a monitor or a killer, it has kept its
    role for ``role_dwell``, the imbalance in its own count has lasted ``persistence``, and the
    robot ranks among the ones that should switch. With ``switch_at_dock`` it then bids to
    follow the dock field until it stands on a dock, and to switch there; once it has set off,
    it keeps going without waiting for ``persistence`` again, for as long as it still ranks
    among the robots the gap needs.

    Attributes:
        config: When and how much to rebalance.
        docks: The dock cells, used when switching happens only at a dock.
        dock_field: The field that leads to a dock.
    """

    name: str = "rebalance-roles"

    def __init__(
        self,
        config: RebalanceConfig | None = None,
        docks: Sequence[CellId] = (),
        dock_field: str = "dock",
    ) -> None:
        self.config: RebalanceConfig = RebalanceConfig()
        if config is not None:
            self.config = config
        self.docks: tuple[CellId, ...] = tuple(docks)
        self.dock_field: str = dock_field
        self._heading_to_dock: bool = False
        self._body: BodyType | None = None
        self._role_since: Time | None = None
        self._gap_sign: int = 0
        self._gap_since: Time = 0

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        state = ctx.self_state
        now = ctx.time
        self._note_role(state.body, now)
        if state.body not in (MONITOR, KILLER):
            return ()
        me = KnownRobot(state.robot_id, state.position, state.body)
        count = count_roles(me, ctx.memory, now, self.config)
        gap = count.killer_gap()
        lasted = self._imbalance_has_lasted(gap, now)
        on_the_way = self._heading_to_dock
        self._heading_to_dock = False
        if not lasted and not on_the_way:
            return ()
        if self._role_since is not None and now - self._role_since < self.config.role_dwell:
            return ()
        at_dock = self.config.switch_at_dock and bool(self.docks)
        if self.config.switch_at_dock and not self.docks:
            return ()

        to_killer = gap > 0
        if gap == 0:
            return ()
        if to_killer and state.body is not MONITOR:
            return ()
        if not to_killer and state.body is not KILLER:
            return ()
        candidates = count.monitors
        target = KILLER
        if not to_killer:
            candidates = count.killers
            target = MONITOR
        docks: tuple[CellId, ...] = ()
        if at_dock:
            docks = self.docks
        ranked = ranked_for_switch(candidates, count.pests, to_killer, docks)
        chosen = [robot.robot_id for robot in ranked[: abs(gap)]]
        if state.robot_id not in chosen:
            return ()
        score = self.config.switch_score
        objective = AgentObjective.SWITCH_ROLE
        if at_dock and state.position not in self.docks:
            self._heading_to_dock = True
            to_dock = FollowField(self.dock_field)
            return (ActionCandidate(to_dock, score, self.name, None, objective),)
        switch = SwitchRole(target)
        return (ActionCandidate(switch, score, self.name, state.position, objective),)

    def _note_role(self, body: BodyType, now: Time) -> None:
        """Remember when the robot last changed role, read off its perceived body."""
        if self._body is not None and body is not self._body:
            self._role_since = now
            self._gap_sign = 0
        self._body = body

    def _imbalance_has_lasted(self, gap: int, now: Time) -> bool:
        """Whether an imbalance in the same direction as ``gap`` has lasted ``persistence``."""
        sign = (gap > 0) - (gap < 0)
        if sign != self._gap_sign:
            self._gap_sign = sign
            self._gap_since = now
        return sign != 0 and now - self._gap_since >= self.config.persistence


class BodyGate:
    """Lets ``inner`` bid only while the robot has one of ``bodies``.

    A robot that may switch role carries the behaviours of every role it can take; the gate
    keeps, for example, treating to killers. The robot's body is read from its perception,
    so the gate follows a switch at once.

    Attributes:
        inner: The gated behaviour.
        bodies: The bodies the behaviour bids for.
        name: The inner behaviour's name, so its bids keep their source.
    """

    def __init__(self, inner: Behavior, bodies: Sequence[BodyType]) -> None:
        self.inner: Behavior = inner
        self.bodies: frozenset[BodyType] = frozenset(bodies)
        self.name: str = inner.name

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        if ctx.self_state.body not in self.bodies:
            return ()
        return self.inner.propose(ctx)
