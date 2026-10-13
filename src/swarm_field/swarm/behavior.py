"""Composable robot behaviours for the high-level cell model."""

from dataclasses import dataclass, field
from typing import Protocol

from swarm_field.contracts import (
    BodyType,
    CellId,
    Charge,
    Idle,
    Message,
    Observation,
    ReplaceBattery,
    RobotSelfState,
    Time,
    Treat,
)
from swarm_field.swarm.capabilities import CapabilitySet, capabilities_for
from swarm_field.swarm.memory import MemoryView
from swarm_field.swarm.provider import FieldProvider
from swarm_field.swarm.seeder import BatteryRequestSeeder, ExploreSeeder, TreatSeeder


@dataclass(frozen=True)
class FollowField:
    """Take one step along the grid potential field of a registered seeder.

    Attributes:
        seeder: The name of the seeder whose field to follow.
    """

    seeder: str


Intent = FollowField | Treat | Charge | ReplaceBattery | Idle
"""What a behaviour wants the robot to do if its candidate wins.

The brain's own counterpart of a contract ``Action``; it never crosses into the simulation.
``Treat``, ``Charge`` and ``Idle`` pass through unchanged. ``Move`` is deliberately absent: a
behaviour names the field to follow with ``FollowField``, and only the navigator decides the
step, for the winning candidate.
"""


@dataclass(frozen=True)
class ActionCandidate:
    """A behaviour's bid at a decision point: what it wants done, and how much.

    Attributes:
        action: What the robot should do if this candidate wins.
        score: How much the behaviour wants it; the highest score wins. The scale shared by all
            behaviours is settled together with the arbiter (#39).
        source: The name of the proposing behaviour.
        target: The cell this candidate commits the robot to, such as the field's target at the
            robot's cell, or ``None``. The arbiter uses it to tell a new target from the one
            the robot is already heading for.
    """

    action: Intent
    score: float
    source: str
    target: CellId | None


@dataclass(frozen=True)
class DecisionContext:
    """Read-only view handed to every behaviour at a decision point.

    Attributes:
        time: The time of the decision.
        self_state: The robot's perception of itself.
        sensed: What the robot sensed at this decision point (already merged into memory).
        memory: The robot's facts about the field and its peers.
        fields: Hands out the grid potential fields of the registered seeders by name, through
            ``field_for``. Behaviours read fields; they never move the robot themselves.
        inbox: The messages received at this decision point.
    """

    time: Time
    self_state: RobotSelfState
    sensed: list[Observation]
    memory: MemoryView
    fields: FieldProvider
    inbox: list[Message] = field(default_factory=list)
    capabilities: CapabilitySet | None = None


class Behavior(Protocol):
    """A single composable piece of decision logic, bidding for what the robot does next.

    At each decision point a behaviour reads the context and returns its candidates, or an
    empty tuple when it has nothing to propose. It decides whether it wants the robot, how much
    (``score``), and what to do if it wins (``action``). It may read the fields of seeders
    through ``ctx.fields``, e.g. to score from the potential at the robot's cell.

    A behaviour never moves the robot itself. To move, it proposes ``FollowField`` with a
    seeder's name, and the navigator steps along that field only if the candidate wins. A
    behaviour standing on its target (``field.target_for(position) == position``) proposes its
    on-the-spot action instead, such as ``Treat`` or ``Charge``.
    """

    name: str

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]: ...


class TreatInfestation:
    """Follow reported infestations and treat them when the robot reaches a target cell."""

    name: str = "treat-infestation"

    def __init__(self, seeder: str = TreatSeeder.name) -> None:
        self._seeder: str = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        position = ctx.self_state.position
        seeded = ctx.fields.field_for(self._seeder, position, ctx.time)
        if seeded is None:
            return ()
        target = seeded.field.target_for(position)
        if target == position:
            infestation = ctx.memory.infestation_at(position)
            if infestation is None or infestation.severity <= 0.0:
                return ()
            return (
                ActionCandidate(
                    action=Treat(cell=position),
                    score=100.0 + infestation.severity,
                    source=self.name,
                    target=position,
                ),
            )
        # Lower field potential means the robot is closer to an attractive target.
        score = -seeded.field.value_at(position)
        candidate = ActionCandidate(
            action=FollowField(self._seeder), score=score, source=self.name, target=target
        )
        return (candidate,)


class ExploreField:
    """Explore unknown cells, using the field built by ``ExploreSeeder``."""

    name: str = "explore-field"

    def __init__(self, seeder: str = ExploreSeeder.name) -> None:
        self._seeder = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        position = ctx.self_state.position
        seeded = ctx.fields.field_for(self._seeder, position, ctx.time)
        if seeded is None:
            return ()
        target = seeded.field.target_for(position)
        if target is None or target == position:
            return ()
        return (
            ActionCandidate(
                action=FollowField(self._seeder),
                score=1.0,
                source=self.name,
                target=target,
            ),
        )


class SupplyBattery:
    """Visit low-battery peers and replace their battery when co-located."""

    name: str = "supply-battery"
    low_battery_threshold: float = 0.25

    def __init__(self, seeder: str = BatteryRequestSeeder.name) -> None:
        self._seeder = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        if ctx.self_state.spare_batteries <= 0:
            return ()

        position = ctx.self_state.position
        seeded = ctx.fields.field_for(self._seeder, position, ctx.time)
        if seeded is None:
            return ()
        target = seeded.field.target_for(position)
        if target is None:
            return ()
        if target == position:
            eligible = [
                peer
                for peer in ctx.memory.peers()
                if peer.cell == position
                and peer.battery is not None
                and peer.battery <= self.low_battery_threshold
                and peer.body in (BodyType.SCOUT, BodyType.KILLER)
                and ctx.time - peer.observed_at <= 60_000
                and ctx.memory.confidence(peer, ctx.time) >= 0.5
            ]
            if not eligible:
                return ()
            recipient = min(eligible, key=lambda peer: (peer.battery, peer.peer or -1))
            if recipient.peer is None:
                return ()
            return (
                ActionCandidate(
                    action=ReplaceBattery(robot_id=recipient.peer),
                    score=100.0 + (1.0 - (recipient.battery or 0.0)),
                    source=self.name,
                    target=position,
                ),
            )

        return (
            ActionCandidate(
                action=FollowField(self._seeder),
                score=10.0,
                source=self.name,
                target=target,
            ),
        )


class ReturnToDock:
    """Return to a known dock when low on energy or when a battery robot is out of spares."""

    name: str = "return-to-dock"

    def __init__(self, seeder: str = "dock") -> None:
        self._seeder = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        profile = ctx.capabilities
        if profile is None:
            profile = capabilities_for(ctx.self_state.body)

        position = ctx.self_state.position
        seeded = ctx.fields.field_for(self._seeder, position, ctx.time)
        if seeded is None:
            return ()
        target = seeded.field.target_for(position)
        if target is None:
            return ()

        at_dock = target == position
        if at_dock and (
            ctx.self_state.battery < 1.0
            or (
                ctx.self_state.body is BodyType.BATTERY
                and ctx.self_state.spare_batteries < profile.spare_battery_capacity
            )
        ):
            return (
                ActionCandidate(
                    action=Charge(),
                    score=200.0,
                    source=self.name,
                    target=position,
                ),
            )

        if ctx.self_state.body is BodyType.BATTERY:
            needs_dock = (
                ctx.self_state.battery <= 0.25 or ctx.self_state.spare_batteries <= 0
            )
        else:
            # A scout/killer stays available to the swarm while a support robot may still help;
            # it heads to the dock only when the battery is nearly exhausted.
            needs_dock = ctx.self_state.battery <= 0.10
        if not needs_dock:
            return ()

        return (
            ActionCandidate(
                action=FollowField(self._seeder),
                score=200.0,
                source=self.name,
                target=target,
            ),
        )
