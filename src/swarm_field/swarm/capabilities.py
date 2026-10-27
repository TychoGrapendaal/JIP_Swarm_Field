"""Per-body physical and operational capabilities.

The profile is the single source of truth used by the brain assembly, its capability gate,
and the high-level sensing model. The speeds and endurance values are initial simulation
assumptions based on the project specification, not hardware guarantees.
"""

from dataclasses import dataclass

from swarm_field.contracts.actions import (
    Action,
    Charge,
    Idle,
    Move,
    ReplaceBattery,
    SwitchRole,
    Treat,
)
from swarm_field.contracts.ids import BodyType


@dataclass(frozen=True)
class CapabilitySet:
    """Capabilities and high-level physical parameters of one robot body.

    ``action_types`` contains the concrete action classes the robot can emit. Behaviours use
    ``can_fly`` and sensing radii for physical constraints; ``ComposedAgent`` applies the action
    gate independently of how behaviours are configured.
    """

    body: BodyType
    action_types: frozenset[type[Action]]
    can_fly: bool
    speed_mps: float
    terrain_sensing_radius: int
    peer_sensing_radius: int
    endurance_seconds: float
    dock_recharge_seconds: float = 0.0
    spare_battery_capacity: int = 0
    battery_replacement_seconds: float = 224.0
    can_detect_pests: bool = False
    can_treat: bool = False
    can_supply_batteries: bool = False

    def can(self, action_type: type[Action]) -> bool:
        """Return whether this body is allowed to emit ``action_type``."""
        return action_type in self.action_types


_COMMON_ACTIONS = frozenset({Move, Charge, Idle})
_ROLE_ACTIONS = frozenset({SwitchRole})
"""Scouts and killers may switch into each other's role (#64); a battery robot may not."""

DOCK_SWAP_SECONDS = 5.0 * 60.0
"""How long a dock takes to fill an empty battery. A dock swaps batteries, which takes a few
minutes whatever the body, so every body charges from empty to full in this time."""

BODY_CAPABILITIES: dict[BodyType, CapabilitySet] = {
    BodyType.SCOUT: CapabilitySet(
        body=BodyType.SCOUT,
        action_types=_COMMON_ACTIONS | _ROLE_ACTIONS,
        can_fly=True,
        speed_mps=8.0,
        terrain_sensing_radius=1,
        peer_sensing_radius=1,
        endurance_seconds=30.0 * 60.0,
        dock_recharge_seconds=DOCK_SWAP_SECONDS,
        can_detect_pests=True,
    ),
    BodyType.KILLER: CapabilitySet(
        body=BodyType.KILLER,
        action_types=frozenset({Move, Treat, Charge, Idle}) | _ROLE_ACTIONS,
        can_fly=False,
        speed_mps=1.0,
        terrain_sensing_radius=1,
        peer_sensing_radius=1,
        endurance_seconds=8.0 * 60.0 * 60.0,
        dock_recharge_seconds=DOCK_SWAP_SECONDS,
        can_detect_pests=True,
        can_treat=True,
    ),
    BodyType.BATTERY: CapabilitySet(
        body=BodyType.BATTERY,
        action_types=frozenset({Move, ReplaceBattery, Charge, Idle}),
        can_fly=False,
        speed_mps=1.0,
        terrain_sensing_radius=1,
        peer_sensing_radius=1,
        endurance_seconds=8.0 * 60.0 * 60.0,
        dock_recharge_seconds=DOCK_SWAP_SECONDS,
        spare_battery_capacity=3,
        battery_replacement_seconds=224.0,
        can_supply_batteries=True,
    ),
}


def capabilities_for(body: BodyType) -> CapabilitySet:
    """Return the canonical capability profile for ``body``."""
    return BODY_CAPABILITIES[body]
