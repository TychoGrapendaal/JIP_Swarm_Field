"""Role rebalancing (#64): a swarm with fixed roles, and the same swarm switching roles.

``rebalancing_configurations`` turns a scenario into the two compared swarms, by name: ``fixed``,
the scenario as it is, and ``rebalancing``, the same robots in the same order and places, but
with every explorer and killer a ``RebalancingPreset`` that starts in the same role and may
switch. A role is a module swapped at a dock: a robot picked to switch drives to the nearest
dock and switches there (``RebalanceConfig.switch_at_dock``). A switching robot's killer half is
the scenario's own killer preset, so a swarm of ``SEEDED_BEST`` killers rebalances into robots
that treat like them, and a swarm of ``HOP_BEST`` killers into robots that treat by hop beacons.

``REBALANCING`` and ``REBALANCING_HOP`` are the experiment plan's default small river field
(``RIVER_SMALL``) with seeded and with hop killers, for the scenario viewer.
"""

from __future__ import annotations

import math
from dataclasses import replace

from swarm_field.contracts import BodyType, Time, TimeUnit
from swarm_field.experiments.presets import (
    HOP_BEST,
    ExplorerPreset,
    HopKillerPreset,
    RebalancingPreset,
    RobotPreset,
    SeededKillerPreset,
)
from swarm_field.experiments.scenario import RIVER_SMALL, RobotGroup, SwarmScenario
from swarm_field.swarm import RebalanceConfig

REVISIT_TARGET: Time = 30 * TimeUnit.MINUTE
"""How fresh the swarm keeps the field at the least: the revisit age ``monitors_needed`` sizes
the explorers for."""

REBALANCING_ARMS: tuple[str, ...] = ("fixed", "rebalancing")
"""The compared swarms of ``rebalancing_configurations``, by name."""

REBALANCING = RIVER_SMALL
"""The default small river field with ``SEEDED_BEST`` killers, whose swarms the viewer shows."""

REBALANCING_HOP = RIVER_SMALL.without_robots(BodyType.KILLER).add_robots(
    HOP_BEST, RIVER_SMALL.killers
)
"""``REBALANCING`` with ``HOP_BEST`` killers instead of seeded ones."""


def monitors_needed(scenario: SwarmScenario, revisit_age: Time = REVISIT_TARGET) -> int:
    """The explorers that sense every free cell of ``scenario`` once per ``revisit_age``, at best.

    An explorer senses at most one new cell per move; a robot decides again as soon as a move
    arrives, so a move takes the time to fly across one cell. This ignores overlap, so it is a
    lower bound.
    """
    cell_seconds = scenario.map.cell_size / scenario.robots.speed_mps_drone
    cells_per_minute = 60.0 / cell_seconds
    revisit_minutes = revisit_age / TimeUnit.MINUTE
    return math.ceil(scenario.map.free_cells() / (revisit_minutes * cells_per_minute))


def kept_explorers(scenario: SwarmScenario) -> int:
    """The explorers a rebalancing swarm keeps whatever the pests: ``monitors_needed``, and at
    least half of the explorers the scenario starts with.

    With the default drones at 8 m/s, ``monitors_needed`` is a single explorer on the small field:
    it ignores the overlap between explorers, which is large, so on its own it would let the
    swarm give up nearly all of them and stop finding pests.
    """
    explorers = 0
    for group in scenario.swarm:
        if isinstance(group.preset, ExplorerPreset):
            explorers += group.count
    return max(monitors_needed(scenario), math.ceil(explorers / 2))


def rebalancing_configurations(
    scenario: SwarmScenario, rebalance: RebalanceConfig | None = None
) -> dict[str, SwarmScenario]:
    """The fixed-role swarm of ``scenario`` and the same swarm rebalancing its roles, by name.

    In ``rebalancing`` every explorer and killer becomes a ``RebalancingPreset`` of the same
    starting role whose killer half is the scenario's killer preset, switching on a dock;
    other robots stay as they are. Without ``rebalance`` the robots keep ``kept_explorers``
    explorers whatever the pests, and default to the rest; a given ``rebalance`` is used with
    ``switch_at_dock`` on.

    Raises:
        ValueError: If the scenario has killers of more than one preset: a switching robot is
            one kind of killer.
    """
    config = rebalance
    if config is None:
        config = RebalanceConfig(min_monitors=kept_explorers(scenario))
    killer = _the_killer(scenario)
    switching = replace(config, switch_at_dock=True)
    groups: list[RobotGroup] = []
    for group in scenario.swarm:
        groups.append(RobotGroup(_switching(group.preset, switching, killer), group.count))
    return {"fixed": scenario, "rebalancing": replace(scenario, swarm=tuple(groups))}


def _the_killer(scenario: SwarmScenario) -> SeededKillerPreset | HopKillerPreset:
    """The one killer preset of ``scenario``, or a default seeded killer if it has none.

    Raises:
        ValueError: If it has killers of more than one preset.
    """
    killers: list[SeededKillerPreset | HopKillerPreset] = []
    for group in scenario.swarm:
        preset = group.preset
        is_killer = isinstance(preset, (SeededKillerPreset, HopKillerPreset))
        if is_killer and group.count > 0 and preset not in killers:
            killers.append(preset)
    if len(killers) > 1:
        raise ValueError("a rebalancing swarm has killers of one preset only")
    if not killers:
        return SeededKillerPreset()
    return killers[0]


def _switching(
    preset: RobotPreset, rebalance: RebalanceConfig, killer: SeededKillerPreset | HopKillerPreset
) -> RobotPreset:
    """``preset`` as a robot that may switch role, if it is an explorer or a killer."""
    prefix = "rebalancing"
    if isinstance(killer, HopKillerPreset):
        prefix = "rebalancing-hop"
    if isinstance(preset, ExplorerPreset):
        return RebalancingPreset(
            BodyType.SCOUT, rebalance, preset, killer, name=f"{prefix}-explorer"
        )
    if isinstance(preset, (SeededKillerPreset, HopKillerPreset)):
        return RebalancingPreset(BodyType.KILLER, rebalance, killer=killer, name=f"{prefix}-killer")
    return preset
