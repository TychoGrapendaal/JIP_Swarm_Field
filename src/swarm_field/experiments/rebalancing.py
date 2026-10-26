"""Role rebalancing (#64): the swarm of a scenario with fixed roles, and the same one switching.

``rebalancing_configurations`` turns a scenario into the two compared swarms, by name: ``fixed``,
the scenario as it is, and ``rebalancing``, the same robots in the same order and places, but
with every explorer and seeded killer a ``RebalancingPreset`` that starts in the same role and may
switch. ``REBALANCING`` is the scenario the comparison in ``docs/research/rebalancing.md`` runs:
eight explorers and six seeded killers on the river field, with a dock in every corner.
"""

from __future__ import annotations

import math
from dataclasses import replace

from swarm_field.contracts import BodyType, Time, TimeUnit
from swarm_field.experiments.maps import MapConfig
from swarm_field.experiments.presets import (
    EXPLORER,
    SEEDED_KILLER,
    ExplorerPreset,
    RebalancingPreset,
    RobotPreset,
    SeededKillerPreset,
)
from swarm_field.experiments.scenario import (
    FAST_TREATMENT,
    PestConfig,
    RobotConfig,
    RobotGroup,
    SwarmScenario,
)
from swarm_field.swarm import RebalanceConfig

REVISIT_TARGET: Time = 30 * TimeUnit.MINUTE
"""How fresh the swarm keeps the field at the least: the revisit age ``monitors_needed`` sizes
the explorers for."""

REBALANCING = (
    SwarmScenario(
        map=MapConfig(docks=((3, 3), (28, 3), (3, 28), (28, 28))),
        pests=PestConfig(pests_per_hour=20.0, treatment_rate=FAST_TREATMENT, treatment_doses=100),
        robots=RobotConfig(battery_capacity=10.0),
    )
    .add_robots(EXPLORER, 8)
    .add_robots(SEEDED_KILLER, 6)
)
"""Eight explorers and six seeded killers on the river field, with a dock in every corner.

Six killers are more than 20 pests an hour need while the field is clean and fewer than a burst
of pests does, which is the split rebalancing has to follow. Batteries hold ten charges and
killers carry a hundred doses, as in ``configs/seeded_vs_hop.json``, so that neither queueing at
the docks nor running out of doses decides the comparison: with ten doses the fixed killers run
dry within the run, while a robot that becomes a killer brings a full supply.
"""


def monitors_needed(scenario: SwarmScenario, revisit_age: Time = REVISIT_TARGET) -> int:
    """The explorers that sense every free cell of ``scenario`` once per ``revisit_age``, at best.

    An explorer senses at most one new cell per decision, and no faster than it can fly across
    one, so this ignores overlap and is a lower bound.
    """
    decision_seconds = scenario.robots.decision_interval / TimeUnit.SECOND
    cell_seconds = scenario.map.cell_size / scenario.robots.speed_mps_drone
    cells_per_minute = 60.0 / max(decision_seconds, cell_seconds)
    revisit_minutes = revisit_age / TimeUnit.MINUTE
    return math.ceil(scenario.map.free_cells() / (revisit_minutes * cells_per_minute))


def rebalancing_configurations(
    scenario: SwarmScenario, rebalance: RebalanceConfig | None = None
) -> dict[str, SwarmScenario]:
    """The fixed-role swarm of ``scenario`` and the same swarm rebalancing its roles, by name.

    In ``rebalancing`` every explorer and seeded killer becomes a ``RebalancingPreset`` of the
    same starting role; other robots stay as they are. Without ``rebalance`` the robots keep
    ``monitors_needed`` explorers whatever the pests, and default to the rest.
    """
    config = rebalance
    if config is None:
        config = RebalanceConfig(min_monitors=monitors_needed(scenario))
    groups: list[RobotGroup] = []
    for group in scenario.swarm:
        groups.append(RobotGroup(_switching(group.preset, config), group.count))
    return {"fixed": scenario, "rebalancing": replace(scenario, swarm=tuple(groups))}


def _switching(preset: RobotPreset, rebalance: RebalanceConfig) -> RobotPreset:
    """``preset`` as a robot that may switch role, if it is an explorer or a seeded killer."""
    if isinstance(preset, ExplorerPreset):
        return RebalancingPreset(
            BodyType.SCOUT,
            rebalance,
            preset.explore,
            preset.revisit_sweeps,
            name="rebalancing-explorer",
        )
    if isinstance(preset, SeededKillerPreset):
        return RebalancingPreset(BodyType.KILLER, rebalance, name="rebalancing-killer")
    return preset
