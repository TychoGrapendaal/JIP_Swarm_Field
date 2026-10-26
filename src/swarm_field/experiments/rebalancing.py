"""Role rebalancing (#64): the swarm of a scenario with fixed roles, and the same one switching.

``rebalancing_configurations`` turns a scenario into the two compared swarms, by name: ``fixed``,
the scenario as it is, and ``rebalancing``, the same robots in the same order and places, but
with every explorer and killer a ``RebalancingPreset`` that starts in the same role and may
switch. A swarm of seeded killers rebalances into robots that treat like seeded killers, and a
swarm of hop killers into robots that treat by hop beacons. ``REBALANCING`` is the scenario the
comparison in ``docs/research/rebalancing.md`` runs: eight explorers and six seeded killers on
the river field, with a dock in every corner; ``REBALANCING_HOP`` is the same with hop killers.
"""

from __future__ import annotations

import math
from dataclasses import replace

from swarm_field.contracts import BodyType, Time, TimeUnit
from swarm_field.experiments.maps import MapConfig
from swarm_field.experiments.presets import (
    EXPLORER,
    HOP_KILLER,
    SEEDED_KILLER,
    ExplorerPreset,
    HopKillerPreset,
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


REBALANCING_HOP = REBALANCING.without_robots(BodyType.KILLER).add_robots(HOP_KILLER, 6)
"""``REBALANCING`` with six hop killers instead of seeded killers."""


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

    In ``rebalancing`` every explorer, seeded killer and hop killer becomes a ``RebalancingPreset``
    of the same starting role; other robots stay as they are. The robots treat by hop beacons,
    with the beacons of the scenario's hop killers, if the scenario has hop killers, and like
    seeded killers otherwise. Without ``rebalance`` the robots keep ``monitors_needed`` explorers
    whatever the pests, and default to the rest.

    Raises:
        ValueError: If the scenario has both seeded and hop killers: a switching robot treats
            in one way.
    """
    config = rebalance
    if config is None:
        config = RebalanceConfig(min_monitors=monitors_needed(scenario))
    hop: HopKillerPreset | None = None
    seeded = False
    for group in scenario.swarm:
        if isinstance(group.preset, HopKillerPreset) and group.count > 0:
            hop = group.preset
        elif isinstance(group.preset, SeededKillerPreset) and group.count > 0:
            seeded = True
    if hop is not None and seeded:
        raise ValueError("a rebalancing swarm treats either like seeded or like hop killers")
    groups: list[RobotGroup] = []
    for group in scenario.swarm:
        groups.append(RobotGroup(_switching(group.preset, config, hop), group.count))
    return {"fixed": scenario, "rebalancing": replace(scenario, swarm=tuple(groups))}


def _switching(
    preset: RobotPreset, rebalance: RebalanceConfig, hop: HopKillerPreset | None
) -> RobotPreset:
    """``preset`` as a robot that may switch role, if it is an explorer or a killer.

    With ``hop``, the swarm's hop killers, the robot treats by their beacons.
    """
    response = "seeded"
    beacons = None
    prefix = "rebalancing"
    if hop is not None:
        response = "hop"
        beacons = hop.beacons
        prefix = "rebalancing-hop"
    if isinstance(preset, ExplorerPreset):
        return RebalancingPreset(
            BodyType.SCOUT,
            rebalance,
            preset.explore,
            preset.revisit_sweeps,
            response=response,
            beacons=beacons,
            name=f"{prefix}-explorer",
        )
    if isinstance(preset, (SeededKillerPreset, HopKillerPreset)):
        return RebalancingPreset(
            BodyType.KILLER,
            rebalance,
            response=response,
            beacons=beacons,
            name=f"{prefix}-killer",
        )
    return preset
