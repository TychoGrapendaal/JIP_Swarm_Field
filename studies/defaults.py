"""The default settings of the studies: the standard maps, the swarm, the pests and the robots.

A study starts from ``default_scenario(map_name)`` and changes only what its question is about,
so that every study measures the same swarm on the same fields. The settings are those of the
experiment plan (``docs/experiment-plan.md``), built from its ready parts:

- **Maps** (``MAPS``): the small field with one river, a bridge and rocks (``SMALL_RIVER_MAP``,
  2 h), the large field with two rivers, three bridges and rocks (``LARGE_RIVERS_MAP``, 4 h),
  and both sizes as an open field without obstacles.
- **Robots** (``DEFAULT_ROBOTS``): 10 explorers (drones at 8 m/s that fly over obstacles) and 4
  ``SEEDED_BEST`` killers (ground robots at 1 m/s), a 100 m radio (10 cells), pests sensed in the
  robot's own cell and obstacles in its direct neighbours, perfect detection and treatment, no
  battery drain and doses without end, no rebalancing. Every preset starts on a jittered
  lattice of its own, and the robots learn the obstacles by sensing them.
- **Pests** (``STATIC_PESTS``): ``pests_per_hour`` pests at the start and as many new ones every
  hour, static: they neither grow nor spread.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from swarm_field.contracts import BodyType
from swarm_field.experiments import (
    DEFAULT_EXPLORERS,
    DEFAULT_KILLERS,
    DEFAULT_ROBOTS,
    EXPLORER,
    HOP_BEST,
    LARGE_FIELD_HOURS,
    LARGE_OPEN_MAP,
    LARGE_RIVERS_MAP,
    SEEDED_BEST,
    SMALL_FIELD_HOURS,
    SMALL_OPEN_MAP,
    SMALL_RIVER_MAP,
    STATIC_PESTS,
    CustomPreset,
    MapConfig,
    PestConfig,
    SwarmScenario,
)
from swarm_field.runner import BrainSpec

__all__ = [
    "DEFAULT_ROBOTS",
    "EXPLORERS",
    "KILLERS",
    "MAPS",
    "StandardMap",
    "default_pests",
    "default_scenario",
    "killer_methods",
    "measured_run",
    "with_grid_sweepers",
    "with_radio",
]

EXPLORERS = DEFAULT_EXPLORERS
KILLERS = DEFAULT_KILLERS


@dataclass(frozen=True)
class StandardMap:
    """One of the standard fields, with the run length and pest rate it is studied at.

    Attributes:
        name: The map's name in arm names and on the command line.
        label: How the map is described in figure captions.
        map: The field.
        hours: How long a run on it lasts.
        pests_per_hour: The pests at the start and the new ones every hour, chosen so that
            the default swarm is neither idle nor overworked.
    """

    name: str
    label: str
    map: MapConfig
    hours: int
    pests_per_hour: float


MAPS: dict[str, StandardMap] = {
    "rivers-small": StandardMap(
        "rivers-small",
        "32 x 32 cells, one river and rocks",
        SMALL_RIVER_MAP,
        hours=SMALL_FIELD_HOURS,
        pests_per_hour=15.0,
    ),
    "rivers-large": StandardMap(
        "rivers-large",
        "64 x 64 cells, two rivers and rocks",
        LARGE_RIVERS_MAP,
        hours=LARGE_FIELD_HOURS,
        pests_per_hour=60.0,
    ),
    "open-small": StandardMap(
        "open-small",
        "32 x 32 cells, open field",
        SMALL_OPEN_MAP,
        hours=SMALL_FIELD_HOURS,
        pests_per_hour=15.0,
    ),
    "open-large": StandardMap(
        "open-large",
        "64 x 64 cells, open field",
        LARGE_OPEN_MAP,
        hours=LARGE_FIELD_HOURS,
        pests_per_hour=60.0,
    ),
}
"""The standard maps by name; the large fields have four times the cells, and the pests."""


def default_pests(standard: StandardMap) -> PestConfig:
    """Static pests: ``pests_per_hour`` at the start and every hour, perfectly treated."""
    pests_per_hour = standard.pests_per_hour
    return replace(STATIC_PESTS, pests_per_hour=pests_per_hour, initial_pests=round(pests_per_hour))


def default_scenario(map_name: str) -> SwarmScenario:
    """The default swarm of explorers and ``SEEDED_BEST`` killers on the standard map
    ``map_name``, for the map's hours, measured from the start.

    Raises:
        KeyError: If there is no standard map of that name.
    """
    standard = MAPS[map_name]
    scenario = (
        SwarmScenario()
        .with_map(standard.map)
        .with_pests(default_pests(standard))
        .with_robot_config(DEFAULT_ROBOTS)
        .add_robots(EXPLORER, EXPLORERS)
        .add_robots(SEEDED_BEST, KILLERS)
    )
    return scenario.with_hours(standard.hours, warmup_hours=0)


def measured_run(
    scenario: SwarmScenario, map_name: str, warmup_hours: int = 1, hours: int | None = None
) -> SwarmScenario:
    """``scenario`` as a run of ``measure_pest_response`` for the standard map's hours, or
    ``hours`` if given.

    ``infested`` leaves out the first ``warmup_hours``, while the swarm finds the pests that were
    there at the start.
    """
    if hours is None:
        hours = MAPS[map_name].hours
    return scenario.with_hours(hours, warmup_hours)


def with_radio(scenario: SwarmScenario, radio_range_m: float) -> SwarmScenario:
    """``scenario`` with every robot's radio reaching ``radio_range_m``."""
    return scenario.with_robot_config(replace(scenario.robots, radio_range_m=radio_range_m))


def killer_methods(scenario: SwarmScenario) -> dict[str, SwarmScenario]:
    """``scenario`` with its killers as the tuned seeded (``SEEDED_BEST``) and the tuned
    hop-based killers (``HOP_BEST``), as many as it has, after its other robots.

    The hop killers' beacons follow the scenario's radio range, so build the methods after
    setting the radio.
    """
    killers = scenario.killers
    others = scenario.without_robots(BodyType.KILLER)
    return {
        "seeded": others.add_robots(SEEDED_BEST, killers),
        "hop": others.add_robots(HOP_BEST, killers),
    }


def with_grid_sweepers(
    scenario: SwarmScenario,
    count: int,
    body: BodyType,
    brain: Callable[[int, int | None], BrainSpec],
) -> SwarmScenario:
    """``scenario`` with ``count`` grid sweepers of ``body`` added, each on its own strip.

    ``brain`` is ``grid_sweep_brain`` for killers or ``grid_explorer_brain`` for explorers. Every
    sweeper is a preset of its own, so that its strip is set by its number rather than by where
    it starts.
    """
    for strip in range(count):
        preset = CustomPreset(f"grid-{strip}", body, brain(count, strip))
        scenario = scenario.add_robots(preset, 1)
    return scenario
