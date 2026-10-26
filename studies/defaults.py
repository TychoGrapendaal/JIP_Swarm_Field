"""The default settings of the studies: the standard maps, the swarm, the pests and the robots.

A study starts from ``default_scenario(map_name)`` and changes only what its question is about,
so that every study measures the same swarm on the same fields:

- **Maps** (``MAPS``): the small field with one river, a bridge and rocks (32 x 32, 2 h), the
  large field with two rivers, three bridges and rocks (64 x 64, 4 h), and both sizes as an
  open field without obstacles.
- **Robots**: 10 explorers (drones at 8 m/s that fly over obstacles) and 4 killers (ground
  robots at 1 m/s), a 100 m radio (10 cells), pests sensed in the robot's own cell and
  obstacles in its direct neighbours, perfect detection and treatment, no battery drain and
  doses without end, no rebalancing. The robots start side by side on the bottom row and learn
  the obstacles by sensing them.
- **Pests**: ``pests_per_hour`` pests at the start and as many new ones every hour, static: they
  neither grow nor spread.

A robot moves at most one cell per decision, so robots decide every ``DECISION_INTERVAL``, the
time an 8 m/s drone takes for a 10 m cell; at the scenarios' usual 15 s every robot would move
one cell per 15 s whatever its speed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from swarm_field.contracts import BodyType, Time, TimeUnit
from swarm_field.experiments import (
    EXPLORER,
    FAST_TREATMENT,
    SEEDED_KILLER,
    CustomPreset,
    MapConfig,
    PestConfig,
    PestRunSpec,
    RobotConfig,
    SwarmScenario,
    pest_response_configurations,
    river_and_rocks,
    two_rivers_and_rocks,
)
from swarm_field.runner import BrainSpec
from swarm_field.simulation import MetricsConfig

DECISION_INTERVAL: Time = 1_250
"""The time between two decisions of a robot: 10 m at 8 m/s, so a drone flies at its speed."""

EXPLORERS = 10
KILLERS = 4

UNLIMITED = 1_000_000
"""Battery charges and treatment doses without end, for a run without battery drain or resupply."""


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
        MapConfig(size=32, obstacles=river_and_rocks(32)),
        hours=2,
        pests_per_hour=15.0,
    ),
    "rivers-large": StandardMap(
        "rivers-large",
        "64 x 64 cells, two rivers and rocks",
        MapConfig(size=64, obstacles=two_rivers_and_rocks(64)),
        hours=4,
        pests_per_hour=60.0,
    ),
    "open-small": StandardMap(
        "open-small",
        "32 x 32 cells, open field",
        MapConfig(size=32, obstacles=()),
        hours=2,
        pests_per_hour=15.0,
    ),
    "open-large": StandardMap(
        "open-large",
        "64 x 64 cells, open field",
        MapConfig(size=64, obstacles=()),
        hours=4,
        pests_per_hour=60.0,
    ),
}
"""The standard maps by name; the large fields have four times the cells, and the pests."""

DEFAULT_ROBOTS = RobotConfig(
    speed_mps_drone=8.0,
    speed_mps_ground=1.0,
    radio_range_m=100.0,
    decision_interval=DECISION_INTERVAL,
    knows_terrain=False,
    lattice_starts=False,
    battery_capacity=UNLIMITED,
)


def default_pests(standard: StandardMap) -> PestConfig:
    """Static pests: ``pests_per_hour`` at the start and every hour, perfectly treated."""
    return PestConfig(
        pests_per_hour=standard.pests_per_hour,
        growth_rate=0.0,
        spread_probability=0.0,
        treatment_rate=FAST_TREATMENT,
        treatment_doses=UNLIMITED,
        source_range_m=None,
        initial_pests=round(standard.pests_per_hour),
    )


def default_scenario(map_name: str) -> SwarmScenario:
    """The default swarm of explorers and seeded killers on the standard map ``map_name``.

    Raises:
        KeyError: If there is no standard map of that name.
    """
    standard = MAPS[map_name]
    scenario = SwarmScenario(
        map=standard.map,
        pests=default_pests(standard),
        robots=DEFAULT_ROBOTS,
        metrics=MetricsConfig(period=TimeUnit.MINUTE),
        until=standard.hours * TimeUnit.HOUR,
    )
    return scenario.add_robots(EXPLORER, EXPLORERS).add_robots(SEEDED_KILLER, KILLERS)


def pest_run_spec(scenario: SwarmScenario, map_name: str, warmup_hours: int = 1) -> PestRunSpec:
    """``scenario`` as a run of ``measure_pest_response`` for the standard map's hours and pests.

    ``infested`` leaves out the first ``warmup_hours``, while the swarm finds the pests that were
    there at the start.
    """
    standard = MAPS[map_name]
    return PestRunSpec(scenario, standard.hours, standard.pests_per_hour, warmup_hours)


def with_radio(scenario: SwarmScenario, radio_range_m: float) -> SwarmScenario:
    """``scenario`` with every robot's radio reaching ``radio_range_m``."""
    return scenario.with_robot_config(replace(scenario.robots, radio_range_m=radio_range_m))


def killer_methods(scenario: SwarmScenario) -> dict[str, SwarmScenario]:
    """``scenario`` with its killers as the best seeded and the best hop-based killers.

    The hop killers' beacons are tuned to the scenario's radio range (``hop-tuned``), so build
    the methods after setting the radio.
    """
    configurations = pest_response_configurations(scenario)
    return {"seeded": configurations["seeded"], "hop": configurations["hop-tuned"]}


def with_grid_sweepers(
    scenario: SwarmScenario,
    count: int,
    body: BodyType,
    brain: Callable[[int, int | None], BrainSpec],
) -> SwarmScenario:
    """``scenario`` with ``count`` grid sweepers of ``body`` added, each on its own strip.

    ``brain`` is ``grid_sweep_brain`` for killers or ``grid_explorer_brain`` for explorers. Every
    sweeper is a preset of its own, so that its strip is set by its number rather than by where
    it starts: the start row may be interrupted by a river.
    """
    for strip in range(count):
        preset = CustomPreset(f"grid-{strip}", body, brain(count, strip))
        scenario = scenario.add_robots(preset, 1)
    return scenario
