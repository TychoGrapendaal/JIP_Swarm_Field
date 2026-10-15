"""Runnable experiments: the simulation (B) and the robot brain (A) assembled for one study.

This is a composition root. It is the only package that imports both ``simulation`` and
``swarm``; they still never import each other.
"""

from swarm_field.experiments.explore import (
    ExploreRun,
    ExploreScenario,
    FollowSeederField,
    RandomWaypointSeeder,
    build_explore_run,
    river_and_rocks,
    terrain_map,
)
from swarm_field.experiments.pests import PestRun, PestScenario, build_pest_run

__all__ = [
    "ExploreRun",
    "ExploreScenario",
    "FollowSeederField",
    "PestRun",
    "PestScenario",
    "RandomWaypointSeeder",
    "build_explore_run",
    "build_pest_run",
    "river_and_rocks",
    "terrain_map",
]
