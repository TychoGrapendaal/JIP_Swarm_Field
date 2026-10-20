"""Experiments: scenarios built by hand, run by ``swarm_field.runner``.

Each scenario describes its field, robots, brains and metrics as an ``ExperimentConfig``; the
runner assembles and runs it. ``LiveRun`` advances one such run step by step, for a viewer.
"""

from swarm_field.experiments.explore import ExploreScenario, cropped_cells, river_and_rocks
from swarm_field.experiments.pests import RELAYED_KILLERS, PestScenario
from swarm_field.experiments.run import LiveRun

__all__ = [
    "RELAYED_KILLERS",
    "ExploreScenario",
    "LiveRun",
    "PestScenario",
    "cropped_cells",
    "river_and_rocks",
]
