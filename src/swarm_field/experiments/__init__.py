"""Experiments: scenarios built by hand, run by ``swarm_field.runner``, and every scenario by name.

Each scenario describes its field, robots, brains and metrics as an ``ExperimentConfig``; the
runner assembles and runs it. The catalogue names every scenario, presets and hand-built ones
alike, so sweeps and viewers take any of them by name. ``LiveRun`` advances one run step by step,
for a viewer.
"""

from swarm_field.experiments.catalogue import (
    build_named_experiment,
    scenario_names,
    sweep_scenarios,
)
from swarm_field.experiments.explore import (
    ExploreScenario,
    cropped_cells,
    river_and_rocks,
    two_rivers_and_rocks,
)
from swarm_field.experiments.pests import (
    FAST_TREATMENT,
    RELAYED_KILLERS,
    RIVER_SMALL,
    RIVERS_LARGE,
    PestScenario,
    pest_response_configurations,
)
from swarm_field.experiments.run import LiveRun

__all__ = [
    "FAST_TREATMENT",
    "RELAYED_KILLERS",
    "RIVERS_LARGE",
    "RIVER_SMALL",
    "ExploreScenario",
    "LiveRun",
    "PestScenario",
    "build_named_experiment",
    "cropped_cells",
    "pest_response_configurations",
    "river_and_rocks",
    "scenario_names",
    "sweep_scenarios",
    "two_rivers_and_rocks",
]
