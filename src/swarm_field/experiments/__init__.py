"""Experiments: scenarios built by hand, run by ``swarm_field.runner``, and every scenario by name.

Each scenario describes its field, robots, brains and metrics as an ``ExperimentConfig``; the
runner assembles and runs it. The catalogue names every scenario, presets and hand-built ones
alike, so sweeps and viewers take any of them by name. ``LiveRun`` advances one run step by step,
for a viewer. ``Experiment`` runs a batch of runs in parallel and keeps their results in a folder
of its own in ``output/``, and ``record_videos`` runs some of them again into videos there.
"""

from swarm_field.experiments.batch import (
    OUTPUT,
    Experiment,
    Run,
    RunResult,
    arm_name,
    by_arm,
    markdown_table,
    mean_and_error,
    paired_difference,
    run_in_parallel,
    runs_for,
)
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
from swarm_field.experiments.pest_response import (
    MeasuredPestRun,
    PestResponse,
    PestRunSpec,
    PestTracker,
    measure_pest_response,
    pest_table_cells,
    pest_table_headers,
    treated_share,
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
from swarm_field.experiments.video import record_video, record_videos, video_path

__all__ = [
    "FAST_TREATMENT",
    "OUTPUT",
    "RELAYED_KILLERS",
    "RIVERS_LARGE",
    "RIVER_SMALL",
    "Experiment",
    "ExploreScenario",
    "LiveRun",
    "MeasuredPestRun",
    "PestResponse",
    "PestRunSpec",
    "PestScenario",
    "PestTracker",
    "Run",
    "RunResult",
    "arm_name",
    "build_named_experiment",
    "by_arm",
    "cropped_cells",
    "markdown_table",
    "mean_and_error",
    "measure_pest_response",
    "paired_difference",
    "pest_response_configurations",
    "pest_table_cells",
    "pest_table_headers",
    "record_video",
    "record_videos",
    "river_and_rocks",
    "run_in_parallel",
    "runs_for",
    "scenario_names",
    "sweep_scenarios",
    "treated_share",
    "two_rivers_and_rocks",
    "video_path",
]
