"""Experiments: swarm scenarios, run by ``swarm_field.runner``, and every scenario by name.

A ``SwarmScenario`` is a field (``MapConfig``), pests (``PestConfig``), settings every robot
shares (``RobotConfig``), and a swarm built with ``add_robots`` from robot presets: ``EXPLORER``,
``RELAY``, ``SEEDED_KILLER``, ``HOP_KILLER``, variants of their classes, or a ``CustomPreset``.
It describes the run as an ``ExperimentConfig``; the runner assembles and runs it. The catalogue
names every scenario, presets and hand-built ones alike, so sweeps and viewers take any of them
by name. ``LiveRun`` advances one run step by step, for a viewer. ``Experiment`` runs a batch of
runs in parallel and keeps their results in a folder of its own in ``output/``, and
``record_videos`` runs some of them again into videos there.
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
from swarm_field.experiments.maps import (
    MapConfig,
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
from swarm_field.experiments.presets import (
    EXPLORER,
    HOP_KILLER,
    RELAY,
    SEEDED_KILLER,
    CustomPreset,
    ExplorerPreset,
    HopKillerPreset,
    Relaying,
    RelayPreset,
    RobotPreset,
    SeededKillerPreset,
    hop_beacon_variants,
)
from swarm_field.experiments.run import LiveRun
from swarm_field.experiments.scenario import (
    EXPLORE,
    FAST_TREATMENT,
    PESTS,
    RELAYED_KILLERS,
    RIVER_SMALL,
    RIVERS_LARGE,
    PestConfig,
    RobotConfig,
    RobotGroup,
    SwarmScenario,
    pest_response_configurations,
)
from swarm_field.experiments.video import record_video, record_videos, video_path

__all__ = [
    "EXPLORE",
    "EXPLORER",
    "FAST_TREATMENT",
    "HOP_KILLER",
    "OUTPUT",
    "PESTS",
    "RELAY",
    "RELAYED_KILLERS",
    "RIVERS_LARGE",
    "RIVER_SMALL",
    "SEEDED_KILLER",
    "CustomPreset",
    "Experiment",
    "ExplorerPreset",
    "HopKillerPreset",
    "LiveRun",
    "MapConfig",
    "MeasuredPestRun",
    "PestConfig",
    "PestResponse",
    "PestRunSpec",
    "PestTracker",
    "RelayPreset",
    "Relaying",
    "RobotConfig",
    "RobotGroup",
    "RobotPreset",
    "Run",
    "RunResult",
    "SeededKillerPreset",
    "SwarmScenario",
    "arm_name",
    "build_named_experiment",
    "by_arm",
    "cropped_cells",
    "hop_beacon_variants",
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
