"""Component B — the environment simulation (discrete-event).

An ``EventQueue`` holds timestamped events and ``SimulationLoop`` processes them in time
order over a shared ``SimulationContext``.
"""

from swarm_field.simulation.channel import Channel, robots_in_range_of
from swarm_field.simulation.context import SimulationContext
from swarm_field.simulation.event import (
    Event,
    EventQueue,
    FailureCheck,
    FailureConfig,
    MetricSample,
    PestDynamicsConfig,
    RobotDecision,
    WorldTick,
)
from swarm_field.simulation.loop import SimulationLoop
from swarm_field.simulation.metrics import (
    MetricsCollector,
    MetricsConfig,
    write_metrics_csv,
    write_metrics_json,
)
from swarm_field.simulation.replay import CommandReplay
from swarm_field.simulation.scenario import build_scenario, build_world
from swarm_field.simulation.world import EnergyConfig, RobotState, WorldState

__all__ = [
    "Channel",
    "CommandReplay",
    "EnergyConfig",
    "Event",
    "EventQueue",
    "FailureCheck",
    "FailureConfig",
    "MetricSample",
    "MetricsCollector",
    "MetricsConfig",
    "PestDynamicsConfig",
    "RobotDecision",
    "RobotState",
    "SimulationContext",
    "SimulationLoop",
    "WorldState",
    "WorldTick",
    "build_scenario",
    "build_world",
    "robots_in_range_of",
    "write_metrics_csv",
    "write_metrics_json",
]
