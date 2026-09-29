"""Component B — the environment simulation (discrete-event).

An ``EventQueue`` holds timestamped events and ``SimulationLoop`` processes them in time
order over a shared ``SimulationContext``.
"""

from swarm_field.simulation.channel import Channel, robots_in_range_of
from swarm_field.simulation.context import SimulationContext
from swarm_field.simulation.event import (
    Event,
    EventQueue,
    PestDynamicsConfig,
    RobotDecision,
    WorldTick,
)
from swarm_field.simulation.loop import SimulationLoop
from swarm_field.simulation.scenario import build_scenario, build_world
from swarm_field.simulation.world import RobotState, WorldState

__all__ = [
    "Channel",
    "Event",
    "EventQueue",
    "PestDynamicsConfig",
    "RobotDecision",
    "RobotState",
    "SimulationContext",
    "SimulationLoop",
    "WorldState",
    "WorldTick",
    "robots_in_range_of",
    "build_scenario",
    "build_world",
]
