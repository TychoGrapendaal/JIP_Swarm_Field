"""Component B — the environment simulation (discrete-event).

An ``EventQueue`` holds timestamped events and ``SimulationLoop`` processes them in time
order over a shared ``SimulationContext``. That context carries the ``World`` (ground truth)
and the ``Channel`` that delivers broadcasts between robots in range. Kept minimal for now:
two example events, scheduled programmatically.
"""

from swarm_field.simulation.channel import Channel, robots_in_range_of
from swarm_field.simulation.context import SimulationContext
from swarm_field.simulation.event import Event, EventQueue, RobotDecision, WorldTick
from swarm_field.simulation.loop import SimulationLoop
from swarm_field.simulation.world import WorldState

__all__ = [
    "Channel",
    "Event",
    "EventQueue",
    "RobotDecision",
    "SimulationContext",
    "SimulationLoop",
    "World",
    "WorldTick",
    "robots_in_range_of",
]
