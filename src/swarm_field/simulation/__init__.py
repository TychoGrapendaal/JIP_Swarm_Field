"""Component B — the environment simulation (discrete-event).

An ``EventQueue`` holds timestamped events and ``SimulationLoop`` processes them in time
order over a shared ``SimulationContext``. Kept minimal for now: two example events,
scheduled programmatically.
"""

from swarm_field.simulation.context import SimulationContext
from swarm_field.simulation.event import Event, EventQueue, RobotDecision, WorldTick
from swarm_field.simulation.loop import SimulationLoop

__all__ = [
    "Event",
    "EventQueue",
    "RobotDecision",
    "SimulationContext",
    "SimulationLoop",
    "WorldTick",
]
