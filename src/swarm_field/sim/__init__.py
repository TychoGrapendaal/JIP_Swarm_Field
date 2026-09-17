"""Component B — the environment simulation (discrete-event).

An ``EventQueue`` holds timestamped events and ``SimLoop`` processes them in time order over
a shared ``SimContext``. Kept minimal for now: two example events, scheduled programmatically.
"""

from swarm_field.sim.context import SimContext
from swarm_field.sim.event import Event, EventQueue, RobotDecision, WorldTick
from swarm_field.sim.loop import SimLoop

__all__ = [
    "Event",
    "EventQueue",
    "RobotDecision",
    "SimContext",
    "SimLoop",
    "WorldTick",
]
