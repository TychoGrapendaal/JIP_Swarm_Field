"""The shared state for one discrete-event run.

Events read and mutate this and schedule follow-ups through it; they never call each other.
"""

from dataclasses import dataclass, field
from random import Random

from swarm_field.contracts import RobotId, SwarmAgent, Time
from swarm_field.simulation.event import Event, EventQueue
from swarm_field.simulation.world import WorldState


@dataclass
class SimulationContext:
    """Mutable state shared across a run.

    Attributes:
        now: The current simulation time (advanced to each event as it is processed).
        rng: The single seeded random source; events draw from it in event order.
        agents: The robot brains, keyed by id (the boundary with the algorithm).
        world: Ground-truth field state owned by the simulation.
        queue: Pending events; schedule follow-ups with ``schedule``.
    """

    now: Time
    rng: Random
    agents: dict[RobotId, SwarmAgent]
    world: WorldState
    queue: EventQueue = field(default_factory=EventQueue)

    def schedule(self, event: Event) -> None:
        """Schedule a future event."""
        self.queue.schedule(event)
