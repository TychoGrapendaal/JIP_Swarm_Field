"""The shared state for one discrete-event run.

Events read and mutate this and schedule follow-ups through it; they never call each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING

from swarm_field.contracts import RobotId, SwarmAgent, Time
from swarm_field.simulation.channel import Channel
from swarm_field.simulation.event import Event, EventQueue
from swarm_field.simulation.world import WorldState

if TYPE_CHECKING:
    from swarm_field.simulation.metrics import MetricsCollector


@dataclass
class SimulationContext:
    """Mutable state shared across a run.

    Attributes:
        now: The current simulation time (advanced to each event as it is processed).
        rng: The single seeded random source; events draw from it in event order.
        agents: The robot brains, keyed by id (the boundary with the algorithm).
        world: Ground-truth field state owned by the simulation.
        queue: Pending events; schedule follow-ups with ``schedule``.
        channel: The radio; hand a robot's outbox to it to reach the robots in range.
        metrics: The run's metrics collector, or ``None`` for a run that records none.
    """

    now: Time
    rng: Random
    agents: dict[RobotId, SwarmAgent]
    world: WorldState
    queue: EventQueue = field(default_factory=EventQueue)
    channel: Channel = field(default_factory=Channel)
    metrics: MetricsCollector | None = None

    def schedule(self, event: Event) -> None:
        """Schedule a future event."""
        self.queue.schedule(event)
