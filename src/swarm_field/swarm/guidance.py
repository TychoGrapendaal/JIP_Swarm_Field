"""The port between an objective and the navigator: what guides the robot towards its goal.

An ``ObjectiveField`` turns the robot's memory into ``Guidance`` for one objective. Two shapes
exist: a ``SeededField``, a cost grid the navigator plans along, and a ``Heading``, a direction
the navigator hands to its reactive steerer. A brain can mix fields of both shapes, one per
objective.
"""

from dataclasses import dataclass
from typing import Protocol

from swarm_field.contracts import CellId, Time, Vector2
from swarm_field.swarm.memory import MemoryView
from swarm_field.swarm.seeder import SeededField


@dataclass(frozen=True)
class Heading:
    """A direction to walk in, rather than a route.

    Attributes:
        offset: Relative displacement in cells from the robot to the point it walks towards.
        target: The cell the robot reports to its peers as its target, if any.
    """

    offset: Vector2
    target: CellId | None


type Guidance = SeededField | Heading
"""What an ``ObjectiveField`` hands the navigator."""


class ObjectiveField(Protocol):
    """Guides the robot towards one objective, named so that a behaviour can bid for it."""

    name: str

    def guidance_for(self, memory: MemoryView, robot: CellId, now: Time) -> Guidance | None:
        """The guidance from the robot's cell at ``now``, or ``None`` with nothing to go to."""
        ...
