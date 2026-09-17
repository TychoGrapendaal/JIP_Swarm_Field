"""Actions — a tagged union of the physical things a robot can do in one tick.

Exactly one physical action per tick (each consumes the tick). Communication is not an
action; it rides on ``Decision.outbox``.

Most variants are intentionally empty placeholders: how movement and treatment are
parameterised depends on how motion is modelled in the discrete simulation (e.g. the
robot may command a direction or speed rather than an absolute cell — a ``CellId`` is an
environment concept the robot cannot be certain of). Those shapes are decided later.
"""

from dataclasses import dataclass

from swarm_field.contracts.ids import Direction


@dataclass(frozen=True)
class Move:
    """Move for this tick. Parameters deferred until movement modelling is decided."""


@dataclass(frozen=True)
class Turn:
    """Reorient in place; cheap, but changes what is sensed next tick.

    Attributes:
        facing: The new heading to adopt.
    """

    facing: Direction


@dataclass(frozen=True)
class Treat:
    """Treat the target at the robot's current location. Parameters deferred."""


@dataclass(frozen=True)
class Charge:
    """Recharge in place; only meaningful at a charging point / for a battery body."""


@dataclass(frozen=True)
class Idle:
    """Do nothing this tick."""


Action = Move | Turn | Treat | Charge | Idle
