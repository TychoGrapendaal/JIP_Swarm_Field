"""Actions — a tagged union of the physical things a robot can do in one tick.

Exactly one physical action per tick (each consumes the tick). Communication is not an
action; it rides on ``Decision.outbox``.

Movement and treatment are parameterised as the simulation models them: ``Move`` carries a
compass heading chosen by the navigator, ``Turn`` a new heading. ``Treat`` stays an empty
placeholder — how treatment is parameterised is decided later.
"""

from dataclasses import dataclass

from swarm_field.contracts.ids import Direction


@dataclass(frozen=True)
class Move:
    """Move one cell along a compass heading this tick.

    Attributes:
        direction: The compass heading of the move — an absolute grid direction, using the
            same convention as ``Turn.facing``.
    """

    direction: Direction


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
