"""Core identifiers, geometry aliases and the discrete clock.

These are the smallest shared vocabulary items; every other contract builds on them.
"""

from enum import Enum, auto
from typing import Final

RobotId = int
CellId = tuple[int, int]
"""Grid coordinate for v1; treated as an opaque handle elsewhere."""

Time = int
"""Simulation time as an integer number of milliseconds.

The simulation is discrete-event: events carry a ``Time`` timestamp and are processed in
time order. There is no fixed timestep. Write durations with ``TimeUnit``.
"""


class TimeUnit:
    """Named durations in ``Time`` units, e.g. ``24 * TimeUnit.HOUR``."""

    MILLISECOND: Final[Time] = 1
    SECOND: Final[Time] = 1000 * MILLISECOND
    MINUTE: Final[Time] = 60 * SECOND
    HOUR: Final[Time] = 60 * MINUTE
    DAY: Final[Time] = 24 * HOUR


class Direction(Enum):
    """The eight compass headings on the grid.

    Members are listed clockwise, so :meth:`cw` / :meth:`ccw` step by one 45-degree
    increment. Offset convention (see :pyattr:`offset`): x increases East, y increases North.
    """

    N = auto()
    NE = auto()
    E = auto()
    SE = auto()
    S = auto()
    SW = auto()
    W = auto()
    NW = auto()

    @property
    def offset(self) -> tuple[int, int]:
        """The ``(dx, dy)`` grid step of a single move in this direction."""
        return _DIRECTION_OFFSETS[self]

    def cw(self) -> "Direction":
        """The heading one 45-degree step clockwise."""
        order = list(Direction)
        return order[(order.index(self) + 1) % len(order)]

    def ccw(self) -> "Direction":
        """The heading one 45-degree step counter-clockwise."""
        order = list(Direction)
        return order[(order.index(self) - 1) % len(order)]


_DIRECTION_OFFSETS: dict[Direction, tuple[int, int]] = {
    Direction.N: (0, 1),
    Direction.NE: (1, 1),
    Direction.E: (1, 0),
    Direction.SE: (1, -1),
    Direction.S: (0, -1),
    Direction.SW: (-1, -1),
    Direction.W: (-1, 0),
    Direction.NW: (-1, 1),
}


class BodyType(Enum):
    """High-level robot body classes used by scenarios and behaviours."""

    SCOUT = auto()
    KILLER = auto()
    BATTERY = auto()


class AgentObjective(Enum):
    """High-level objective a robot is currently pursuing.

    This is deliberately separate from a behaviour's implementation name: several behaviours
    may serve the same objective, and peers should compare stable values rather than free-form
    strings.
    """

    TREAT = auto()
    EXPLORE = auto()
    VERIFY = auto()
    RECHARGE = auto()


class Terrain(Enum):
    """What a cell is made of, as far as moving through it is concerned.

    ``OUTSIDE`` marks cells beyond the field boundary. There is no ``UNKNOWN`` member: a cell
    whose terrain nobody has observed yet is represented by the absence of a terrain
    observation.
    """

    FREE = auto()
    OBSTACLE = auto()
    OUTSIDE = auto()


class CropType(Enum):
    """Crop species used by the built-in scenario presets."""

    MAIZE = auto()
    WHEAT = auto()
    POTATO = auto()
    SUGAR_BEET = auto()


class PestType(Enum):
    """Pest / target type. Members are scenario-defined; may be unknown at low confidence."""
