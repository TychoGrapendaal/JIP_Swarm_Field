"""Core identifiers, geometry aliases and the discrete clock.

These are the smallest shared vocabulary items; every other contract builds on them.
"""

from enum import Enum, auto

RobotId = int
CellId = tuple[int, int]
"""Grid coordinate for v1; treated as an opaque handle elsewhere."""

Tick = int
"""Discrete simulation clock. Timestamps are tick counts, never seconds.

Wall-clock seconds, where needed, derive as ``tick * dt`` (``dt`` lives in ``ScenarioConfig``).
"""


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


class CropType(Enum):
    """Crop species present in a scenario. Members are scenario-defined (S3 has several)."""


class PestType(Enum):
    """Pest / target type. Members are scenario-defined; may be unknown at low confidence."""
