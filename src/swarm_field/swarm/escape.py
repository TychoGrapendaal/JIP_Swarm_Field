"""Escaping by trial and error: a robot stuck against a wall pushes itself off it.

A robot following a heading can settle where the pull towards its objective and a wall in the
way cancel out. ``StuckEscape`` notices that the robot has stayed in the same cell for a while
with a wall ahead of it, and places a repelling beacon on that wall. If the robot is still in
the same cell the next time, the next beacon pushes harder, so the robot eventually breaks out
of the equilibrium.
"""

from dataclasses import dataclass

from swarm_field.contracts import CellId


@dataclass(frozen=True)
class EscapeConfig:
    """When a robot counts as stuck, and how hard its escape beacons push.

    Attributes:
        stuck_decisions: How many decisions in a row the robot must stay in the same cell to
            count as stuck. A killer needs several decisions to cross one cell, so this must be
            more than that.
        escape_strength: The push of the first escape beacon.
        escape_boost: How much harder each further escape beacon pushes, while the robot stays
            in the same cell.
    """

    stuck_decisions: int = 10
    escape_strength: float = 1.0
    escape_boost: float = 2.0


@dataclass(frozen=True)
class EscapeBeacon:
    """Where to place an escape beacon, and how hard it pushes.

    Attributes:
        cell: The wall cell the beacon marks.
        strength: Its push.
    """

    cell: CellId
    strength: float


class StuckEscape:
    """Counts how long one robot stays in its cell, and hands out ever stronger escape beacons."""

    def __init__(self, config: EscapeConfig) -> None:
        self._config: EscapeConfig = config
        self._cell: CellId | None = None
        self._decisions: int = 0
        self._escapes: int = 0

    def check(self, robot: CellId, wall_ahead: CellId | None) -> EscapeBeacon | None:
        """The escape beacon to place at this decision, or ``None``.

        A beacon is placed when the robot has stayed in ``robot`` for ``stuck_decisions``
        decisions and has a wall ahead of it. Leaving the cell starts the count and the boost
        over.

        Args:
            robot: The robot's cell.
            wall_ahead: The cell ahead of the robot along its heading, if the robot senses that
                it cannot enter it.
        """
        if robot != self._cell:
            self._cell = robot
            self._decisions = 0
            self._escapes = 0
        self._decisions += 1
        if self._decisions < self._config.stuck_decisions or wall_ahead is None:
            return None
        strength = self._config.escape_strength * self._config.escape_boost**self._escapes
        self._escapes += 1
        self._decisions = 0
        return EscapeBeacon(wall_ahead, strength)
