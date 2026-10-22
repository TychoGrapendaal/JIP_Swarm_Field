"""Escaping by trial and error: a robot stuck against a wall pushes itself off it.

A robot following a heading can settle where the pull towards its objective and a wall in the
way cancel out. ``StuckEscape`` notices that the robot has stayed within one cell for a while
with a wall ahead of it, and places a repelling beacon on that wall. If the robot gets stuck
against the same wall again, the next beacon pushes harder than the one still there, so the
robot eventually breaks out of the equilibrium.
"""

from collections import deque
from dataclasses import dataclass

from swarm_field.contracts import CellId


@dataclass(frozen=True)
class EscapeConfig:
    """When a robot counts as stuck, and how hard its escape beacons push.

    Attributes:
        stuck_decisions: Over how many decisions on a heading the robot must stay within one
            cell to count as stuck. A move is under way for several decisions, during which
            the robot reports the cell it left, so this must be more than two diagonal moves:
            a ground killer needs 10 decisions for one.
        escape_strength: The push of an escape beacon on a wall that pushes no more than that.
        escape_boost: How much harder an escape beacon pushes than the push still on its wall.
    """

    stuck_decisions: int = 30
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
    """Remembers where one robot was at its last decisions, and hands out escape beacons."""

    def __init__(self, config: EscapeConfig) -> None:
        self._config: EscapeConfig = config
        self._recent: deque[CellId] = deque(maxlen=config.stuck_decisions)

    def check(
        self, robot: CellId, wall_ahead: CellId | None, wall_push: float
    ) -> EscapeBeacon | None:
        """The escape beacon to place at this decision, or ``None``.

        A beacon is placed when the robot stayed within one cell over the last
        ``stuck_decisions`` decisions and has a wall ahead of it. It pushes ``escape_boost``
        times the push still on that wall, and at least ``escape_strength``. After a beacon the
        count starts over.

        Args:
            robot: The robot's cell.
            wall_ahead: The cell ahead of the robot along its heading, if the robot senses that
                it cannot enter it.
            wall_push: The push of the obstacle beacon already on ``wall_ahead``, or 0.
        """
        self._recent.append(robot)
        if wall_ahead is None or not self._is_stuck():
            return None
        self._recent.clear()
        strength = max(self._config.escape_strength, wall_push * self._config.escape_boost)
        return EscapeBeacon(wall_ahead, strength)

    def _is_stuck(self) -> bool:
        """Whether the window is full and its cells fit in a block of two by two cells."""
        if len(self._recent) < self._config.stuck_decisions:
            return False
        xs = [cell[0] for cell in self._recent]
        ys = [cell[1] for cell in self._recent]
        return max(xs) - min(xs) <= 1 and max(ys) - min(ys) <= 1
