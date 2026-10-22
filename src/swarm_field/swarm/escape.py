"""Escaping by trial and error: a robot that circles in place pushes itself out.

A robot following a heading can be caught where pulls cancel out: between pests that pull it
equally hard, or where the pull towards its objective runs into a wall. It then shuttles between
two cells, paces along a wall or stands still. ``StuckEscape`` notices that the robot's walk
over its last decisions made little progress, and places a repelling beacon on the place it
circles around. If the robot gets stuck in the same area again, that beacon pushes harder and
lasts longer, so it does not fade while the robot circles, and the robot eventually breaks out.
"""

from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from math import dist, floor

from swarm_field.contracts import CellId, Time, TimeUnit


@dataclass(frozen=True)
class EscapeConfig:
    """When a robot counts as stuck, and how hard its escape beacons push.

    Attributes:
        stuck_decisions: Over how many decisions on a heading the robot's progress is measured.
            A move is under way for several decisions, during which the robot reports the cell
            it left, so this must span several moves: a ground killer needs 10 decisions for one
            diagonal move.
        min_progress: Below this progress over the window the robot counts as stuck. Progress
            is how far the robot got divided by how far it walked: 1 for a straight walk, near
            0 for a shuttle, and 0 for standing still.
        escape_strength: The push of a first escape beacon in an area.
        escape_decay: The decay time of a first escape beacon's push.
        escape_boost: How much harder an escape beacon pushes, and how much longer it lasts,
            each time the robot is stuck again in the same area while the beacon is younger than
            its decay time.
    """

    stuck_decisions: int = 40
    min_progress: float = 0.3
    escape_strength: float = 1.0
    escape_decay: Time = 10 * TimeUnit.MINUTE
    escape_boost: float = 2.0


@dataclass(frozen=True)
class EscapeBeacon:
    """Where to place an escape beacon, how hard it pushes and how long it lasts.

    Attributes:
        cell: The cell the beacon marks.
        strength: Its push.
        decay: The decay time of its push.
    """

    cell: CellId
    strength: float
    decay: Time


class StuckEscape:
    """Remembers where one robot was at its last decisions on a heading, and escapes circling.

    It also remembers the last escape beacon it handed out, to boost it while it keeps failing.
    """

    def __init__(self, config: EscapeConfig) -> None:
        self._config: EscapeConfig = config
        self._recent: deque[CellId] = deque(maxlen=config.stuck_decisions)
        self._last: EscapeBeacon | None = None
        self._last_at: Time = 0

    def check(self, robot: CellId, ahead: CellId, now: Time) -> EscapeBeacon | None:
        """The escape beacon to place at this decision on a heading, or ``None``.

        A beacon is placed when the robot's progress over the last ``stuck_decisions`` decisions
        is below ``min_progress``. It goes on the centre of the cells the robot was in, or on
        ``ahead`` if that centre is the robot's own cell, where it would push in no direction.
        If that place is within one cell of the last escape beacon, and that beacon is younger
        than its decay time, the last beacon is placed again on its own cell with
        ``escape_boost`` times its strength and its decay time. After a beacon the window starts
        over.

        Args:
            robot: The robot's cell.
            ahead: The neighbouring cell the robot's heading points into.
            now: The time of the decision.
        """
        self._recent.append(robot)
        if len(self._recent) < self._config.stuck_decisions:
            return None
        if self._progress() >= self._config.min_progress:
            return None
        cell = self._centre()
        if cell == robot:
            cell = ahead
        self._recent.clear()
        escape = EscapeBeacon(cell, self._config.escape_strength, self._config.escape_decay)
        last = self._last
        if (
            last is not None
            and _within_one_cell(cell, last.cell)
            and now - self._last_at < last.decay
        ):
            boost = self._config.escape_boost
            escape = EscapeBeacon(last.cell, last.strength * boost, int(last.decay * boost))
        self._last = escape
        self._last_at = now
        return escape

    def forget(self) -> None:
        """Start the window over, after a decision that followed no heading."""
        self._recent.clear()

    def _progress(self) -> float:
        """How far the robot got over the window, divided by how far it walked; 0 standing."""
        walked = 0.0
        for before, after in pairwise(self._recent):
            walked += dist(before, after)
        if walked == 0:
            return 0.0
        return dist(self._recent[0], self._recent[-1]) / walked

    def _centre(self) -> CellId:
        """The cell nearest to the mean of the window's cells."""
        count = len(self._recent)
        mean_x = sum(cell[0] for cell in self._recent) / count
        mean_y = sum(cell[1] for cell in self._recent) / count
        return (floor(mean_x + 0.5), floor(mean_y + 0.5))


def _within_one_cell(a: CellId, b: CellId) -> bool:
    """Whether two cells are the same or neighbours, diagonals included."""
    return abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1
