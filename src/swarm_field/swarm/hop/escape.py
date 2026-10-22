"""Escaping by trial and error: a robot that circles in place pushes itself out.

A robot following a heading can be caught where pulls cancel out: between pests that pull it
equally hard, or where the pull towards its objective runs into a wall. It then shuttles between
two cells, paces along a wall or stands still, and may lose its heading for a few decisions on
the way. ``StuckEscape`` notices that the robot got much less far over its last minutes than its
speed allows, and places a repelling beacon on the place it circles around. If the robot gets
stuck in the same area again, that beacon pushes harder and lasts longer, so it does not fade
while the robot circles, and the robot eventually breaks out.
"""

from collections import deque
from dataclasses import dataclass
from math import copysign, dist, floor, hypot, radians, sin

from swarm_field.contracts import CellId, Time, TimeUnit, Vector2

_SIN_22_5 = sin(radians(22.5))
"""Half the angle between two of the eight directions, as the sine that separates them."""


@dataclass(frozen=True)
class EscapeConfig:
    """When a robot counts as stuck, and how hard its escape beacons push.

    Attributes:
        window: Over how long the robot's progress is measured. A move is under way for a
            while, during which the robot reports the cell it left, so this must span several
            moves: a ground killer at 0.1 m/s takes 100 s for one 10 m cell.
        min_progress: Below this progress over the window the robot counts as stuck. Progress
            is how far the robot got divided by how far its top speed would have taken it: near
            1 for a straight walk, near 0 for a shuttle, and 0 for standing still.
        escape_strength: The push of a first escape beacon in an area.
        escape_decay: The decay time of a first escape beacon's push.
        escape_boost: How much harder an escape beacon pushes, and how much longer it lasts,
            each time the robot is stuck again in the same area while the beacon is younger than
            its decay time.
    """

    window: Time = 10 * TimeUnit.MINUTE
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
    """Remembers where one robot was over its last ``window``, and escapes circling.

    It also remembers the last escape beacon it handed out, to boost it while it keeps failing.
    """

    def __init__(self, config: EscapeConfig) -> None:
        self._config: EscapeConfig = config
        self._recent: deque[tuple[Time, CellId]] = deque()
        self._last_heading_at: Time | None = None
        self._last: EscapeBeacon | None = None
        self._last_at: Time = 0

    def check(
        self, robot: CellId, ahead: CellId | None, now: Time, speed: float
    ) -> EscapeBeacon | None:
        """Record the robot's cell at this decision; the escape beacon to place, or ``None``.

        Every decision is recorded, also one that followed no heading, since a robot that loses
        its heading while it circles is still circling. Only a decision on a heading is judged:
        a beacon is placed when the recorded cells span at least the ``window`` and the robot's
        progress over them is below ``min_progress``. A heading after more than a ``window``
        without one starts the record over, so a robot that waited for work is not stuck.

        The beacon goes on the centre of the cells the robot was in, or on ``ahead`` if that
        centre is the robot's own cell, where it would push in no direction. If that place is
        within one cell of the last escape beacon, and that beacon is younger than its decay
        time, the last beacon is placed again on its own cell with ``escape_boost`` times its
        strength and its decay time. After a beacon the record starts over.

        Args:
            robot: The robot's cell.
            ahead: The neighbouring cell the robot's heading points into, or ``None`` for a
                decision that followed no heading.
            now: The time of the decision.
            speed: The robot's top speed, in cells per second.
        """
        window = self._config.window
        on_heading = ahead is not None
        if on_heading and (self._last_heading_at is None or now - self._last_heading_at > window):
            self._recent.clear()
        self._recent.append((now, robot))
        while len(self._recent) > 1 and self._recent[1][0] <= now - window:
            self._recent.popleft()
        if ahead is None:
            return None
        self._last_heading_at = now
        if now - self._recent[0][0] < window:
            return None
        if self._progress(robot, now, speed) >= self._config.min_progress:
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
        """Start the record over, e.g. after a treatment: the robot stood still on purpose."""
        self._recent.clear()
        self._last_heading_at = None

    def _progress(self, robot: CellId, now: Time, speed: float) -> float:
        """How far the robot got since the oldest record, over how far its speed would take it.

        A robot that could not have moved at all made no progress.
        """
        start, first = self._recent[0]
        reach = speed * (now - start) / TimeUnit.SECOND
        if reach <= 0:
            return 0.0
        return dist(first, robot) / reach

    def _centre(self) -> CellId:
        """The cell nearest to the mean of the recorded cells."""
        count = len(self._recent)
        mean_x = sum(cell[0] for _, cell in self._recent) / count
        mean_y = sum(cell[1] for _, cell in self._recent) / count
        return (floor(mean_x + 0.5), floor(mean_y + 0.5))


def _within_one_cell(a: CellId, b: CellId) -> bool:
    """Whether two cells are the same or neighbours, diagonals included."""
    return abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1


def cell_ahead(robot: CellId, offset: Vector2) -> CellId:
    """The neighbouring cell the robot would step into along ``offset``, of the eight around it.

    A component counts as a step when it is more than ``sin(22.5°)`` of the offset's length,
    so the eight directions each cover an equal angle.
    """
    length = hypot(offset[0], offset[1])
    step_x = 0
    step_y = 0
    if length > 0:
        if abs(offset[0]) / length > _SIN_22_5:
            step_x = int(copysign(1, offset[0]))
        if abs(offset[1]) / length > _SIN_22_5:
            step_y = int(copysign(1, offset[1]))
    return (robot[0] + step_x, robot[1] + step_y)
