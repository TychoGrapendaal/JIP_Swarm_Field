"""Escaping by trial and error: a robot that stops making progress pushes itself away.

A robot following a heading can settle where the pull towards its objective and the walls in
the way cancel out, standing still or pacing up and down a wall. ``StuckEscape`` notices that
the robot's own hop number has not come down for a while and places an obstacle beacon on its
cell that pushes it off. If the robot is stuck again before it made any progress, the next
beacon pushes harder, so the robot eventually breaks out of the equilibrium.
"""

from collections import deque
from dataclasses import dataclass

from swarm_field.contracts import CellId, Time, TimeUnit


@dataclass(frozen=True)
class EscapeConfig:
    """When a robot counts as stuck, and how hard its escape beacons push.

    Attributes:
        stuck_window: How long the robot is watched before it can count as stuck.
        progress_hops: A robot whose own hop number came down by less than this over
            ``stuck_window`` is stuck.
        escape_strength: The push of the first escape beacon.
        escape_boost: How much harder each further escape beacon pushes, while the robot
            stays stuck.
        escape_at_wall: Whether the beacon goes on the blocked cell ahead of the robot, when
            there is one, instead of on the robot's own cell.
    """

    stuck_window: Time = 5 * TimeUnit.MINUTE
    progress_hops: float = 0.2
    escape_strength: float = 1.0
    escape_boost: float = 2.0
    escape_at_wall: bool = False


@dataclass(frozen=True)
class EscapeBeacon:
    """Where to place an escape beacon, and how hard it pushes.

    Attributes:
        cell: The cell the beacon marks.
        strength: Its push.
    """

    cell: CellId
    strength: float


class StuckEscape:
    """Watches one robot's progress and hands out ever stronger escape beacons while it is stuck."""

    def __init__(self, config: EscapeConfig) -> None:
        self._config: EscapeConfig = config
        self._history: deque[tuple[Time, float]] = deque()
        self._escapes: int = 0

    def check(
        self, robot: CellId, hop: float, now: Time, blocked_ahead: CellId | None
    ) -> EscapeBeacon | None:
        """The escape beacon to place now, or ``None`` while the robot makes progress.

        Args:
            robot: The robot's cell.
            hop: The robot's own hop number.
            now: The time of the decision.
            blocked_ahead: The cell ahead of the robot along its heading, if it cannot enter it.
        """
        self._history.append((now, hop))
        while self._history and now - self._history[0][0] > self._config.stuck_window:
            self._history.popleft()
        started, earlier = self._history[0]
        if now - started < self._config.stuck_window:
            return None
        if earlier - hop >= self._config.progress_hops:
            self._escapes = 0
            return None
        strength = self._config.escape_strength * self._config.escape_boost**self._escapes
        self._escapes += 1
        self._history.clear()
        cell = robot
        if self._config.escape_at_wall and blocked_ahead is not None:
            cell = blocked_ahead
        return EscapeBeacon(cell, strength)
