"""A.3 Navigation — turn a relative goal into concrete ``Move`` / ``Turn`` actions.

The navigator is the movement policy of component A (the brain). It receives a goal as a
relative displacement vector (where the target is *from here*, in cells) and returns one
physical action per tick: heading changes are ``Turn``, steps along the current heading are
``Move``. It never sees an absolute position — spatial knowledge arrives only through a
``Sensing`` port that reports whether each adjacent grid cell is clear to enter. That keeps
the policy honest about what a robot can actually know (see ``docs/navigation-module.md``).

Obstacle avoidance is a committed right-hand wall-follow (Bug-algorithm style). When the
goal heading is blocked the navigator picks the most goal-aligned clear heading (ties to
the fixed right hand) and commits to it for the episode: it hugs the wall, rounding corners
toward the goal as soon as the tangent ahead gives out, and never re-consults the raw goal
bearing while sliding (that is what turns a row's south aisle into a dead-end trap). It
only leaves the wall when the goal is right next to it. A dead end with no route at all
flips ``is_stalled`` — re-routing stays the brain's job (A.1), not the navigator's.
"""

from dataclasses import dataclass
from math import hypot
from typing import Protocol

from swarm_field.contracts import Direction, Idle, Move, Turn

Vector2 = tuple[float, float]
"""The relative displacement from the robot to its goal, in grid cells.

Units are cells (matching ``Direction.offset``): ``(dx, dy)`` with y increasing North.
"""


class Sensing(Protocol):
    """Local proximity ring — is the adjacent cell in a given heading clear to enter?

    The robot can ask, per compass heading, whether the grid cell one step in that direction
    is free to move into. A real implementation (backed by memory / perception) is future
    work; tests supply fakes driven by a hidden grid, so the navigator itself stays free of
    any absolute geometry.
    """

    def clear(self, direction: Direction) -> bool: ...


@dataclass(frozen=True)
class SteeringConfig:
    """Tuning knobs for the steering policy.

    Attributes:
        target_tolerance: Cell distance below which the goal counts as reached.
        near_goal: Cell distance at which the goal counts as right next to the robot, so a
            wall-following episode may legitimately break off toward it.
        stuck_ticks: Detour ticks without the goal distance beating its running minimum
            before the navigator declares itself stuck.
        progress_epsilon: Smallest goal-distance decrease, in cells, that counts as
            sensible progress toward the goal.
    """

    target_tolerance: float = 0.75
    near_goal: float = 1.5
    stuck_ticks: int = 40
    progress_epsilon: float = 0.01


class Navigator:
    """One-tick steering policy: a relative goal in, a physical action out.

    Public API: :meth:`steer` returns this tick's action; :pyattr:`is_stalled` reports that
    the goal cannot be reached from here; :pyattr:`is_detouring` reports that the robot is
    currently wall-following. State is limited to the policy's short memory (the committed
    tangent, stuck accounting); the robot's own position and the field layout live outside
    the navigator.
    """

    def __init__(self, sensing: Sensing, config: SteeringConfig | None = None) -> None:
        self._sensing: Sensing = sensing
        self._config: SteeringConfig = config if config is not None else SteeringConfig()
        self._tangent: Direction | None = None
        self._detour_ticks: int = 0
        self._min_goal_dist: float | None = None
        self._stuck_ticks: int = 0
        self._stalled: bool = False

    @property
    def is_stalled(self) -> bool:
        """True once the goal is unreachable from here (no clear move, or no progress).

        A stalled navigator idles and keeps reporting until :meth:`reset` is called — the
        brain reacts by re-issuing or replanning the goal.
        """
        return self._stalled

    @property
    def is_detouring(self) -> bool:
        """True while the robot is wall-following around an obstruction."""
        return self._detour_ticks > 0

    def reset(self) -> None:
        """Forget all steering state — after a stall, or when handed a new goal."""
        self._end_detour()
        self._min_goal_dist = None
        self._stuck_ticks = 0
        self._stalled = False

    def steer(self, target: Vector2, facing: Direction) -> Move | Turn | Idle:
        """Pick this tick's action for a goal at relative offset ``target``.

        Args:
            target: Relative displacement from the robot to the goal, in cells.
            facing: The robot's current heading, read from its self view.
        """
        dx, dy = target
        goal_dist = hypot(dx, dy)

        if goal_dist <= self._config.target_tolerance:
            self.reset()
            return Idle()

        if self._stalled:
            return Idle()

        desired = self._best_aligned(target, facing)

        if self._detour_ticks == 0:
            if self._sensing.clear(desired):
                return self._head(desired, facing)
            tangent = self._choose_tangent(desired)
            if tangent is None:
                self._stalled = True
                return Idle()
            self._tangent = tangent
            self._detour_ticks = 1
            return self._track_progress(goal_dist, tangent, facing)

        tangent = self._tangent
        assert tangent is not None
        if not self._sensing.clear(tangent):
            perp = _perp_toward_goal(tangent, target)
            if perp is not None and self._sensing.clear(perp):
                tangent = perp
                self._tangent = tangent
            else:
                tangent = self._shift_tangent(tangent)
                if tangent is None:
                    self._stalled = True
                    return Idle()
                self._tangent = tangent

        if self._sensing.clear(desired) and (
            goal_dist <= self._config.near_goal or self._level_along(tangent, target)
        ):
            self._end_detour()
            return self._head(desired, facing)

        self._detour_ticks += 1
        return self._track_progress(goal_dist, tangent, facing)

    def _track_progress(
        self, goal_dist: float, tangent: Direction, facing: Direction
    ) -> Move | Turn | Idle:
        if (
            self._min_goal_dist is None
            or goal_dist < self._min_goal_dist - self._config.progress_epsilon
        ):
            self._min_goal_dist = goal_dist
            self._stuck_ticks = 0
        else:
            self._stuck_ticks += 1
            if self._stuck_ticks >= self._config.stuck_ticks:
                self._stalled = True
                return Idle()
        return self._head(tangent, facing)

    def _head(self, direction: Direction, facing: Direction) -> Move | Turn:
        if facing == direction:
            return Move(direction=direction)
        return Turn(facing=direction)

    def _end_detour(self) -> None:
        """Leave the wall: the goal is reachable, or we are handed a new goal."""
        self._tangent = None
        self._detour_ticks = 0

    def _level_along(self, tangent: Direction, target: Vector2) -> bool:
        """True when the goal is level with the robot along the slide axis.

        Sliding north or south the goal's north-south offset decides; sliding east or west
        the east-west offset does. A diagonal slide has no single axis — never level.
        """
        ox, oy = tangent.offset
        dx, dy = target
        tolerance = self._config.target_tolerance
        if oy != 0:
            return abs(dy) <= tolerance
        if ox != 0:
            return abs(dx) <= tolerance
        return False

    def _best_aligned(self, target: Vector2, facing: Direction) -> Direction:
        """The compass heading closest to the goal direction, ties broken by heading change.

        Candidates are ranked by alignment score (max), then by the number of 45-degree
        turns from ``facing`` (min, to keep the robot from spinning), then by enum order as
        the final, deterministic tie-break.
        """
        dx, dy = target
        goal_dist = hypot(dx, dy)
        order = list(Direction)
        rows: list[tuple[float, int, int, Direction]] = []
        for d in order:
            ux, uy = _unit_offset(d)
            score = (dx * ux + dy * uy) / goal_dist
            span = abs(order.index(d) - order.index(facing))
            turn_cost = min(span, len(order) - span)
            rows.append((-score, turn_cost, order.index(d), d))
        return min(rows)[3]

    def _choose_tangent(self, desired: Direction) -> Direction | None:
        """The clear heading to start a detour with, when ``desired`` is blocked.

        Among the clear neighbours, pick the one most aligned with ``desired``; ties go to
        whichever is closest clockwise from ``desired`` (the fixed right hand). Chosen once
        per episode — the navigator commits to it; only the wall itself re-routes it.
        """
        order = list(Direction)
        dux, duy = _unit_offset(desired)
        rows: list[tuple[float, int, Direction]] = []
        for d in order:
            if not self._sensing.clear(d):
                continue
            ux, uy = _unit_offset(d)
            rows.append(
                (-(dux * ux + duy * uy), (order.index(d) - order.index(desired)) % len(order), d)
            )
        if not rows:
            return None
        return min(rows)[2]

    def _shift_tangent(self, tangent: Direction) -> Direction | None:
        """Continue the wall around a corner: first clear heading clockwise from ``tangent``.

        Used mid-episode when the committed tangent is blocked again. Scanning clockwise
        keeps the obstruction hugging the robot's right, matching the right-hand start.
        """
        order = list(Direction)
        shift = order.index(tangent)
        for step in range(1, len(order) + 1):
            candidate = order[(shift + step) % len(order)]
            if self._sensing.clear(candidate):
                return candidate
        return None


def _unit_offset(direction: Direction) -> tuple[float, float]:
    """The unit (length-1) compass offset of a direction, so diagonals score fairly."""
    ox, oy = direction.offset
    length = hypot(ox, oy)
    return ox / length, oy / length


def _perp_toward_goal(tangent: Direction, target: Vector2) -> Direction | None:
    """The cardinal perpendicular to ``tangent`` that leads toward the goal, if any.

    While sliding east or west the goal's north-south offset decides; while sliding north
    or south the east-west offset does. Diagonals have no single perpendicular axis, and a
    goal exactly abeam has none either — both return ``None``.
    """
    ox, oy = tangent.offset
    if abs(ox) == abs(oy):
        return None
    dx, dy = target
    if ox != 0:
        if dy > 0:
            return Direction.N
        if dy < 0:
            return Direction.S
        return None
    if dx < 0:
        return Direction.W
    if dx > 0:
        return Direction.E
    return None
