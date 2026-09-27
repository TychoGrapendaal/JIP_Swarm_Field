"""A.3 Steering — turn a relative goal into a concrete ``Move`` action, with no map.

The ``Steerer`` is the reactive movement policy of component A (the brain) and the fallback
the ``Navigator`` uses when a potential field cannot guide the robot (see
:mod:`swarm_field.swarm.navigator`). It turns a goal — a relative displacement vector from the
robot to the cell it is heading for — into one physical action per decision point: a ``Move``
naming the adjacent target cell to enter (the world has no turn/facing model; movement is a
direct target-cell command). It hands in its absolute ``position`` each decision point, and
spatial knowledge arrives only through the ``Sensing`` port, which reports whether each
adjacent grid cell is clear to enter.

Obstacle avoidance is a committed right-hand wall-follow (Bug-algorithm style). When the goal
heading is blocked the steerer picks the most goal-aligned clear heading (ties to the fixed
right hand) and commits to it for the episode: it hugs the wall, rounding corners toward the
goal as soon as the tangent ahead gives out, and never re-consults the raw goal bearing while
sliding (that is what turns a row's south aisle into a dead-end trap). It only leaves the wall
when the goal is right next to it. A dead end with no route at all flips ``is_stalled`` —
re-routing stays the brain's job (A.1), not the steerer's.

Because it reads nothing but the clearance ring, the steerer needs no map and no memory: it
works over unknown ground and at any distance, but it can only see one cell ahead. Guiding a
robot over a whole field is the ``Navigator``'s job, through the potential field of an
objective; this policy is what that plan falls back on.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Protocol

from swarm_field.contracts import CellId, Direction, Idle, Move, Vector2
from swarm_field.swarm.field import Quadtree


class Sensing(Protocol):
    """Local proximity ring — is the adjacent cell in a given heading clear to enter?

    The robot can ask, per compass heading, whether the grid cell one step from where it
    stands is free to move into. Clearance may come from a live sensor ring or from believed
    terrain; :class:`QuadtreeSensing` is the map-backed one, and tests supply fakes driven by
    a hidden grid. The steerer itself stays free of any absolute geometry beyond the origin it
    is handed.
    """

    def clear(self, origin: CellId, direction: Direction) -> bool: ...


@dataclass(frozen=True)
class QuadtreeSensing:
    """A ``Sensing`` that answers from the traversable cells of a quadtree.

    It asks the quadtree the same question its own routing does, through
    :meth:`~swarm_field.swarm.field.Quadtree.cell_neighbours`, so the steerer and the fields
    built from that quadtree never disagree on where a robot may go: a diagonal needs both
    cells beside it traversable, and nothing outside the field is traversable.

    Attributes:
        quadtree: Where this robot believes it can be and move.
    """

    quadtree: Quadtree

    def clear(self, origin: CellId, direction: Direction) -> bool:
        dx, dy = direction.offset
        return (origin[0] + dx, origin[1] + dy) in set(self.quadtree.cell_neighbours(origin))


@dataclass(frozen=True)
class SteeringConfig:
    """Tuning knobs for the steering policy.

    Attributes:
        target_tolerance: Cell distance below which the goal counts as reached.
        near_goal: Cell distance at which the goal counts as right next to the robot, so a
            wall-following episode may legitimately break off toward it.
        stuck_ticks: Detour ticks without the goal distance beating its running minimum
            before the steerer declares itself stuck.
        progress_epsilon: Smallest goal-distance decrease, in cells, that counts as
            sensible progress toward the goal.
    """

    target_tolerance: float = 0.75
    near_goal: float = 1.5
    stuck_ticks: int = 40
    progress_epsilon: float = 0.01


class Steerer:
    """One-tick steering policy: a relative goal in, a physical action out.

    Public API: :meth:`steer` returns this tick's action; :pyattr:`is_stalled` reports that
    the goal cannot be reached from here; :pyattr:`is_detouring` reports that the robot is
    currently wall-following. State is limited to the policy's short memory (the committed
    tangent, stuck accounting); the robot's own position and the field layout live outside
    the steerer.
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

        A stalled steerer idles and keeps reporting until :meth:`reset` is called — the brain
        reacts by re-issuing or replanning the goal.
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

    def steer(self, target: Vector2, position: CellId) -> Move | Idle:
        """Pick this tick's action for a goal at relative offset ``target``.

        Args:
            target: Relative displacement from the robot to the goal, in cells.
            position: The robot's current cell, the origin every clearance probe is asked about.
        """
        dx, dy = target
        goal_dist = hypot(dx, dy)

        if goal_dist <= self._config.target_tolerance:
            self.reset()
            return Idle()

        if self._stalled:
            return Idle()

        desired = self._best_aligned(target)

        if self._detour_ticks == 0:
            if self._clear(position, desired):
                return self._head(desired, position)
            tangent = self._choose_tangent(position, desired)
            if tangent is None:
                self._stalled = True
                return Idle()
            self._tangent = tangent
            self._detour_ticks = 1
            return self._track_progress(goal_dist, tangent, position)

        tangent = self._tangent
        assert tangent is not None
        if not self._clear(position, tangent):
            perp = _perp_toward_goal(tangent, target)
            if perp is not None and self._clear(position, perp):
                tangent = perp
                self._tangent = tangent
            else:
                tangent = self._shift_tangent(position, tangent)
                if tangent is None:
                    self._stalled = True
                    return Idle()
                self._tangent = tangent

        if self._clear(position, desired) and (
            goal_dist <= self._config.near_goal or self._level_along(tangent, target)
        ):
            self._end_detour()
            return self._head(desired, position)

        self._detour_ticks += 1
        return self._track_progress(goal_dist, tangent, position)

    def _track_progress(
        self, goal_dist: float, tangent: Direction, position: CellId
    ) -> Move | Idle:
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
        return self._head(tangent, position)

    def _head(self, direction: Direction, position: CellId) -> Move:
        """The ``Move`` one step ``direction`` away from ``position``."""
        dx, dy = direction.offset
        return Move(to=(position[0] + dx, position[1] + dy))

    def _clear(self, position: CellId, direction: Direction) -> bool:
        """Whether the cell one ``direction`` step from ``position`` is clear to enter."""
        return self._sensing.clear(position, direction)

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

    def _best_aligned(self, target: Vector2) -> Direction:
        """The compass heading closest to the goal direction, ties by enum order.

        Candidates are ranked by alignment score (max); ties fall back to enum order so the
        result stays deterministic across ticks without a turn/facing model to break them.
        """
        dx, dy = target
        goal_dist = hypot(dx, dy)
        order = list(Direction)
        rows: list[tuple[float, int, Direction]] = []
        for d in order:
            ux, uy = _unit_offset(d)
            score = (dx * ux + dy * uy) / goal_dist
            rows.append((-score, order.index(d), d))
        return min(rows)[2]

    def _choose_tangent(self, position: CellId, desired: Direction) -> Direction | None:
        """The clear heading to start a detour with, when ``desired`` is blocked.

        Among the clear neighbours, pick the one most aligned with ``desired``; ties go to
        whichever is closest clockwise from ``desired`` (the fixed right hand). Chosen once
        per episode — the steerer commits to it; only the wall itself re-routes it.
        """
        order = list(Direction)
        dux, duy = _unit_offset(desired)
        rows: list[tuple[float, int, Direction]] = []
        for d in order:
            if not self._clear(position, d):
                continue
            ux, uy = _unit_offset(d)
            rows.append(
                (-(dux * ux + duy * uy), (order.index(d) - order.index(desired)) % len(order), d)
            )
        if not rows:
            return None
        return min(rows)[2]

    def _shift_tangent(self, position: CellId, tangent: Direction) -> Direction | None:
        """Continue the wall around a corner: first clear heading clockwise from ``tangent``.

        Used mid-episode when the committed tangent is blocked again. Scanning clockwise
        keeps the obstruction hugging the robot's right, matching the right-hand start.
        """
        order = list(Direction)
        shift = order.index(tangent)
        for step in range(1, len(order) + 1):
            candidate = order[(shift + step) % len(order)]
            if self._clear(position, candidate):
                return candidate
        return None


def _unit_offset(direction: Direction) -> tuple[float, float]:
    """The unit (length-1) compass offset of a direction, so diagonals score fairly."""
    ox, oy = direction.offset
    length = hypot(ox, oy)
    return ox / length, oy / length


def _perp_toward_goal(tangent: Direction, target: Vector2) -> Direction | None:
    """The cardinal perpendicular to ``tangent`` that leads toward the goal, if any.

    While sliding east or west the goal's north-south offset decides; while sliding north or
    south the east-west offset does. Diagonals have no single perpendicular axis, and a goal
    exactly abeam has none either — both return ``None``.
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
