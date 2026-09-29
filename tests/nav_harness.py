"""Shared harness for steering tests: a hidden grid world the tests stand in for the sim.

The simulation's world, perception and Act stage do not exist yet, so every steering test uses
this kit to stand in for them: a tiny grid world keeps hidden ground truth, a ``GridSensing``
answers the steerer's proximity questions from that world, and a ``Harness`` integrates ``Move``
the way the future Act stage would.
"""

from dataclasses import dataclass

from swarm_field.contracts import CellId, Direction, Idle, Move
from swarm_field.swarm import Steerer


@dataclass(frozen=True)
class GridWorld:
    """A tiny hidden field: passable cells plus the cells that block movement.

    Attributes:
        width: Number of columns; valid x coords are 0..width-1.
        height: Number of rows; valid y coords are 0..height-1.
        blocked: The cells that land robots may not enter.
    """

    width: int
    height: int
    blocked: frozenset[CellId]

    def can_enter(self, position: CellId) -> bool:
        """Whether ``position`` lies inside the world and is not blocked."""
        x, y = position
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return position not in self.blocked

    def clear_cell(self, position: CellId, direction: Direction) -> bool:
        """Whether the cell one ``direction`` step from ``position`` is clear to enter."""
        x, y = position
        dx, dy = direction.offset
        return self.can_enter((x + dx, y + dy))


class Harness:
    """Mock movement — the quiet stand-in for the sim's Perceive / Act stages.

    Applies one action to a book-kept position, and refuses to step into a blocked cell:
    if the steerer ever tries, the test blows up.
    """

    def __init__(self, world: GridWorld, start: CellId) -> None:
        self.world: GridWorld = world
        self.position: CellId = start
        self.visited: list[CellId] = [start]
        self.actions: list[Move | Idle] = []

    def take(self, action: Move | Idle) -> Move | Idle:
        """Integrate one action, asserting it never steps into a blocked cell."""
        self.actions.append(action)
        if isinstance(action, Move):
            assert self.world.can_enter(action.to), "move into a blocked cell"
            self.position = action.to
            self.visited.append(self.position)
        return action

    def apply(self, steerer: Steerer, goal: CellId) -> Move | Idle:
        """Ask the steerer for a step toward an absolute ``goal`` and integrate it."""
        dx = float(goal[0] - self.position[0])
        dy = float(goal[1] - self.position[1])
        return self.take(steerer.steer((dx, dy), self.position))


@dataclass(frozen=True)
class GridSensing:
    """A fake ``Sensing`` backed by the hidden grid of a harness."""

    world: GridWorld

    def clear(self, origin: CellId, direction: Direction) -> bool:
        return self.world.clear_cell(origin, direction)


def drive(steerer: Steerer, harness: Harness, goal: CellId, max_ticks: int = 400) -> int:
    """Drive ``harness`` toward ``goal`` until reached, stalled, or ``max_ticks``."""
    for tick in range(1, max_ticks + 1):
        harness.apply(steerer, goal)
        if harness.position == goal:
            return tick
        if steerer.is_stalled:
            raise AssertionError(f"steerer got stuck at {harness.position}")
    raise AssertionError(f"did not reach {goal}; ended at {harness.position}")


def crop_field() -> GridWorld:
    """The 13-by-9 crop field of ``docs/navigation-module.md`` §8.1.

    Crop rows occupy the odd y-rows (y = 1, 3, 5, 7) between x = 1 and x = 11; the even
    y-rows are aisles and the x = 0 and x = 12 columns are headland lanes.
    """
    blocked = frozenset((x, y) for y in (1, 3, 5, 7) for x in range(1, 12))
    return GridWorld(width=13, height=9, blocked=blocked)
