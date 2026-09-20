"""Shared harness for navigator tests: a hidden grid world the tests stand in for the sim.

The simulation's world, perception and Act stage do not exist yet, so every navigator test
uses this kit to stand in for them: a tiny grid world keeps hidden ground truth, a
``GridSensing`` answers the navigator's proximity questions from that world, and a
``Harness`` integrates ``Turn`` / ``Move`` the way the future Act stage would.
"""

from dataclasses import dataclass

from swarm_field.contracts import Direction, Idle, Move, Turn
from swarm_field.swarm import Navigator


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
    blocked: frozenset[tuple[int, int]]

    def clear_cell(self, position: tuple[int, int], direction: Direction) -> bool:
        """Whether the cell one ``direction`` step from ``position`` is clear to enter."""
        x, y = position
        dx, dy = direction.offset
        nx, ny = x + dx, y + dy
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return False
        return (nx, ny) not in self.blocked


class Harness:
    """Mock movement — the quiet stand-in for the sim's Perceive / Act stages.

    Applies one action to a book-kept position and heading, and refuses to step into a
    blocked cell: if the navigator ever tries, the test blows up.
    """

    def __init__(self, world: GridWorld, start: tuple[int, int], facing: Direction) -> None:
        self.world: GridWorld = world
        self.position: tuple[int, int] = start
        self.facing: Direction = facing
        self.visited: list[tuple[int, int]] = [start]
        self.actions: list[Move | Turn | Idle] = []

    def take(self, action: Move | Turn | Idle) -> Move | Turn | Idle:
        """Integrate one action, asserting it never steps into a blocked cell."""
        self.actions.append(action)
        if isinstance(action, Turn):
            assert self.world.clear_cell(self.position, action.facing)
            self.facing = action.facing
        if isinstance(action, Move):
            assert self.world.clear_cell(self.position, action.direction), (
                "move into a blocked cell"
            )
            x, y = self.position
            dx, dy = action.direction.offset
            self.position = (x + dx, y + dy)
            self.visited.append(self.position)
        return action

    def apply(self, navigator: Navigator, goal: tuple[int, int]) -> Move | Turn | Idle:
        """Ask the navigator for a step toward an absolute ``goal`` and integrate it."""
        dx = float(goal[0] - self.position[0])
        dy = float(goal[1] - self.position[1])
        return self.take(navigator.steer((dx, dy), self.facing))


class GridSensing:
    """A fake ``Sensing`` backed by the hidden grid via the harness's true position."""

    def __init__(self, harness: Harness) -> None:
        self._harness: Harness = harness

    def clear(self, direction: Direction) -> bool:
        return self._harness.world.clear_cell(self._harness.position, direction)


def drive(
    navigator: Navigator, harness: Harness, goal: tuple[int, int], max_ticks: int = 400
) -> int:
    """Drive ``harness`` toward ``goal`` until reached, stalled, or ``max_ticks``."""
    for tick in range(1, max_ticks + 1):
        harness.apply(navigator, goal)
        if harness.position == goal:
            return tick
        if navigator.is_stalled:
            raise AssertionError(f"navigator got stuck at {harness.position}")
    raise AssertionError(f"did not reach {goal}; ended at {harness.position}")


def crop_field() -> GridWorld:
    """The 13-by-9 crop field of ``docs/navigation-module.md`` §8.1.

    Crop rows occupy the odd y-rows (y = 1, 3, 5, 7) between x = 1 and x = 11; the even
    y-rows are aisles and the x = 0 and x = 12 columns are headland lanes.
    """
    blocked = frozenset((x, y) for y in (1, 3, 5, 7) for x in range(1, 12))
    return GridWorld(width=13, height=9, blocked=blocked)
