"""Navigator tests: the A.3 steering policy driven by a mock sensing harness.

The rest of the system (perception, memory, the sim's world and its Act stage) does not
exist yet, so these tests stand in for it: a tiny grid world keeps hidden ground truth, a
``_GridSensing`` answers the navigator's proximity questions from that world, and a
``_Harness`` integrates ``Turn`` / ``Move`` the way the future Act stage would.
"""

from dataclasses import dataclass

from swarm_field.contracts import Direction, Idle, Move, Turn
from swarm_field.swarm import Navigator, SteeringConfig


@dataclass(frozen=True)
class _GridWorld:
    """A tiny hidden field: passable cells plus the cells that block movement."""

    width: int
    height: int
    blocked: frozenset[tuple[int, int]]

    def clear_cell(self, position: tuple[int, int], direction: Direction) -> bool:
        x, y = position
        dx, dy = direction.offset
        nx, ny = x + dx, y + dy
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return False
        return (nx, ny) not in self.blocked


class _Harness:
    """Mock movement — the quiet stand-in for the sim's Perceive / Act stages.

    Applies one action to a book-kept position and heading, and refuses to step into a
    blocked cell: if the navigator ever tries, the test blows up.
    """

    def __init__(self, world: _GridWorld, start: tuple[int, int], facing: Direction) -> None:
        self.world: _GridWorld = world
        self.position: tuple[int, int] = start
        self.facing: Direction = facing
        self.visited: list[tuple[int, int]] = [start]
        self.actions: list[Move | Turn | Idle] = []

    def apply(self, navigator: Navigator, goal: tuple[int, int]) -> Move | Turn | Idle:
        dx = float(goal[0] - self.position[0])
        dy = float(goal[1] - self.position[1])
        action = navigator.steer((dx, dy), self.facing)
        self.actions.append(action)
        if isinstance(action, Turn):
            assert self.world.clear_cell(self.position, action.facing)
            self.facing = action.facing
        if isinstance(action, Move):
            assert self.world.clear_cell(self.position, action.direction), (
                "move into a blocked cell"
            )
            x, y = self.position
            dx2, dy2 = action.direction.offset
            self.position = (x + dx2, y + dy2)
            self.visited.append(self.position)
        return action


class _GridSensing:
    """A fake ``Sensing`` backed by the hidden grid via the harness's true position."""

    def __init__(self, harness: _Harness) -> None:
        self._harness: _Harness = harness

    def clear(self, direction: Direction) -> bool:
        return self._harness.world.clear_cell(self._harness.position, direction)


def _drive(
    navigator: Navigator,
    harness: _Harness,
    goal: tuple[int, int],
    max_ticks: int = 400,
) -> int:
    for tick in range(1, max_ticks + 1):
        harness.apply(navigator, goal)
        if harness.position == goal:
            return tick
        if navigator.is_stalled:
            raise AssertionError(f"navigator got stuck at {harness.position}")
    raise AssertionError(f"did not reach {goal}; ended at {harness.position}")


def _field() -> _GridWorld:
    """The 13-by-9 crop field of navigation-module.md §8.1.

    Crop rows occupy the odd y-rows (y = 1, 3, 5, 7) between x = 1 and x = 11; the even
    y-rows are aisles and the x = 0 and x = 12 columns are headland lanes.
    """
    blocked = frozenset((x, y) for y in (1, 3, 5, 7) for x in range(1, 12))
    return _GridWorld(width=13, height=9, blocked=blocked)


def test_reaches_goal_across_open_field() -> None:
    world = _GridWorld(width=5, height=5, blocked=frozenset())
    harness = _Harness(world, start=(0, 0), facing=Direction.N)
    navigator = Navigator(_GridSensing(harness))

    ticks = _drive(navigator, harness, goal=(3, 2), max_ticks=20)

    assert harness.position == (3, 2)
    assert ticks <= 20
    assert isinstance(harness.actions[0], Turn)


def test_detours_around_a_single_crop_cell() -> None:
    world = _GridWorld(width=5, height=5, blocked=frozenset({(2, 1)}))
    harness = _Harness(world, start=(2, 0), facing=Direction.N)
    navigator = Navigator(_GridSensing(harness))

    ticks = _drive(navigator, harness, goal=(2, 2), max_ticks=30)

    assert harness.position == (2, 2)
    assert (2, 1) not in harness.visited
    assert ticks <= 30


def test_rounds_a_crop_row_to_reach_a_goal_aisle_cell() -> None:
    world = _field()
    harness = _Harness(world, start=(5, 2), facing=Direction.N)
    navigator = Navigator(_GridSensing(harness))

    ticks = _drive(navigator, harness, goal=(5, 4), max_ticks=200)

    assert harness.position == (5, 4)
    assert ticks <= 200
    assert any(cell in harness.visited for cell in ((12, 2), (12, 3), (12, 4)))


def test_reports_stalled_when_fully_surrounded() -> None:
    pocket = frozenset({(x, y) for x in range(1, 4) for y in range(1, 4)} - {(2, 2)})
    world = _GridWorld(width=5, height=5, blocked=pocket)
    harness = _Harness(world, start=(2, 2), facing=Direction.N)
    navigator = Navigator(_GridSensing(harness))

    action = harness.apply(navigator, goal=(0, 4))

    assert isinstance(action, Idle)
    assert navigator.is_stalled


def test_reports_stalled_after_circling_in_a_dead_end() -> None:
    wall = frozenset({(1, 3), (2, 3), (3, 3), (1, 1), (2, 1), (3, 1), (3, 2)})
    world = _GridWorld(width=7, height=5, blocked=wall)
    harness = _Harness(world, start=(0, 2), facing=Direction.E)
    navigator = Navigator(_GridSensing(harness), SteeringConfig(stuck_ticks=12))

    for _ in range(300):
        harness.apply(navigator, goal=(6, 2))
        if navigator.is_stalled:
            break

    assert navigator.is_stalled
    assert harness.position != (6, 2)


def test_idles_when_goal_is_already_reached() -> None:
    world = _GridWorld(width=3, height=3, blocked=frozenset())
    harness = _Harness(world, start=(0, 0), facing=Direction.N)
    navigator = Navigator(_GridSensing(harness))

    action = harness.apply(navigator, goal=(0, 0))

    assert isinstance(action, Idle)
    assert not navigator.is_stalled


def test_steering_is_deterministic() -> None:
    world = _field()
    harnesses = [_Harness(world, start=(5, 2), facing=Direction.N) for _ in range(2)]
    navigators = [Navigator(_GridSensing(h)) for h in harnesses]
    traces: list[list[Move | Turn | Idle]] = [[], []]
    for i in range(2):
        for _ in range(200):
            traces[i].append(harnesses[i].apply(navigators[i], (5, 4)))
            if harnesses[i].position == (5, 4):
                break

    assert harnesses[0].position == (5, 4)
    assert harnesses[1].position == (5, 4)
    assert traces[0] == traces[1]
