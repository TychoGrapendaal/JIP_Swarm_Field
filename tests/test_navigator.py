"""Navigator tests: the A.3 steering policy driven by a mock sensing harness.

Uses the shared fake world defined in ``nav_harness``: a tiny grid world keeps hidden ground
truth, ``GridSensing`` answers the navigator's proximity questions from that world, and
``Harness`` integrates ``Turn`` / ``Move`` the way the future Act stage would.
"""

from swarm_field.contracts import Direction, Idle, Move, Turn
from swarm_field.swarm import Navigator, SteeringConfig
from tests.nav_harness import GridSensing, GridWorld, Harness, crop_field, drive


def test_reaches_goal_across_open_field() -> None:
    world = GridWorld(width=5, height=5, blocked=frozenset())
    harness = Harness(world, start=(0, 0), facing=Direction.N)
    navigator = Navigator(GridSensing(harness))

    ticks = drive(navigator, harness, goal=(3, 2), max_ticks=20)

    assert harness.position == (3, 2)
    assert ticks <= 20
    assert isinstance(harness.actions[0], Turn)


def test_detours_around_a_single_crop_cell() -> None:
    world = GridWorld(width=5, height=5, blocked=frozenset({(2, 1)}))
    harness = Harness(world, start=(2, 0), facing=Direction.N)
    navigator = Navigator(GridSensing(harness))

    ticks = drive(navigator, harness, goal=(2, 2), max_ticks=30)

    assert harness.position == (2, 2)
    assert (2, 1) not in harness.visited
    assert ticks <= 30


def test_rounds_a_crop_row_to_reach_a_goal_aisle_cell() -> None:
    world = crop_field()
    harness = Harness(world, start=(5, 2), facing=Direction.N)
    navigator = Navigator(GridSensing(harness))

    ticks = drive(navigator, harness, goal=(5, 4), max_ticks=200)

    assert harness.position == (5, 4)
    assert ticks <= 200
    assert any(cell in harness.visited for cell in ((12, 2), (12, 3), (12, 4)))


def test_reports_stalled_when_fully_surrounded() -> None:
    pocket = frozenset({(x, y) for x in range(1, 4) for y in range(1, 4)} - {(2, 2)})
    world = GridWorld(width=5, height=5, blocked=pocket)
    harness = Harness(world, start=(2, 2), facing=Direction.N)
    navigator = Navigator(GridSensing(harness))

    action = harness.apply(navigator, goal=(0, 4))

    assert isinstance(action, Idle)
    assert navigator.is_stalled


def test_reports_stalled_after_circling_in_a_dead_end() -> None:
    wall = frozenset({(1, 3), (2, 3), (3, 3), (1, 1), (2, 1), (3, 1), (3, 2)})
    world = GridWorld(width=7, height=5, blocked=wall)
    harness = Harness(world, start=(0, 2), facing=Direction.E)
    navigator = Navigator(GridSensing(harness), SteeringConfig(stuck_ticks=12))

    for _ in range(300):
        harness.apply(navigator, goal=(6, 2))
        if navigator.is_stalled:
            break

    assert navigator.is_stalled
    assert harness.position != (6, 2)


def test_idles_when_goal_is_already_reached() -> None:
    world = GridWorld(width=3, height=3, blocked=frozenset())
    harness = Harness(world, start=(0, 0), facing=Direction.N)
    navigator = Navigator(GridSensing(harness))

    action = harness.apply(navigator, goal=(0, 0))

    assert isinstance(action, Idle)
    assert not navigator.is_stalled


def test_steering_is_deterministic() -> None:
    world = crop_field()
    harnesses = [Harness(world, start=(5, 2), facing=Direction.N) for _ in range(2)]
    navigators = [Navigator(GridSensing(h)) for h in harnesses]
    traces: list[list[Move | Turn | Idle]] = [[], []]
    for i in range(2):
        for _ in range(200):
            traces[i].append(harnesses[i].apply(navigators[i], (5, 4)))
            if harnesses[i].position == (5, 4):
                break

    assert harnesses[0].position == (5, 4)
    assert harnesses[1].position == (5, 4)
    assert traces[0] == traces[1]
