"""Scenario 2 tests: the utility robot chasing the swarm's pest gradient.

Replays the walkthrough of ``docs/navigation-module.md`` §8.1. Three explorers hover over
the y=4 aisle of the 13-by-9 crop field and broadcast hop beacons; a small ``_HopField``
stands in for A.4, decoding each decision point into the ``HopFieldEntry`` list a real
comms layer would produce from the radio: the robot's relative vector to every in-range
explorer plus that explorer's ``parent_vector``. The navigator composes ``to_holder +
parent_vector`` and steers — the "swarm gives the vectors" claim, end to end.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot

from swarm_field.contracts import (
    HopFieldEntry,
    Idle,
    NavGoal,
    NavMode,
    Vector2,
)
from swarm_field.swarm import Navigator, resolve_chase_target
from tests.nav_harness import GridSensing, Harness, crop_field


@dataclass(frozen=True)
class _Explorer:
    """An explorer hovering over ``hover_cell``, broadcasting a hop beacon.

    Attributes:
        hover_cell: The cell the explorer hovers above (aerial bodies ignore crop cells).
        hop: The explorer's hop number; 0 means it sees the pest itself.
        parent_vector: The explorer's vector to the next hop down (its beacon payload).
    """

    hover_cell: tuple[int, int]
    hop: int
    parent_vector: Vector2


def _walkthrough_swarm() -> tuple[_Explorer, _Explorer, _Explorer]:
    """The three explorers of ``docs/navigation-module.md`` §8.1, on the y=4 aisle."""
    return (
        _Explorer(hover_cell=(9, 4), hop=2, parent_vector=(-2.0, 0.0)),
        _Explorer(hover_cell=(7, 4), hop=1, parent_vector=(-2.0, 0.0)),
        _Explorer(hover_cell=(5, 4), hop=0, parent_vector=(0.0, 1.0)),
    )


class _HopField:
    """A.4 stand-in: what the utility robot hears from a static swarm right now.

    Each call recomputes the robot's radio vector to every explorer in range and dresses it
    with that explorer's beacon — the decoded values a real A.4 would produce from the
    channel and the robot's radio sensing. Freshness is fixed so the tie-breaks stay
    deterministic.
    """

    def __init__(self, explorers: Sequence[_Explorer], radio_range: float) -> None:
        self._explorers: Sequence[_Explorer] = explorers
        self._radio_range: float = radio_range

    def at(self, position: tuple[int, int]) -> list[HopFieldEntry]:
        entries: list[HopFieldEntry] = []
        for explorer in self._explorers:
            dx = explorer.hover_cell[0] - position[0]
            dy = explorer.hover_cell[1] - position[1]
            if hypot(dx, dy) <= self._radio_range:
                entries.append(
                    HopFieldEntry(
                        holder_hop=explorer.hop,
                        parent_vector=explorer.parent_vector,
                        to_holder=(float(dx), float(dy)),
                        seen_at=0,
                    )
                )
        return entries


def test_composes_the_doc_walkthrough_vectors() -> None:
    field = _HopField(_walkthrough_swarm(), radio_range=3.0)

    step1 = resolve_chase_target(field.at((11, 4)), max_hop=2)
    step2 = resolve_chase_target(field.at((9, 4)), max_hop=2)
    step3 = resolve_chase_target(field.at((7, 4)), max_hop=2)
    step4 = resolve_chase_target(field.at((5, 4)), max_hop=2)

    assert step1 is not None and step1.target == (-4.0, 0.0) and step1.holder_hop == 2
    assert step2 is not None and step2.target == (-4.0, 0.0) and step2.holder_hop == 1
    assert step3 is not None and step3.target == (-2.0, 1.0) and step3.holder_hop == 0
    assert step4 is not None and step4.target == (0.0, 1.0) and step4.holder_hop == 0


def test_pilot_prefers_lower_hop_then_closer_then_fresher() -> None:
    lower_hop = [
        HopFieldEntry(holder_hop=2, parent_vector=(0.0, 0.0), to_holder=(0.5, 0.0), seen_at=0),
        HopFieldEntry(holder_hop=1, parent_vector=(0.0, 0.0), to_holder=(5.0, 0.0), seen_at=0),
    ]
    resolved = resolve_chase_target(lower_hop, max_hop=2)
    assert resolved is not None and resolved.holder_hop == 1

    nearer = [
        HopFieldEntry(holder_hop=1, parent_vector=(0.0, 0.0), to_holder=(2.0, 0.0), seen_at=5),
        HopFieldEntry(holder_hop=1, parent_vector=(0.0, 0.0), to_holder=(1.0, 0.0), seen_at=5),
    ]
    resolved = resolve_chase_target(nearer, max_hop=2)
    assert resolved is not None and resolved.to_holder == (1.0, 0.0)

    fresher = [
        HopFieldEntry(holder_hop=1, parent_vector=(0.0, 1.0), to_holder=(1.0, 0.0), seen_at=5),
        HopFieldEntry(holder_hop=1, parent_vector=(0.0, 2.0), to_holder=(1.0, 0.0), seen_at=12),
    ]
    resolved = resolve_chase_target(fresher, max_hop=2)
    assert resolved is not None and resolved.target == (1.0, 2.0)


def test_max_hop_filters_the_field() -> None:
    field = [
        HopFieldEntry(holder_hop=3, parent_vector=(-2.0, 0.0), to_holder=(0.0, 0.0), seen_at=0)
    ]

    assert resolve_chase_target(field, max_hop=2) is None


def test_chase_drives_utility_robot_to_the_pest_row_edge() -> None:
    world = crop_field()
    harness = Harness(world, start=(11, 4))
    navigator = Navigator(GridSensing(harness))
    hop_field = _HopField(_walkthrough_swarm(), radio_range=3.0)
    goal = NavGoal(mode=NavMode.CHASE, max_hop=2)
    max_hop = goal.max_hop
    assert max_hop is not None

    pilots: list[int] = []
    stood: bool = False
    for _ in range(250):
        heard = hop_field.at(harness.position)
        resolved = resolve_chase_target(heard, max_hop)
        if resolved is None:
            raise AssertionError("no beacon heard at the start of the walkthrough")
        pilots.append(resolved.holder_hop)
        action = harness.take(navigator.steer_goal(goal, heard, harness.position))
        if isinstance(action, Idle) and resolved.holder_hop == 0:
            stood = True
            break
        if navigator.is_stalled:
            raise AssertionError(f"navigator stalled at {harness.position}")

    assert stood
    assert harness.position == (5, 4)
    assert not navigator.is_stalled
    assert pilots[0] == 2
    first_seen = [pilots.index(hop) for hop in (2, 1, 0)]
    assert first_seen == sorted(first_seen)
    visited = harness.visited
    assert visited.index((9, 4)) < visited.index((7, 4)) < visited.index((5, 4))


def test_chase_stands_at_the_hop_zero_row_edge() -> None:
    harness = Harness(crop_field(), start=(5, 4))
    navigator = Navigator(GridSensing(harness))
    field = [HopFieldEntry(holder_hop=0, parent_vector=(0.0, 1.0), to_holder=(0.0, 0.0), seen_at=0)]

    action = navigator.steer_chase(field, max_hop=2, position=harness.position)

    assert isinstance(action, Idle)
    assert not navigator.is_stalled
    assert harness.position == (5, 4)


def test_chase_idles_when_no_beacon_speaks() -> None:
    harness = Harness(crop_field(), start=(11, 4))
    navigator = Navigator(GridSensing(harness))

    action = navigator.steer_chase([], max_hop=2, position=harness.position)

    assert isinstance(action, Idle)
    assert not navigator.is_stalled


def test_hold_goal_stands_still() -> None:
    harness = Harness(crop_field(), start=(11, 4))
    navigator = Navigator(GridSensing(harness))

    action = navigator.steer_goal(NavGoal(mode=NavMode.HOLD), [], position=harness.position)

    assert isinstance(action, Idle)
    assert harness.position == (11, 4)


def test_spread_goal_moves_along_drift_without_entering_crops() -> None:
    world = crop_field()
    harness = Harness(world, start=(11, 2))
    navigator = Navigator(GridSensing(harness))
    goal = NavGoal(mode=NavMode.SPREAD, drift=(0.0, 10.0))

    for _ in range(20):
        harness.take(navigator.steer_goal(goal, [], position=harness.position))
        if navigator.is_stalled:
            break

    assert harness.position != (11, 2)
    assert not isinstance(harness.actions[0], Idle)
    assert not (set(harness.visited) & set(world.blocked))
