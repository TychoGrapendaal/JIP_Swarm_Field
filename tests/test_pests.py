"""Tests for the pest experiment's scenario description."""

from collections import deque
from dataclasses import replace

from swarm_field.contracts import CellId
from swarm_field.experiments import RELAYED_KILLERS, RIVERS_LARGE


def test_lattice_starts_are_inside_the_field_and_reproducible() -> None:
    starts = RELAYED_KILLERS.starts()

    assert len(starts) == RELAYED_KILLERS.scouts + RELAYED_KILLERS.killers
    killers = starts[RELAYED_KILLERS.scouts :]
    assert len(set(killers)) == RELAYED_KILLERS.killers
    for x, y in starts:
        assert 0 <= x < RELAYED_KILLERS.size
        assert 0 <= y < RELAYED_KILLERS.size
    assert RELAYED_KILLERS.starts() == starts
    assert replace(RELAYED_KILLERS, seed=1).starts() != starts


def test_lattice_starts_avoid_obstacles() -> None:
    scenario = replace(RELAYED_KILLERS, obstacles=None, scouts=64, killers=1)
    blocked = set(scenario.blocked_cells())

    for cell in scenario.starts():
        assert cell not in blocked


def test_the_bridges_join_every_free_cell_of_the_two_rivers_field() -> None:
    size = RIVERS_LARGE.size
    blocked = set(RIVERS_LARGE.blocked_cells())
    start = (0, 0)
    reached: set[CellId] = {start}
    queue: deque[CellId] = deque([start])
    while queue:
        x, y = queue.popleft()
        for cell in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            inside = 0 <= cell[0] < size and 0 <= cell[1] < size
            if inside and cell not in blocked and cell not in reached:
                reached.add(cell)
                queue.append(cell)

    assert len(reached) == size * size - len(blocked)
