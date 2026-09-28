"""Tests for the quadtree: its leaves, and the queries derived from them."""

from random import Random

import numpy as np
import pytest
from numpy.typing import NDArray

from swarm_field.contracts import CellId
from swarm_field.swarm.field.quadtree import Leaf, Quadtree, find_leaves

BoolGrid = NDArray[np.bool_]


def _drawn(rows: list[str]) -> BoolGrid:
    """Traversability drawn as text rows, top row first: ``.`` traversable, ``#`` blocked."""
    rows = [row.replace(" ", "") for row in rows]
    traversable = np.zeros((len(rows[0]), len(rows)), dtype=np.bool_)
    for row_index, row in enumerate(rows):
        for x, symbol in enumerate(row):
            traversable[x, len(rows) - 1 - row_index] = symbol == "."
    return traversable


def _scattered(width: int, height: int, obstacles: int, seed: int) -> BoolGrid:
    rng = Random(seed)
    traversable = np.ones((width, height), dtype=np.bool_)
    for _ in range(obstacles):
        traversable[rng.randrange(width), rng.randrange(height)] = False
    return traversable


def _leaf_links_cell_by_cell(quadtree: Quadtree) -> dict[Leaf, set[Leaf]]:
    """Link two leaves wherever a cell of one may hop into a cell of the other."""
    links: dict[Leaf, set[Leaf]] = {leaf: set() for leaf in quadtree.leaves()}
    for x in range(quadtree.width):
        for y in range(quadtree.height):
            leaf = quadtree.leaf_at((x, y))
            if leaf is None:
                continue
            for neighbour_cell in quadtree.cell_neighbours((x, y)):
                neighbour = quadtree.leaf_at(neighbour_cell)
                if neighbour is not None and neighbour != leaf:
                    links[leaf].add(neighbour)
    return links


def test_find_leaves_merges_uniform_blocks_of_either_type() -> None:
    traversable = _drawn(
        [
            ". . . . . # . .",
            ". . . . . # . .",
            ". . . . . # . .",
            ". . . . . # . .",
        ]
    )
    expected_rows = [
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 2, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [4, 0, 0, 0, 1, 1, 2, 0],
    ]

    leaf_size = find_leaves(traversable, max_leaf_side=16)

    for row_index, expected in enumerate(expected_rows):
        assert leaf_size[:, 3 - row_index].tolist() == expected


def test_leaves_partition_the_traversable_cells() -> None:
    quadtree = Quadtree(_scattered(37, 29, obstacles=120, seed=7), max_leaf_side=8)

    for x in range(quadtree.width):
        for y in range(quadtree.height):
            leaf = quadtree.leaf_at((x, y))
            if quadtree.traversable[x, y]:
                assert leaf is not None
                assert leaf.contains((x, y))
            else:
                assert leaf is None
    total_area = sum(leaf.size**2 for leaf in quadtree.leaves())
    assert total_area == int(quadtree.traversable.sum())


@pytest.mark.parametrize("connectivity", [4, 8])
def test_leaf_neighbours_match_the_cell_level_hops(connectivity: int) -> None:
    quadtree = Quadtree(
        _scattered(40, 33, obstacles=150, seed=3), max_leaf_side=8, connectivity=connectivity
    )

    expected = _leaf_links_cell_by_cell(quadtree)

    for leaf in quadtree.leaves():
        assert set(quadtree.leaf_neighbours(leaf)) == expected[leaf]


def test_river_through_the_middle_of_a_block_separates_its_sides() -> None:
    quadtree = Quadtree(
        _drawn(
            [
                ". . . . . # # . . . . .",
                ". . . . . # # . . . . .",
                ". . . . . # # . . . . .",
                ". . . . . # # . . . . .",
            ]
        )
    )
    start = quadtree.leaf_at((0, 0))
    assert start is not None

    reached = {start}
    frontier = [start]
    while frontier:
        for neighbour in quadtree.leaf_neighbours(frontier.pop()):
            if neighbour not in reached:
                reached.add(neighbour)
                frontier.append(neighbour)

    assert quadtree.leaf_at((4, 0)) in reached
    assert quadtree.leaf_at((7, 0)) not in reached


def test_diagonal_hops_never_cut_a_corner() -> None:
    quadtree = Quadtree(
        _drawn(
            [
                "# .",
                ". #",
            ]
        )
    )

    assert list(quadtree.cell_neighbours((0, 0))) == []


def test_update_rebuilds_the_same_leaves_as_a_fresh_build() -> None:
    traversable = _scattered(45, 30, obstacles=60, seed=11)
    quadtree = Quadtree(traversable, max_leaf_side=16)
    rng = Random(5)
    changes: dict[CellId, bool] = {}
    for _ in range(40):
        changes[(rng.randrange(45), rng.randrange(30))] = rng.random() < 0.5

    quadtree.update(changes)
    for (x, y), value in changes.items():
        traversable[x, y] = value

    assert np.array_equal(quadtree.leaf_size, find_leaves(traversable, max_leaf_side=16))


def test_leaf_side_must_be_a_power_of_two() -> None:
    with pytest.raises(ValueError):
        Quadtree(np.ones((8, 8), dtype=np.bool_), max_leaf_side=12)
