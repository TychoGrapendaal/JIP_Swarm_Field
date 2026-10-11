"""A quadtree over the field's cells, held as just two arrays: traversability and leaf sizes.

A leaf is a square block of cells that are all traversable or all blocked, aligned to
multiples of its size. ``leaf_size[x, y]`` is the size of the leaf whose corner (lowest x and
y) is cell ``(x, y)``, and 0 for every other cell of a leaf. An 8 x 4 field with a line of
obstacles ``#`` at x = 5, with each cell labelled by the leaf it belongs to::

    y   cells              leaves             leaf_size
    3   . . . . . # . .    A A A A e # G G    0 0 0 0 1 1 0 0
    2   . . . . . # . .    A A A A d # G G    0 0 0 0 1 1 2 0
    1   . . . . . # . .    A A A A c # F F    0 0 0 0 1 1 0 0
    0   . . . . . # . .    A A A A b # F F    4 0 0 0 1 1 2 0
        0 1 2 3 4 5 6 7

A is a 4 x 4 leaf with its corner at (0, 0); F and G are 2 x 2 leaves; the cells b to e at
x = 4, and the obstacle cells, are leaves of size 1: every 2 x 2 block they are part of mixes
traversable and blocked cells, so it can't be one leaf.

Everything else is derived from the two arrays. The leaf containing a cell is found by
rounding the cell down to multiples of 1, 2, 4, ... until it lands on a non-zero leaf size.
The neighbours of a leaf are found by walking along its sides, one neighbouring leaf at a
time.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import CellId, Direction, Terrain

_STRAIGHT_DIRECTIONS = (Direction.N, Direction.E, Direction.S, Direction.W)
_DIAGONAL_DIRECTIONS = (Direction.NE, Direction.SE, Direction.SW, Direction.NW)


class TerrainView(Protocol):
    """The terrain the traversability is read from; ``None`` means unknown."""

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    def terrain_at(self, cell: CellId) -> Terrain | None: ...


@dataclass(frozen=True)
class Leaf:
    """A square block of traversable cells: a node of the coarse search.

    Attributes:
        x: Column of the leaf's corner (its lowest x).
        y: Row of the leaf's corner (its lowest y).
        size: Side of the leaf, in cells.
    """

    x: int
    y: int
    size: int

    @property
    def centre(self) -> tuple[float, float]:
        """The centre of the leaf, in cell coordinates (cell ``(x, y)`` spans x to x + 1)."""
        return (self.x + self.size / 2, self.y + self.size / 2)

    def contains(self, cell: CellId) -> bool:
        x, y = cell
        return self.x <= x < self.x + self.size and self.y <= y < self.y + self.size


class Quadtree:
    """Which cells a robot can be in, per cell and grouped into leaves.

    Queries on cells: ``is_traversable`` and ``cell_neighbours``, for searching cell by
    cell. Queries on leaves: ``leaf_at``, ``leaf_neighbours`` and ``leaves``, for searching
    leaf by leaf. A robot may hop between neighbouring cells, straight or diagonally, but a
    diagonal hop is allowed only if both cells beside it are traversable, so robots never cut
    a corner past an obstacle. Two leaves are neighbours when a robot may hop from a cell of
    one into a cell of the other.

    Attributes:
        traversable: Whether a robot can be in each cell.
        leaf_size: The size of the leaf whose corner is at each cell; 0 inside a bigger leaf.
        max_leaf_side: The largest leaf size. Every leaf lies inside one aligned block of
            this many cells per side.
    """

    def __init__(
        self, traversable: NDArray[np.bool_], max_leaf_side: int = 16, connectivity: int = 8
    ) -> None:
        # Bit check if max_leaf_side is a power of 2
        if max_leaf_side < 1 or max_leaf_side & (max_leaf_side - 1) != 0:
            raise ValueError(f"max_leaf_side must be a power of two, got {max_leaf_side}")
        self.traversable = traversable.copy()
        self.leaf_size = find_leaves(self.traversable, max_leaf_side)
        self.max_leaf_side = max_leaf_side
        self._directions: tuple[Direction, ...] = _STRAIGHT_DIRECTIONS
        if connectivity == 8:
            self._directions = _STRAIGHT_DIRECTIONS + _DIAGONAL_DIRECTIONS
        # The hops as plain offsets: `cell_neighbours` runs for every node of every field built.
        self._offsets: tuple[tuple[int, int], ...] = tuple(d.offset for d in self._directions)
        self._version: int = 0

    @property
    def version(self) -> int:
        """Counts the changes to the traversability; a new value whenever a cell changed.

        A caller that keeps something derived from the quadtree — a potential field, say —
        compares this to notice that it was built on terrain that has since changed. An update
        that only repeats what the quadtree already holds, such as a robot re-sensing known
        terrain, keeps the version.
        """
        return self._version

    @property
    def width(self) -> int:
        return self.traversable.shape[0]

    @property
    def height(self) -> int:
        return self.traversable.shape[1]

    def update(self, changes: Mapping[CellId, bool]) -> None:
        """Set the traversability of the changed cells and rebuild only the leaves around them.

        Cells given their current traversability are skipped, and if none is left the
        quadtree, its version included, stays as it is. Every leaf lies inside one aligned
        block of ``max_leaf_side`` cells, so a change can only affect the leaves of the block
        it falls in; the other blocks stay as they are.
        """
        side = self.max_leaf_side
        blocks: set[CellId] = set()
        for (x, y), traversable in changes.items():
            if self.traversable[x, y] == traversable:
                continue
            self.traversable[x, y] = traversable
            blocks.add((x - x % side, y - y % side))
        if not blocks:
            return
        for block_x, block_y in blocks:
            block = (slice(block_x, block_x + side), slice(block_y, block_y + side))
            self.leaf_size[block] = find_leaves(self.traversable[block], side)
        self._version += 1

    # Queries on cells.

    def is_inside(self, cell: CellId) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_traversable(self, cell: CellId) -> bool:
        """Whether a robot can be in ``cell``; cells outside the field are not traversable."""
        return self.is_inside(cell) and bool(self.traversable[cell])

    def cell_neighbours(self, cell: CellId) -> Iterator[CellId]:
        """The cells a robot in ``cell`` may hop to.

        A diagonal hop, e.g. north-east from ``a`` to ``d``, needs both cells beside it::

            v d      a -> d is allowed only if h and v are traversable too
            a h
        """
        x, y = cell
        width, height = self.traversable.shape
        traversable = self.traversable
        for dx, dy in self._offsets:
            to_x = x + dx
            to_y = y + dy
            if not (0 <= to_x < width and 0 <= to_y < height and traversable[to_x, to_y]):
                continue
            is_diagonal = dx != 0 and dy != 0
            if is_diagonal and not (
                0 <= y < height and traversable[to_x, y] and 0 <= x < width and traversable[x, to_y]
            ):
                continue
            yield (to_x, to_y)

    # Queries on leaves.

    def leaf_at(self, cell: CellId) -> Leaf | None:
        """The leaf containing ``cell``, or ``None`` if a robot can't be there."""
        if not self.is_traversable(cell):
            return None
        return self._block_containing(cell)

    def leaves(self) -> Iterator[Leaf]:
        """Every leaf, i.e. every block of traversable cells."""
        corner_xs, corner_ys = np.nonzero((self.leaf_size > 0) & self.traversable)
        for x, y in zip(corner_xs.tolist(), corner_ys.tolist(), strict=True):
            yield Leaf(x, y, int(self.leaf_size[x, y]))

    def leaf_neighbours(self, leaf: Leaf) -> list[Leaf]:
        """The leaves a robot in ``leaf`` may hop into.

        Along each side, walk from one neighbouring block to the next: find the block containing
        the cell just beyond the side, then jump past that block. Blocked blocks are skipped.
        Two traversable leaves that share a side can always be hopped between, since all cells
        of both are traversable. Walking up the east side of the 4 x 4 leaf L::

            y
            3   L L L L D D      the column beyond the side is x = 4:
            2   L L L L D D      y = 0 is in b, jump past b to y = 1,
            1   L L L L c #      y = 1 is in c, jump past c to y = 2,
            0   L L L L b #      y = 2 is in D, jump past D to y = 4: the end of the side.
                0 1 2 3 4 5

        With diagonal hops, leaves that touch only at a corner are neighbours too, if a robot
        may hop diagonally between the two corner cells.
        """
        found: list[Leaf] = []
        for direction in _STRAIGHT_DIRECTIONS:
            for neighbour in self._blocks_along_side(leaf, direction):
                if self.traversable[neighbour.x, neighbour.y] and neighbour not in found:
                    found.append(neighbour)
        if Direction.NE in self._directions:
            for corner, beyond in self._diagonal_corners(leaf):
                if beyond in self.cell_neighbours(corner):
                    neighbour = self._block_containing(beyond)
                    if neighbour not in found:
                        found.append(neighbour)
        return found

    def _block_containing(self, cell: CellId) -> Leaf:
        """The block containing ``cell``, traversable or not.

        Blocks are aligned to multiples of their size, so rounding the cell down to multiples of
        1, 2, 4, ... stays inside the block containing it, until it lands on that block's corner:
        the first rounded cell with a non-zero leaf size.
        """
        x, y = cell
        for level in range(self.max_leaf_side.bit_length()):
            if self.leaf_size[x, y] != 0:
                return Leaf(x, y, int(self.leaf_size[x, y]))
            # `-1 << k` is all ones except the lowest k bits, so the AND clears those bits:
            # it rounds down to a multiple of 2**k.
            x = x & (-1 << (level + 1))
            y = y & (-1 << (level + 1))
        raise AssertionError(f"no block contains {cell}; leaf_size is inconsistent")

    def _blocks_along_side(self, leaf: Leaf, direction: Direction) -> Iterator[Leaf]:
        """The blocks just beyond one side of ``leaf``, in order along that side."""
        # The first cell beyond the side, and the unit step that walks along the side.
        match direction:
            case Direction.E:
                first, step = (leaf.x + leaf.size, leaf.y), (0, 1)
            case Direction.W:
                first, step = (leaf.x - 1, leaf.y), (0, 1)
            case Direction.N:
                first, step = (leaf.x, leaf.y + leaf.size), (1, 0)
            case _:
                first, step = (leaf.x, leaf.y - 1), (1, 0)
        if not self.is_inside(first):
            return

        # `along` counts cells along the side, from 0 to leaf.size - 1.
        along = 0
        while along < leaf.size:
            cell = (first[0] + step[0] * along, first[1] + step[1] * along)
            block = self._block_containing(cell)
            yield block
            # Jump past this block: continue at the first cell along the side beyond it.
            if step[0] == 1:
                along = block.x + block.size - first[0]
            else:
                along = block.y + block.size - first[1]

    def _diagonal_corners(self, leaf: Leaf) -> Iterator[tuple[CellId, CellId]]:
        """Each corner cell of ``leaf`` with the cell diagonally beyond it."""
        low_x, low_y = leaf.x, leaf.y
        high_x, high_y = leaf.x + leaf.size - 1, leaf.y + leaf.size - 1
        yield (high_x, high_y), (high_x + 1, high_y + 1)
        yield (high_x, low_y), (high_x + 1, low_y - 1)
        yield (low_x, low_y), (low_x - 1, low_y - 1)
        yield (low_x, high_y), (low_x - 1, high_y + 1)


def traversable_cells(terrain: TerrainView, flies: bool) -> NDArray[np.bool_]:
    """Read the traversability of every cell from the terrain."""
    traversable = np.zeros((terrain.width, terrain.height), dtype=np.bool_)
    for x in range(terrain.width):
        for y in range(terrain.height):
            traversable[x, y] = can_enter(terrain.terrain_at((x, y)), flies)
    return traversable


def can_enter(terrain: Terrain | None, flies: bool) -> bool:
    """Whether a robot can enter a cell of the given terrain (``None``: unknown).

    v1 cost model: every cell costs the same to enter, except that no robot can enter the
    outside of the field, and robots that don't fly can't enter obstacles.
    """
    return terrain is not Terrain.OUTSIDE and (flies or terrain is not Terrain.OBSTACLE)


def find_leaves(traversable: NDArray[np.bool_], max_leaf_side: int) -> NDArray[np.uint16]:
    """The size of the leaf whose corner (lowest x and y) is at each cell; 0 for cells that
    lie inside a bigger leaf. A leaf is a square block of cells that are all traversable or
    all blocked, aligned to multiples of its size."""
    width, height = traversable.shape
    # Pad with blocked cells so both sides are multiples of max_leaf_side: every block is whole.
    traversable = np.pad(traversable, ((0, -width % max_leaf_side), (0, -height % max_leaf_side)))

    # At the bottom, every cell is a leaf of size 1.
    leaf_size = np.ones(traversable.shape, dtype=np.uint16)

    big_block_size = 2
    while big_block_size <= max_leaf_side:
        # Four blocks with a side 2 times smaller constitute one big block.
        joined_blocks_size = big_block_size >> 1
        # The corners of the four joined blocks, one entry per big block. The big block's own
        # corner is its south-west quarter's corner:
        #
        #     +------------+------------+
        #     | north_west | north_east |     e.g. north_east[i, j] is the corner of the
        #     +------------+------------+     north-east quarter of big block (i, j)
        #     | south_west | south_east |
        #     +------------+------------+
        big = big_block_size
        half = joined_blocks_size
        south_west = (slice(0, None, big), slice(0, None, big))
        south_east = (slice(half, None, big), slice(0, None, big))
        north_west = (slice(0, None, big), slice(half, None, big))
        north_east = (slice(half, None, big), slice(half, None, big))
        quarters = (south_west, south_east, north_west, north_east)

        # Are all of the small blocks leaves of the same type? A quarter is still one whole
        # leaf when the size at its corner is the full quarter size (a smaller size means it
        # was split). The four have the same type when their corner cells agree on
        # traversability, since each whole leaf is uniform.
        should_merge = np.ones(traversable[south_west].shape, dtype=np.bool_)
        for quarter in quarters:
            should_merge &= leaf_size[quarter] == joined_blocks_size
        for quarter in (south_east, north_west, north_east):
            should_merge &= traversable[quarter] == traversable[south_west]

        # Merge: the big block's corner gets the big size, the other three corners become
        # members of a bigger leaf. `leaf_size[quarter]` is a view, so assigning through the
        # mask writes into leaf_size itself.
        leaf_size[south_west][should_merge] = big_block_size
        for quarter in (south_east, north_west, north_east):
            leaf_size[quarter][should_merge] = 0

        big_block_size <<= 1

    return leaf_size[:width, :height]
