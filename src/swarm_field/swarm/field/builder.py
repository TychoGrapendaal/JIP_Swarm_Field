"""Build grid potential fields with one multi-source Dijkstra over a mixed-resolution graph.

Around the robot the graph has single cells; everywhere else it has the leaves of the
quadtree. The window of single cells is made of whole blocks of ``max_leaf_side`` cells, the
robot's block and the blocks within ``window_blocks`` of it. Every leaf lies inside one such
block, so a leaf is either entirely inside the window, and its cells are used, or entirely
outside it, and the leaf itself is used. With ``window_blocks = 1``::

    +---+---+---+---+---+
    | L | L | L | L | L |     R  the robot's block
    +---+---+---+---+---+     .  blocks inside the window, searched cell by cell
    | L | . | . | . | L |     L  blocks outside it, searched leaf by leaf; a block is
    +---+---+---+---+---+        one leaf or several smaller ones
    | L | . | R | . | L |
    +---+---+---+---+---+
    | L | . | . | . | L |
    +---+---+---+---+---+
    | L | L | L | L | L |
    +---+---+---+---+---+

All costs are in cells: 1 for a straight hop between cells, 1.4 for a diagonal one, and the
octile distance between centres wherever a leaf is involved.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import inf

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import CellId
from swarm_field.swarm.field.dijkstra import IndexedGraph, SearchResult, SettledNode
from swarm_field.swarm.field.quadtree import Leaf, Quadtree

Node = CellId | Leaf
"""A node of the search: a cell inside the window, or a leaf outside it."""

_DIAGONAL_LENGTH = 1.4
"""Cost of a diagonal hop relative to a straight one (an approximation of the square root of 2)."""


@dataclass(frozen=True)
class Seed:
    """A target cell and its start value, ``-beta_o * v_t``: the lower, the more attractive.

    Attributes:
        cell: The target cell.
        value: The start value of the wave spreading from the target, in cells.
    """

    cell: CellId
    value: float


@dataclass(frozen=True)
class BoundingBox:
    """A rectangle of cells: ``x_min <= x < x_max`` and ``y_min <= y < y_max``.

    Attributes:
        x_min: First column inside the box.
        y_min: First row inside the box.
        x_max: First column past the box.
        y_max: First row past the box.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def contains(self, cell: CellId) -> bool:
        x, y = cell
        return self.x_min <= x < self.x_max and self.y_min <= y < self.y_max

    def edge_cells(self) -> Iterator[CellId]:
        """The cells on the box's border, each once."""
        top = self.y_max - 1
        right = self.x_max - 1
        for x in range(self.x_min, self.x_max):
            yield (x, self.y_min)
            if top != self.y_min:
                yield (x, top)
        for y in range(self.y_min + 1, top):
            yield (self.x_min, y)
            if right != self.x_min:
                yield (right, y)


@dataclass(frozen=True)
class _Crossing:
    """The cheapest way from an edge cell of the window into one leaf outside it.

    Attributes:
        step: The cell just outside the window that the edge cell hops into.
        cost: The hop into ``step``, plus the way from ``step`` to the leaf's centre.
    """

    step: CellId
    cost: float


class MixedGraph:
    """The graph the search runs on: cells inside the window, the quadtree's leaves outside.

    Nodes of the two kinds meet at the window's border, where an edge cell is linked to every
    leaf just outside it that it can hop into.

    A graph describes the quadtree as it was at ``version``, and is indexed for the search
    once, on the first field solved on it, so a robot that stays in its window can reuse the
    graph for every field it builds there until its terrain changes; ``is_current_for`` says
    whether it still can.

    Attributes:
        quadtree: The terrain the graph was built from.
        window: The cells searched one by one.
        version: The quadtree's version when the graph was built.
    """

    def __init__(self, quadtree: Quadtree, window: BoundingBox) -> None:
        self.quadtree: Quadtree = quadtree
        self.window: BoundingBox = window
        self.version: int = quadtree.version
        self._crossings: dict[CellId, dict[Leaf, _Crossing]] = {}
        self._cells_bordering: dict[Leaf, list[tuple[CellId, float]]] = {}
        self._indexed: IndexedGraph[Node] | None = None
        self._link_border()

    @classmethod
    def around(cls, quadtree: Quadtree, robot: CellId, window_blocks: int) -> MixedGraph:
        """The graph with a window of the robot's block and the blocks within ``window_blocks``."""
        return cls(quadtree, _window_around(robot, quadtree, window_blocks))

    def is_current_for(self, quadtree: Quadtree, robot: CellId, window_blocks: int) -> bool:
        """Whether ``around(quadtree, robot, window_blocks)`` would build this same graph."""
        return (
            self.quadtree is quadtree
            and self.version == quadtree.version
            and self.window == _window_around(robot, quadtree, window_blocks)
        )

    def node_at(self, cell: CellId) -> Node | None:
        """The node containing ``cell``, or ``None`` if a robot can't be there."""
        if not self.quadtree.is_traversable(cell):
            return None
        if self.window.contains(cell):
            return cell
        return self.quadtree.leaf_at(cell)

    def indexed(self) -> IndexedGraph[Node]:
        """The graph indexed for the search, built on the first call and kept."""
        if self._indexed is None:
            self._indexed = self._index_hops()
        return self._indexed

    def _index_hops(self) -> IndexedGraph[Node]:
        """Index every node and hop: the traversable cells of the window, then the leaves.

        The hops between cells of the window, nearly all of them, are found with numpy, one
        shifted copy of the window per hop direction; the hops to and from leaves one by one.
        """
        window = self.window
        inside = self.quadtree.traversable[window.x_min : window.x_max, window.y_min : window.y_max]
        xs, ys = np.nonzero(inside)
        cell_index = np.full(inside.shape, -1, dtype=np.intp)
        cell_index[xs, ys] = np.arange(len(xs))
        nodes: list[Node] = []
        for x, y in zip((xs + window.x_min).tolist(), (ys + window.y_min).tolist(), strict=True):
            nodes.append((x, y))
        first_leaf = len(nodes)
        for leaf in self.quadtree.leaves():
            if not window.contains((leaf.x, leaf.y)):
                nodes.append(leaf)

        starts: list[NDArray[np.intp]] = []
        ends: list[NDArray[np.intp]] = []
        costs: list[NDArray[np.float64]] = []
        for dx, dy in self.quadtree.hop_offsets:
            from_x, to_x = _shifted_ranges(inside.shape[0], dx)
            from_y, to_y = _shifted_ranges(inside.shape[1], dy)
            allowed = inside[from_x, from_y] & inside[to_x, to_y]
            cost = 1.0
            if dx != 0 and dy != 0:
                # A diagonal hop needs both cells beside it, as in ``Quadtree.cell_neighbours``.
                allowed &= inside[to_x, from_y] & inside[from_x, to_y]
                cost = _DIAGONAL_LENGTH
            starts.append(cell_index[from_x, from_y][allowed])
            ends.append(cell_index[to_x, to_y][allowed])
            costs.append(np.full(int(allowed.sum()), cost))

        leaf_numbers: dict[Node, int] = {}
        for number, leaf in enumerate(nodes[first_leaf:], start=first_leaf):
            leaf_numbers[leaf] = number

        def number_of(node: Node) -> int:
            if isinstance(node, Leaf):
                return leaf_numbers[node]
            return int(cell_index[node[0] - window.x_min, node[1] - window.y_min])

        leaf_starts: list[int] = []
        leaf_ends: list[int] = []
        leaf_costs: list[float] = []
        for edge, crossings in self._crossings.items():
            for leaf, crossing in crossings.items():
                leaf_starts.append(number_of(edge))
                leaf_ends.append(number_of(leaf))
                leaf_costs.append(crossing.cost)
        for leaf in nodes[first_leaf:]:
            for neighbour, cost in self.neighbours(leaf):
                leaf_starts.append(number_of(leaf))
                leaf_ends.append(number_of(neighbour))
                leaf_costs.append(cost)
        starts.append(np.array(leaf_starts, dtype=np.intp))
        ends.append(np.array(leaf_ends, dtype=np.intp))
        costs.append(np.array(leaf_costs, dtype=np.float64))
        return IndexedGraph(
            nodes, np.concatenate(starts), np.concatenate(ends), np.concatenate(costs)
        )

    def neighbours(self, node: Node) -> Iterator[tuple[Node, float]]:
        """The nodes a robot may hop to from ``node``, with the cost of each hop."""
        if isinstance(node, Leaf):
            for leaf in self.quadtree.leaf_neighbours(node):
                # The window is made of whole blocks, so a leaf is inside it iff its corner is.
                if not self.window.contains((leaf.x, leaf.y)):
                    yield leaf, _octile_distance(node.centre, leaf.centre)
            yield from self._cells_bordering.get(node, [])
        else:
            for cell in self.quadtree.cell_neighbours(node):
                if self.window.contains(cell):
                    yield cell, _hop_cost(node, cell)
            for leaf, crossing in self._crossings.get(node, {}).items():
                yield leaf, crossing.cost

    def step_towards(self, cell: CellId, next_node: Node) -> CellId:
        """The cell to move to from ``cell`` when the route continues at ``next_node``."""
        if isinstance(next_node, Leaf):
            return self._crossings[cell][next_node].step
        return next_node

    def _link_border(self) -> None:
        """Link every edge cell of the window to the leaves just outside that it can hop into.

        An edge cell may reach the same leaf through several of its outside neighbours; the
        cheapest crossing is kept. The links go both ways, with the same cost.
        """
        for edge in self.window.edge_cells():
            if not self.quadtree.is_traversable(edge):
                continue
            for outside in self.quadtree.cell_neighbours(edge):
                if self.window.contains(outside):
                    continue
                leaf = self.quadtree.leaf_at(outside)
                if leaf is None:
                    continue
                cost = _hop_cost(edge, outside) + _octile_distance(_centre_of(outside), leaf.centre)
                crossings = self._crossings.setdefault(edge, {})
                if leaf not in crossings or cost < crossings[leaf].cost:
                    crossings[leaf] = _Crossing(step=outside, cost=cost)
        for edge, crossings in self._crossings.items():
            for leaf, crossing in crossings.items():
                self._cells_bordering.setdefault(leaf, []).append((edge, crossing.cost))


class GridPotentialField:
    """A solved grid potential field over a ``MixedGraph``.

    Inside the window every cell has its own value. Outside it, a cell has the value of its
    leaf, which is measured from the leaf's centre.

    Attributes:
        graph: The graph the field was solved on.
        settled: The result per reached node.
    """

    def __init__(self, graph: MixedGraph, settled: SearchResult[Node]) -> None:
        self.graph: MixedGraph = graph
        self.settled: SearchResult[Node] = settled

    def value_at(self, cell: CellId) -> float:
        """The potential at ``cell``; ``inf`` if no target is reachable from it."""
        result = self._result_at(cell)
        if result is None:
            return inf
        return result.value

    def next_cell(self, cell: CellId) -> CellId | None:
        """The cell to move to from ``cell``, which must lie inside the window.

        ``None`` means the robot stands on the winning target, or no target is reachable.
        """
        if not self.graph.window.contains(cell):
            return None
        result = self.settled.get(cell)
        if result is None or result.parent is None:
            return None
        return self.graph.step_towards(cell, result.parent)

    def target_for(self, cell: CellId) -> CellId | None:
        """The target whose wave won at ``cell``, even when that target is far away."""
        result = self._result_at(cell)
        if result is None:
            return None
        return result.origin

    def _result_at(self, cell: CellId) -> SettledNode[Node] | None:
        node = self.graph.node_at(cell)
        if node is None:
            return None
        return self.settled.get(node)


def build_grid_potential_field(
    seeds: tuple[Seed, ...],
    quadtree: Quadtree,
    robot: CellId,
    window_blocks: int = 2,
) -> GridPotentialField:
    """Solve the field with one multi-source Dijkstra from the targets.

    The window of single cells covers the robot's block of ``quadtree.max_leaf_side`` cells
    and the blocks within ``window_blocks`` of it in each direction.
    """
    return solve_grid_potential_field(seeds, MixedGraph.around(quadtree, robot, window_blocks))


def solve_grid_potential_field(seeds: tuple[Seed, ...], graph: MixedGraph) -> GridPotentialField:
    """Solve the field on an existing graph, e.g. one kept from an earlier build."""
    node_seeds: dict[Node, SettledNode[Node]] = {}
    for seed in seeds:
        node = graph.node_at(seed.cell)
        if node is None:
            continue
        value = seed.value
        if isinstance(node, Leaf):
            # Leaf values are measured from the leaf's centre, so a leaf starts with the
            # target's value plus the way from its centre to the target.
            value += _octile_distance(node.centre, _centre_of(seed.cell))

        # A node containing several targets starts from the most attractive (lowest) one.
        candidate = SettledNode(value=value, origin=seed.cell, parent=None)
        stored = node_seeds.get(node)
        if stored is None or candidate.value < stored.value:
            node_seeds[node] = candidate

    return GridPotentialField(graph, graph.indexed().search(node_seeds))


def _window_around(robot: CellId, quadtree: Quadtree, window_blocks: int) -> BoundingBox:
    """The whole blocks within ``window_blocks`` of the robot's block, clipped to the field."""
    side = quadtree.max_leaf_side
    block_x = robot[0] // side
    block_y = robot[1] // side
    return BoundingBox(
        x_min=max(0, (block_x - window_blocks) * side),
        y_min=max(0, (block_y - window_blocks) * side),
        x_max=min(quadtree.width, (block_x + window_blocks + 1) * side),
        y_max=min(quadtree.height, (block_y + window_blocks + 1) * side),
    )


def _shifted_ranges(length: int, shift: int) -> tuple[slice, slice]:
    """Along an axis of ``length`` cells, the cells a hop of ``shift`` can start from, and the
    cells it then ends in, as slices of the same length."""
    return slice(max(0, -shift), length - max(0, shift)), slice(
        max(0, shift), length + min(0, shift)
    )


def _hop_cost(start: CellId, end: CellId) -> float:
    """The cost of a hop between two neighbouring cells: 1 straight, 1.4 diagonally."""
    if start[0] != end[0] and start[1] != end[1]:
        return _DIAGONAL_LENGTH
    return 1.0


def _centre_of(cell: CellId) -> tuple[float, float]:
    return (cell[0] + 0.5, cell[1] + 0.5)


def _octile_distance(start: tuple[float, float], end: tuple[float, float]) -> float:
    """The length of the shortest 8-neighbour route between two points.

    Go diagonally until one of the two distances is used up, then straight for the rest::

        route = min(dx, dy) diagonals + (max(dx, dy) - min(dx, dy)) straights
              = max(dx, dy) + (DIAGONAL_LENGTH - 1) * min(dx, dy)
    """
    dx = abs(start[0] - end[0])
    dy = abs(start[1] - end[1])
    return max(dx, dy) + (_DIAGONAL_LENGTH - 1) * min(dx, dy)
