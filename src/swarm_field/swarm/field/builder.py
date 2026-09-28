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

from collections.abc import Iterator
from dataclasses import dataclass
from math import inf

from swarm_field.contracts import CellId
from swarm_field.swarm.field.dijkstra import SettledNode, multi_source_dijkstra
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
    """

    def __init__(self, quadtree: Quadtree, window: BoundingBox) -> None:
        self.quadtree: Quadtree = quadtree
        self.window: BoundingBox = window
        self._crossings: dict[CellId, dict[Leaf, _Crossing]] = {}
        self._cells_bordering: dict[Leaf, list[tuple[CellId, float]]] = {}
        self._link_border()

    def node_at(self, cell: CellId) -> Node | None:
        """The node containing ``cell``, or ``None`` if a robot can't be there."""
        if not self.quadtree.is_traversable(cell):
            return None
        if self.window.contains(cell):
            return cell
        return self.quadtree.leaf_at(cell)

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

    def __init__(self, graph: MixedGraph, settled: dict[Node, SettledNode[Node]]) -> None:
        self.graph: MixedGraph = graph
        self.settled: dict[Node, SettledNode[Node]] = settled

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
    graph = MixedGraph(quadtree, _window_around(robot, quadtree, window_blocks))
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

    # Hops and their costs are symmetric, so the hops into a node are the hops out of it.
    settled = multi_source_dijkstra(node_seeds, graph.neighbours)
    return GridPotentialField(graph, settled)


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
