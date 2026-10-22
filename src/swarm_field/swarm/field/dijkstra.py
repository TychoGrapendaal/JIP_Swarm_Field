"""Multi-source Dijkstra: spread waves from several seeds and record which wave won where.

The graph is indexed once into a sparse matrix (``IndexedGraph``). Each search is then one run
of scipy's compiled Dijkstra from a virtual source linked to every seed, with the seed's start
value as the cost of its link, so a graph searched from many sets of seeds is indexed only once.
"""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra

from swarm_field.contracts import CellId


@dataclass(frozen=True)
class SettledNode[N: Hashable]:
    """The result of the search at one node.

    Attributes:
        value: The cheapest start value plus route cost to this node, in cells.
        origin: The target cell whose wave won here.
        parent: The next node towards the origin, or ``None`` on a seed that nothing improved.
    """

    value: float
    origin: CellId
    parent: N | None


class IndexedGraph[N: Hashable]:
    """A graph of hashable nodes, held as a sparse matrix to search from many sets of seeds.

    Attributes:
        nodes: Every node, at its index in the matrix.
    """

    def __init__(
        self,
        nodes: list[N],
        starts: NDArray[np.intp],
        ends: NDArray[np.intp],
        costs: NDArray[np.float64],
    ) -> None:
        """Index ``nodes`` and the hops between them.

        Hop ``i`` goes from node ``starts[i]`` into node ``ends[i]``, both indices into
        ``nodes``, and costs ``costs[i]``, which must be non-negative.
        """
        self.nodes: list[N] = nodes
        self._index: dict[N, int] = {node: index for index, node in enumerate(nodes)}
        # Row ``node`` of the matrix holds the hops *into* it, since the waves spread from the
        # seeds against the hops.
        order = np.argsort(ends, kind="stable")
        self._neighbours: NDArray[np.int32] = starts[order].astype(np.int32)
        self._costs: NDArray[np.float64] = costs[order]
        hops_into = np.bincount(ends, minlength=len(nodes))
        self._row_starts: NDArray[np.int32] = np.concatenate(([0], np.cumsum(hops_into))).astype(
            np.int32
        )

    def index_of(self, node: N) -> int | None:
        """The node's index in the matrix, or ``None`` if it is not part of the graph."""
        return self._index.get(node)

    def search(self, seeds: Mapping[N, SettledNode[N]]) -> "SearchResult[N]":
        """Settle every node reachable from the seeds, cheapest first.

        Each seed starts with its own ``value``, which may be negative, and its ``origin``;
        its ``parent`` is ignored. Nodes that no wave reaches are absent from the result.
        """
        count = len(self.nodes)
        if not seeds:
            return SearchResult(self, _unreached(count))
        seed_indices = np.array([self._index[node] for node in seeds], dtype=np.int32)
        start_values = np.array([seed.value for seed in seeds.values()], dtype=np.float64)
        # The virtual source's links must not be negative, so every start value is shifted up
        # by the lowest one, and the distances back down by as much.
        lowest = float(start_values.min())
        source = count
        matrix = csr_array(
            (
                np.concatenate([self._costs, start_values - lowest]),
                np.concatenate([self._neighbours, seed_indices]),
                np.append(self._row_starts, self._row_starts[-1] + len(seed_indices)),
            ),
            shape=(count + 1, count + 1),
        )
        distances, predecessors = dijkstra(matrix, indices=source, return_predecessors=True)
        reached = np.isfinite(distances[:count])
        predecessors = predecessors[:count].astype(np.intp)
        is_root = (predecessors == source) | ~reached
        parents = np.where(is_root, -1, predecessors)

        # Each node's winning seed is the root of its chain of parents, found by pointer
        # jumping: every pass doubles the steps each node has looked up the chain.
        roots = np.where(is_root, np.arange(count), predecessors)
        while True:
            jumped = roots[roots]
            if np.array_equal(jumped, roots):
                break
            roots = jumped
        origins: dict[int, CellId] = {}
        for index, seed in zip(seed_indices.tolist(), seeds.values(), strict=True):
            origins[index] = seed.origin
        found = _Settled(
            values=distances[:count] + lowest,
            reached=reached,
            parents=parents,
            roots=roots,
            origins=origins,
        )
        return SearchResult(self, found)


@dataclass(frozen=True)
class _Settled:
    """The arrays one search produced, over the nodes of the graph.

    Attributes:
        values: Each node's value, ``inf`` where no wave reached it.
        reached: Whether a wave reached each node.
        parents: The index of each node's parent, -1 on a seed that nothing improved.
        roots: The index of the seed whose wave won at each node.
        origins: The origin of each seed, by the seed's index.
    """

    values: NDArray[np.float64]
    reached: NDArray[np.bool_]
    parents: NDArray[np.intp]
    roots: NDArray[np.intp]
    origins: dict[int, CellId]


def _unreached(count: int) -> _Settled:
    """The result of a search without seeds: no node reached."""
    return _Settled(
        values=np.full(count, np.inf),
        reached=np.zeros(count, dtype=np.bool_),
        parents=np.full(count, -1, dtype=np.intp),
        roots=np.arange(count),
        origins={},
    )


class SearchResult[N: Hashable]:
    """The settled nodes of one search on an ``IndexedGraph``, read one node at a time."""

    def __init__(self, graph: IndexedGraph[N], settled: _Settled) -> None:
        self._graph: IndexedGraph[N] = graph
        self._settled: _Settled = settled

    def __len__(self) -> int:
        """How many nodes a wave reached."""
        return int(self._settled.reached.sum())

    def get(self, node: N) -> SettledNode[N] | None:
        """The result at ``node``, or ``None`` if no wave reached it."""
        index = self._graph.index_of(node)
        settled = self._settled
        if index is None or not settled.reached[index]:
            return None
        parent: N | None = None
        parent_index = int(settled.parents[index])
        if parent_index >= 0:
            parent = self._graph.nodes[parent_index]
        return SettledNode(
            value=float(settled.values[index]),
            origin=settled.origins[int(settled.roots[index])],
            parent=parent,
        )
