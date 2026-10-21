"""Multi-source Dijkstra: spread waves from several seeds and record which wave won where."""

import heapq
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from itertools import count
from math import inf

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


def multi_source_dijkstra[N: Hashable](
    seeds: Mapping[N, SettledNode[N]],
    incoming_hops: Callable[[N], Iterable[tuple[N, float]]],
) -> dict[N, SettledNode[N]]:
    """Settle every node reachable from the seeds, cheapest first.

    Each seed starts with its own value, which may be negative. ``incoming_hops(node)`` yields
    the neighbours that can hop *into* ``node`` together with the cost of that hop, which must
    be non-negative. Nodes that no wave reaches are absent from the result.
    """
    # The frontier holds plain tuples (value, order, node, origin, parent) and a SettledNode is
    # made only once a node settles: most pushes are superseded, and this loop is the hot path.
    order = count()
    best: dict[N, float] = {}
    frontier: list[tuple[float, int, N, CellId, N | None]] = []
    for node, seed in seeds.items():
        best[node] = seed.value
        heapq.heappush(frontier, (seed.value, next(order), node, seed.origin, seed.parent))

    settled: dict[N, SettledNode[N]] = {}
    while frontier:
        value, _, node, origin, parent = heapq.heappop(frontier)
        if node in settled:
            continue
        settled[node] = SettledNode(value=value, origin=origin, parent=parent)
        for neighbour, cost in incoming_hops(node):
            candidate = value + cost
            # A settled neighbour is never improved: its best is at most ``value``.
            if candidate >= best.get(neighbour, inf):
                continue
            best[neighbour] = candidate
            heapq.heappush(frontier, (candidate, next(order), neighbour, origin, node))
    return settled
