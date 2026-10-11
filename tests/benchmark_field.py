"""Benchmark: the quadtree potential field against a plain Dijkstra over every cell.

Run it with ``uv run python -m tests.benchmark_field``, optionally with ``--sizes 100 250 500``,
``--repeats 3`` or ``--seed 0``. It is not a pytest module: the plain Dijkstra over a 500 x 500
field takes seconds per build, too slow for the test suite.

Every number comes from the code the robots run. The terrain is a robot's ``AgentMemory``, the
quadtree is built from it as the ``Navigator`` does, and fields come from
``build_grid_potential_field``. The *uniform grid* is the same builder with a window of single
cells that covers the whole field, so its search runs over every cell and its values are exact.
The *quadtree* is the builder with its default window. The trip is a ``Navigator`` walking a
robot from its start to the target, along fields from a ``FieldProvider`` that rebuilds them
whenever its cache says so. The two-field trip walks the same route while a second seeder's
field is also read at every step, as a behaviour that bids and loses would read it to score.

The field: free ground split by a river along x = size / 2 with one bridge near the top, plus
one scattered obstacle per 500 cells. The robot starts low on the west bank and the target lies
low on the east bank, so the route has to detour through the bridge.
"""

import argparse
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter

import numpy as np

from swarm_field.contracts import CellId, FieldGeometry, Idle, Terrain, TerrainObservation, Time
from swarm_field.swarm import AgentMemory, FieldProvider, MemoryView, Navigator
from swarm_field.swarm.field import Quadtree, Seed, build_grid_potential_field, traversable_cells

_WHOLE_FIELD_BLOCKS = 1_000
"""A window so large that it covers the whole field: a plain Dijkstra over all cells."""

_CELLS_PER_OBSTACLE = 500
"""One scattered obstacle per this many cells of field."""


@dataclass(frozen=True)
class FieldTiming:
    """The cost of one field build.

    Attributes:
        seconds: The fastest of the repeated builds.
        nodes: The nodes the search settled: cells, plus leaves outside the window.
    """

    seconds: float
    nodes: int


@dataclass(frozen=True)
class TripResult:
    """A robot walked by the ``Navigator`` from its start to the target.

    Attributes:
        arrived: Whether the robot ended on the target.
        builds: How many fields were built on the way for the followed seeder.
        second_builds: How many were built for the second seeder, read at every step but
            never followed; 0 on a trip without one.
        seconds: The wall time of the whole trip, almost all of it spent building fields.
        route_length: The length of the route walked, in cells.
    """

    arrived: bool
    builds: int
    second_builds: int
    seconds: float
    route_length: float


@dataclass(frozen=True)
class SizeResult:
    """Everything measured on one field size.

    Attributes:
        size: The side of the square field, in cells.
        leaves: The traversable leaves of the quadtree.
        quadtree_seconds: The fastest quadtree build from the traversability array.
        uniform: One field build over every cell.
        quadtree: One field build over the quadtree, with the default window.
        trip: The robot's trip along quadtree fields.
        trip_two_fields: The same trip with a second seeder's field read at every step.
        shortest_route: The exact shortest route from start to target, in cells.
    """

    size: int
    leaves: int
    quadtree_seconds: float
    uniform: FieldTiming
    quadtree: FieldTiming
    trip: TripResult
    trip_two_fields: TripResult
    shortest_route: float

    @property
    def route_excess(self) -> float:
        """How much longer the trip's route is than the shortest one, as a fraction."""
        return self.trip.route_length / self.shortest_route - 1


class FixedTargetSeeder:
    """A seeder with one target that never changes, counting the fields built for it."""

    def __init__(self, target: CellId, name: str = "benchmark") -> None:
        self.name: str = name
        self.window_blocks: int = 2
        self.builds: int = 0
        self._target: CellId = target

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        return None

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        self.builds += 1
        return (Seed(self._target, 0.0),)


def build_river_field_memory(size: int, start: CellId, target: CellId, seed: int) -> AgentMemory:
    """A robot's memory of a square field with a river, one bridge and scattered obstacles."""
    geometry = FieldGeometry(
        width=size, height=size, cell_size=1.0, row_sp=0.0, in_row_sp=0.0, headland=0.0
    )
    river_x = size // 2
    bridge_y = size * 7 // 8
    obstacles = {(river_x, y) for y in range(size) if y != bridge_y}

    rng = np.random.default_rng(seed)
    kept_free = {
        start,
        target,
        (river_x - 1, bridge_y),
        (river_x, bridge_y),
        (river_x + 1, bridge_y),
    }
    for x, y in rng.integers(0, size, size=(size * size // _CELLS_PER_OBSTACLE, 2)).tolist():
        if (x, y) not in kept_free:
            obstacles.add((x, y))

    memory = AgentMemory(1, geometry)
    facts: list[TerrainObservation] = []
    for cell in sorted(obstacles):
        facts.append(
            TerrainObservation(
                cell=cell, confidence=1.0, observed_at=0, source=1, terrain=Terrain.OBSTACLE
            )
        )
    memory.merge(facts)
    return memory


def time_fastest[T](build: Callable[[], T], repeats: int) -> tuple[float, T]:
    """The fastest wall time of ``repeats`` calls to ``build``, with the last result."""
    fastest = float("inf")
    result = build()
    for _ in range(repeats):
        started = perf_counter()
        result = build()
        fastest = min(fastest, perf_counter() - started)
    return fastest, result


def measure_field_build(
    quadtree: Quadtree, target: CellId, robot: CellId, window_blocks: int, repeats: int
) -> FieldTiming:
    """Time one field build towards ``target`` with the robot at ``robot``."""
    seeds = (Seed(target, 0.0),)
    seconds, field = time_fastest(
        lambda: build_grid_potential_field(seeds, quadtree, robot, window_blocks), repeats
    )
    return FieldTiming(seconds=seconds, nodes=len(field.settled))


def walk_navigator_trip(
    memory: AgentMemory, start: CellId, target: CellId, second_target: CellId | None = None
) -> TripResult:
    """Walk a robot from ``start`` along the navigator's fields until it stops.

    With ``second_target``, a second seeder towards it is registered too, and its field is
    read at every step without ever being followed.
    """
    seeder = FixedTargetSeeder(target)
    seeders = [seeder]
    second: FixedTargetSeeder | None = None
    if second_target is not None:
        second = FixedTargetSeeder(second_target, name="second")
        seeders.append(second)
    fields = FieldProvider(memory, flies=False, seeders=seeders)
    navigator = Navigator(fields.quadtree)
    route = [start]
    started = perf_counter()
    for _ in range(10 * memory.width * memory.height):
        if second is not None:
            fields.field_for(second.name, route[-1], 0)
        seeded = fields.field_for(seeder.name, route[-1], 0)
        if seeded is None:
            break
        action = navigator.step_towards(seeded, route[-1])
        if isinstance(action, Idle):
            break
        route.append(action.to)
    second_builds = 0
    if second is not None:
        second_builds = second.builds
    return TripResult(
        arrived=route[-1] == target,
        builds=seeder.builds,
        second_builds=second_builds,
        seconds=perf_counter() - started,
        route_length=measure_route_length(route),
    )


def measure_route_length(route: list[CellId]) -> float:
    """The length of a route of neighbouring cells: 1 per straight hop, 1.4 per diagonal one."""
    length = 0.0
    for a, b in pairwise(route):
        if a[0] != b[0] and a[1] != b[1]:
            length += 1.4
        else:
            length += 1.0
    return length


def measure_size(size: int, repeats: int, seed: int) -> SizeResult:
    """Run every measurement on a square field of ``size`` cells per side."""
    start = (size // 8, size // 8)
    target = (size - 1 - size // 8, size // 8)
    second_target = (size // 8, size - 1 - size // 8)
    memory = build_river_field_memory(size, start, target, seed)
    traversable = traversable_cells(memory, flies=False)
    quadtree_seconds, quadtree = time_fastest(lambda: Quadtree(traversable), repeats)

    uniform = measure_field_build(quadtree, target, start, _WHOLE_FIELD_BLOCKS, repeats)
    shortest_route = build_grid_potential_field(
        (Seed(target, 0.0),), quadtree, start, _WHOLE_FIELD_BLOCKS
    ).value_at(start)
    return SizeResult(
        size=size,
        leaves=sum(1 for _ in quadtree.leaves()),
        quadtree_seconds=quadtree_seconds,
        uniform=uniform,
        quadtree=measure_field_build(quadtree, target, start, 2, repeats),
        trip=walk_navigator_trip(memory, start, target),
        trip_two_fields=walk_navigator_trip(memory, start, target, second_target),
        shortest_route=shortest_route,
    )


def format_results_table(results: list[SizeResult]) -> str:
    """The results as a plain-text table, one row per field size."""
    header = (
        f"{'field':>9} {'cells':>8} {'leaves':>7} {'tree ms':>8} "
        f"{'uniform s':>10} {'nodes':>8} {'quadtree s':>11} {'nodes':>7} {'speed-up':>9} "
        f"{'trip builds':>12} {'trip s':>7} {'arrived':>8} {'route +%':>9} "
        f"{'2-field builds':>15} {'2-field s':>10}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        two = result.trip_two_fields
        lines.append(
            f"{f'{result.size}x{result.size}':>9} {result.size * result.size:>8} "
            f"{result.leaves:>7} {result.quadtree_seconds * 1000:>8.2f} "
            f"{result.uniform.seconds:>10.3f} {result.uniform.nodes:>8} "
            f"{result.quadtree.seconds:>11.3f} {result.quadtree.nodes:>7} "
            f"{result.uniform.seconds / result.quadtree.seconds:>8.1f}x "
            f"{result.trip.builds:>12} {result.trip.seconds:>7.2f} {result.trip.arrived!s:>8} "
            f"{result.route_excess * 100:>9.2f} "
            f"{f'{two.builds} + {two.second_builds}':>15} {two.seconds:>10.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 250, 500])
    parser.add_argument("--repeats", type=int, default=3, help="builds timed per measurement")
    parser.add_argument("--seed", type=int, default=0, help="seed of the scattered obstacles")
    arguments = parser.parse_args()

    results: list[SizeResult] = []
    for size in arguments.sizes:
        print(f"measuring {size} x {size} ...", flush=True)
        results.append(measure_size(size, arguments.repeats, arguments.seed))
    print()
    print(format_results_table(results))
    print()
    print("uniform s / quadtree s: one field build, the fastest of the repeats.")
    print("trip: the Navigator walks the robot to the target, rebuilding when its cache misses;")
    print("a uniform field is exact everywhere, so it would need one build for the whole trip.")
    print("2-field: the same trip while a second seeder's field is read at every step, as a")
    print("losing behaviour would to score; builds are followed + second.")
    print("route +%: how much longer the trip's route is than the exact shortest route.")


if __name__ == "__main__":
    main()
