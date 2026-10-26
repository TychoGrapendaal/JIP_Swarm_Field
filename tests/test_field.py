"""Tests for the grid potential field, using the worked examples of the field-level design."""

from itertools import pairwise
from math import inf, isclose

from swarm_field.contracts import CellId, FieldGeometry, Terrain, TerrainObservation
from swarm_field.swarm import AgentMemory
from swarm_field.swarm.field import (
    GridPotentialField,
    Quadtree,
    Seed,
    TerrainView,
    build_grid_potential_field,
    traversable_cells,
)

_WHOLE_FIELD = 1_000
"""A window so large that it covers the whole field: a plain Dijkstra over all cells."""


class DrawnTerrain:
    """Terrain drawn as text rows, top row first: ``.`` free, ``#`` obstacle, ``?`` unknown."""

    def __init__(self, rows: list[str]) -> None:
        self._rows: list[str] = [row.replace(" ", "") for row in rows]
        self.width: int = len(self._rows[0])
        self.height: int = len(self._rows)

    def terrain_at(self, cell: CellId) -> Terrain | None:
        x, y = cell
        match self._rows[self.height - 1 - y][x]:
            case "#":
                return Terrain.OBSTACLE
            case "?":
                return None
            case _:
                return Terrain.FREE


class OpenField:
    """A free field of the given size, except for the listed obstacle cells."""

    def __init__(self, width: int, height: int, obstacles: set[CellId]) -> None:
        self.width: int = width
        self.height: int = height
        self.obstacles: set[CellId] = obstacles

    def terrain_at(self, cell: CellId) -> Terrain | None:
        if cell in self.obstacles:
            return Terrain.OBSTACLE
        return Terrain.FREE


def _walk(
    seeds: tuple[Seed, ...], quadtree: Quadtree, start: CellId, window_blocks: int
) -> list[CellId]:
    """Follow the field from ``start``, rebuilding it around the robot after every step."""
    path = [start]
    while len(path) <= 10_000:
        field = build_grid_potential_field(seeds, quadtree, path[-1], window_blocks)
        step = field.next_cell(path[-1])
        if step is None:
            return path
        path.append(step)
    raise AssertionError("the robot never arrived")


def _route_cost(path: list[CellId]) -> float:
    cost = 0.0
    for a, b in pairwise(path):
        if a[0] != b[0] and a[1] != b[1]:
            cost += 1.4
        else:
            cost += 1.0
    return cost


def test_treat_example_prefers_the_fresh_far_pest() -> None:
    """§4 of the field-level design: K drives through the gap to P1, not to the nearer P2."""
    terrain = DrawnTerrain(
        [
            ". . . . . # . . ? ?",
            ". . . . . # . . ? ?",
            ". . . . . # . . . .",
            ". . . . . # . . . .",
            ". . . . . # . . . .",
            ". . . . . . . . . .",
        ]
    )
    p1, p2, k = (8, 1), (2, 4), (1, 1)
    seeds = (Seed(p1, -8.28), Seed(p2, -1.74))
    quadtree = Quadtree(traversable_cells(terrain, flies=False))

    field = build_grid_potential_field(seeds, quadtree, k, _WHOLE_FIELD)
    path = _walk(seeds, quadtree, k, _WHOLE_FIELD)

    assert isclose(field.value_at(k), -0.48)
    assert field.target_for(k) == p1
    assert path[-1] == p1
    assert (5, 0) in path
    assert isclose(_route_cost(path), 7.8)


def test_wavefront_flows_around_a_cup_and_never_enters_it() -> None:
    """§3.1.2 of the field-level design, with 4-neighbour steps."""
    terrain = DrawnTerrain(
        [
            ". . . . . . . . . . .",
            ". . . . # # # . . . .",
            ". . . . . . # . . . .",
            ". . . . . . # . . . .",
            ". . . . . . # . . . .",
            ". . . . # # # . . . .",
            ". . . . . . . . . . .",
        ]
    )
    goal = (9, 3)
    quadtree = Quadtree(traversable_cells(terrain, flies=False), connectivity=4)
    field = build_grid_potential_field((Seed(goal, 0.0),), quadtree, goal, _WHOLE_FIELD)
    cup = {(4, 2), (4, 3), (4, 4), (5, 2), (5, 3), (5, 4)}

    path = [(0, 3)]
    step = field.next_cell(path[-1])
    while step is not None:
        path.append(step)
        step = field.next_cell(step)

    assert [field.value_at((x, 3)) for x in range(6)] == [15, 14, 13, 12, 13, 14]
    assert path[-1] == goal
    assert len(path) - 1 == 15
    assert cup.isdisjoint(path)


def test_small_window_finds_a_near_optimal_route_on_an_open_field() -> None:
    quadtree = Quadtree(traversable_cells(OpenField(80, 80, set()), flies=False), max_leaf_side=8)
    seeds = (Seed((75, 62), 0.0),)

    windowed = _walk(seeds, quadtree, (2, 2), window_blocks=1)
    optimal = build_grid_potential_field(seeds, quadtree, (2, 2), _WHOLE_FIELD).value_at((2, 2))

    assert windowed[-1] == (75, 62)
    assert _route_cost(windowed) <= 1.05 * optimal


def test_field_leads_to_a_bridge_outside_the_window() -> None:
    river = {(20, y) for y in range(40) if y != 35}
    quadtree = Quadtree(traversable_cells(OpenField(40, 40, river), flies=False), max_leaf_side=8)
    seeds = (Seed((30, 5), 0.0),)
    start = (10, 5)

    field = build_grid_potential_field(seeds, quadtree, start, window_blocks=1)
    path = _walk(seeds, quadtree, start, window_blocks=1)

    assert not field.graph.window.contains((20, 35))
    assert field.target_for(start) == (30, 5)
    assert path[1][1] > start[1]
    assert (20, 35) in path
    assert path[-1] == (30, 5)


def test_no_reachable_target_means_infinite_potential() -> None:
    terrain = DrawnTerrain(
        [
            ". # .",
            ". # .",
        ]
    )
    quadtree = Quadtree(traversable_cells(terrain, flies=False))

    field = build_grid_potential_field((Seed((2, 0), 0.0),), quadtree, (0, 0))

    assert field.value_at((0, 0)) == inf
    assert field.next_cell((0, 0)) is None
    assert field.target_for((0, 0)) is None


def test_a_wall_on_the_edge_of_the_window_is_not_crossed() -> None:
    terrain = DrawnTerrain(
        [
            ". # . .",
            ". # . .",
        ]
    )
    quadtree = Quadtree(traversable_cells(terrain, flies=False), max_leaf_side=2)

    field = build_grid_potential_field((Seed((3, 0), 0.0),), quadtree, (0, 0), window_blocks=0)

    assert field.value_at((0, 0)) == inf


def test_drones_fly_over_obstacles() -> None:
    terrain = DrawnTerrain([". # ."])
    seeds = (Seed((2, 0), 0.0),)

    walking = build_grid_potential_field(
        seeds, Quadtree(traversable_cells(terrain, flies=False)), (0, 0)
    )
    flying = build_grid_potential_field(
        seeds, Quadtree(traversable_cells(terrain, flies=True)), (0, 0)
    )

    assert walking.value_at((0, 0)) == inf
    assert flying.value_at((0, 0)) == 2.0


def test_agent_memory_is_a_terrain_view() -> None:
    geometry = FieldGeometry(
        width=3, height=1, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0
    )
    memory = AgentMemory(1, geometry)
    memory.merge(
        [
            TerrainObservation(
                cell=(1, 0), confidence=1.0, observed_at=0, source=1, terrain=Terrain.OBSTACLE
            )
        ]
    )
    terrain: TerrainView = memory

    field: GridPotentialField = build_grid_potential_field(
        (Seed((2, 0), 0.0),), Quadtree(traversable_cells(terrain, flies=False)), (0, 0)
    )

    assert field.value_at((0, 0)) == inf
