"""Tests for the navigator: the one movement seam, its field cache, and the steering fallback."""

from collections.abc import Hashable

from swarm_field.contracts import (
    CellId,
    Direction,
    FieldGeometry,
    Idle,
    Move,
    Terrain,
    TerrainObservation,
    Time,
)
from swarm_field.swarm import AgentMemory, MemoryView, Navigator, QuadtreeSensing
from swarm_field.swarm.field import Seed

_FIELD = FieldGeometry(
    width=40, height=40, cell_size=1.0, row_sp=0.0, in_row_sp=0.0, headland=0.0, connectivity=8
)


class _FixedSeeder:
    """A seeder with fixed seeds and a settable key, counting how often it is asked for seeds."""

    name: str = "fixed"

    def __init__(self, seeds: tuple[Seed, ...]) -> None:
        self.key: int = 0
        self.seeds_asked: int = 0
        self._seeds: tuple[Seed, ...] = seeds

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        return self.key

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        self.seeds_asked += 1
        return self._seeds


def _navigator(obstacles: set[CellId], seeder: _FixedSeeder | None = None) -> Navigator:
    """A navigator over a 40-by-40 field that knows the given cells as obstacles."""
    memory = AgentMemory(1, _FIELD)
    facts = [
        TerrainObservation(
            cell=cell, confidence=1.0, observed_at=0, source=1, terrain=Terrain.OBSTACLE
        )
        for cell in sorted(obstacles)
    ]
    memory.merge(facts)
    seeders: list[_FixedSeeder] = []
    if seeder is not None:
        seeders.append(seeder)
    return Navigator(memory, flies=False, seeders=seeders)


def _step(navigator: Navigator, position: CellId) -> Move | Idle:
    """One decision point: fetch the fixed seeder's field and step along it."""
    seeded = navigator.field_for(_FixedSeeder.name, position, 0)
    assert seeded is not None
    return navigator.step_towards(seeded, position)


def test_the_field_walks_the_robot_around_a_wall_to_its_target() -> None:
    """A field guides the robot the whole way, at a distance, around what it knows."""
    wall = {(20, y) for y in range(18, 24)}
    navigator = _navigator(wall, _FixedSeeder((Seed((30, 21), 0.0),)))
    position: CellId = (10, 21)
    visited = [position]

    for _ in range(200):
        action = _step(navigator, position)
        if isinstance(action, Idle):
            break
        position = action.to
        visited.append(position)

    assert position == (30, 21)
    assert not wall.intersection(visited[1:])


def test_the_field_is_reused_until_the_key_the_block_or_the_terrain_changes() -> None:
    """A field is kept for as long as it still answers for the same plan."""
    seeder = _FixedSeeder((Seed((30, 30), 0.0),))
    navigator = _navigator(set(), seeder)

    in_first_block = navigator.field_for(seeder.name, (0, 0), 0)
    assert navigator.field_for(seeder.name, (3, 3), 1_000) is in_first_block
    assert seeder.seeds_asked == 1

    in_next_block = navigator.field_for(seeder.name, (17, 0), 0)
    assert in_next_block is not in_first_block
    assert navigator.field_for(seeder.name, (17, 0), 0) is in_next_block

    navigator.update_terrain([(1, 1)])
    after_terrain = navigator.field_for(seeder.name, (17, 0), 0)
    assert after_terrain is not in_next_block

    seeder.key = 1
    assert navigator.field_for(seeder.name, (17, 0), 0) is not after_terrain
    assert seeder.seeds_asked == 4


def test_a_walled_off_target_hands_over_to_the_steerer_and_then_reports() -> None:
    """No route at all: the steerer tries its own way, and the stall is the brain's cue."""
    wall = {(20, y) for y in range(40)}
    navigator = _navigator(wall, _FixedSeeder((Seed((30, 20), 0.0),)))
    position: CellId = (10, 20)
    actions: list[Move | Idle] = []

    for _ in range(200):
        action = _step(navigator, position)
        actions.append(action)
        if isinstance(action, Idle):
            break
        position = action.to

    assert any(isinstance(action, Move) for action in actions)
    assert not {action.to for action in actions if isinstance(action, Move)} & wall
    assert isinstance(actions[-1], Idle)


def test_no_seeds_means_no_field() -> None:
    navigator = _navigator(set(), _FixedSeeder(()))

    assert navigator.field_for(_FixedSeeder.name, (0, 0), 0) is None


def test_the_map_backed_sensing_never_cuts_a_blocked_corner() -> None:
    """The steerer must not read as free a diagonal the quadtree's own routing refuses."""
    both_corners_blocked = QuadtreeSensing(_navigator({(11, 0), (10, 1)}).quadtree)
    assert not both_corners_blocked.clear((10, 0), Direction.NE)
    assert not both_corners_blocked.clear((10, 0), Direction.E)
    assert not both_corners_blocked.clear((10, 0), Direction.N)

    one_corner_blocked = QuadtreeSensing(_navigator({(11, 0)}).quadtree)
    assert not one_corner_blocked.clear((10, 0), Direction.NE)
    assert one_corner_blocked.clear((10, 0), Direction.N)

    both_corners_free = QuadtreeSensing(_navigator({(12, 0)}).quadtree)
    assert both_corners_free.clear((10, 0), Direction.NE)
