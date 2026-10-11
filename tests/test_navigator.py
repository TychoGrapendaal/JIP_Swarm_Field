"""Tests for the navigator: the one movement seam, and its steering fallback."""

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
from swarm_field.swarm import AgentMemory, FieldProvider, MemoryView, Navigator, QuadtreeSensing
from swarm_field.swarm.field import Seed

_FIELD = FieldGeometry(
    width=40, height=40, cell_size=1.0, row_sp=0.0, in_row_sp=0.0, headland=0.0, connectivity=8
)


class _FixedSeeder:
    """A seeder with fixed seeds and a constant key."""

    name: str = "fixed"

    def __init__(self, seeds: tuple[Seed, ...]) -> None:
        self._seeds: tuple[Seed, ...] = seeds

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        return None

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        return self._seeds


def _fields(obstacles: set[CellId], seeds: tuple[Seed, ...] = ()) -> FieldProvider:
    """A field provider over a 40-by-40 field that knows the given cells as obstacles."""
    memory = AgentMemory(1, _FIELD)
    facts = [
        TerrainObservation(
            cell=cell, confidence=1.0, observed_at=0, source=1, terrain=Terrain.OBSTACLE
        )
        for cell in sorted(obstacles)
    ]
    memory.merge(facts)
    return FieldProvider(memory, flies=False, seeders=[_FixedSeeder(seeds)])


def _drive(fields: FieldProvider, start: CellId) -> list[Move | Idle]:
    """The navigator's actions along the fixed seeder's field, until it idles."""
    navigator = Navigator(fields.quadtree)
    position = start
    actions: list[Move | Idle] = []
    for _ in range(200):
        seeded = fields.field_for(_FixedSeeder.name, position, 0)
        assert seeded is not None
        action = navigator.step_towards(seeded, position)
        actions.append(action)
        if isinstance(action, Idle):
            break
        position = action.to
    return actions


def test_the_field_walks_the_robot_around_a_wall_to_its_target() -> None:
    """A field guides the robot the whole way, at a distance, around what it knows."""
    wall = {(20, y) for y in range(18, 24)}

    actions = _drive(_fields(wall, (Seed((30, 21), 0.0),)), (10, 21))

    visited = [action.to for action in actions if isinstance(action, Move)]
    assert visited[-1] == (30, 21)
    assert not wall.intersection(visited)


def test_a_walled_off_target_hands_over_to_the_steerer_and_then_reports() -> None:
    """No route at all: the steerer tries its own way, and the stall is the brain's cue."""
    wall = {(20, y) for y in range(40)}

    actions = _drive(_fields(wall, (Seed((30, 20), 0.0),)), (10, 20))

    assert any(isinstance(action, Move) for action in actions)
    assert not {action.to for action in actions if isinstance(action, Move)} & wall
    assert isinstance(actions[-1], Idle)


def test_the_map_backed_sensing_never_cuts_a_blocked_corner() -> None:
    """The steerer must not read as free a diagonal the quadtree's own routing refuses."""
    both_corners_blocked = QuadtreeSensing(_fields({(11, 0), (10, 1)}).quadtree)
    assert not both_corners_blocked.clear((10, 0), Direction.NE)
    assert not both_corners_blocked.clear((10, 0), Direction.E)
    assert not both_corners_blocked.clear((10, 0), Direction.N)

    one_corner_blocked = QuadtreeSensing(_fields({(11, 0)}).quadtree)
    assert not one_corner_blocked.clear((10, 0), Direction.NE)
    assert one_corner_blocked.clear((10, 0), Direction.N)

    both_corners_free = QuadtreeSensing(_fields({(12, 0)}).quadtree)
    assert both_corners_free.clear((10, 0), Direction.NE)
