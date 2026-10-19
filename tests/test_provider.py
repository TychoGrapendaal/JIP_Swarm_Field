"""Tests for the field provider: its quadtree, and one cached field per seeder."""

from collections.abc import Hashable

from swarm_field.contracts import (
    CellId,
    FieldGeometry,
    Terrain,
    TerrainObservation,
    Time,
)
from swarm_field.swarm import AgentMemory, FieldProvider, MemoryView
from swarm_field.swarm.field import Seed

_FIELD = FieldGeometry(
    width=40, height=40, cell_size=1.0, row_sp=0.0, in_row_sp=0.0, headland=0.0, connectivity=8
)


class _CountingSeeder:
    """A seeder with fixed seeds and a settable key, counting how often it is asked for seeds."""

    def __init__(self, name: str, seeds: tuple[Seed, ...]) -> None:
        self.name: str = name
        self.window_blocks: int = 2
        self.key: int = 0
        self.seeds_asked: int = 0
        self._seeds: tuple[Seed, ...] = seeds

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        return self.key

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        self.seeds_asked += 1
        return self._seeds


def _obstacle(cell: CellId) -> TerrainObservation:
    return TerrainObservation(
        cell=cell, confidence=1.0, observed_at=0, source=1, terrain=Terrain.OBSTACLE
    )


def test_the_field_is_reused_until_the_key_the_block_or_the_terrain_changes() -> None:
    """A field is kept for as long as it still answers for the same plan."""
    seeder = _CountingSeeder("fixed", (Seed((30, 30), 0.0),))
    memory = AgentMemory(1, _FIELD)
    fields = FieldProvider(memory, flies=False, seeders=[seeder])

    in_first_block = fields.field_for(seeder.name, (0, 0), 0)
    assert fields.field_for(seeder.name, (3, 3), 1_000) is in_first_block
    assert seeder.seeds_asked == 1

    in_next_block = fields.field_for(seeder.name, (17, 0), 0)
    assert in_next_block is not in_first_block
    assert fields.field_for(seeder.name, (17, 0), 0) is in_next_block

    fields.update_terrain([(1, 1)])
    assert fields.field_for(seeder.name, (17, 0), 0) is in_next_block

    memory.merge([_obstacle((1, 1))])
    fields.update_terrain([(1, 1)])
    after_terrain = fields.field_for(seeder.name, (17, 0), 0)
    assert after_terrain is not in_next_block

    seeder.key = 1
    assert fields.field_for(seeder.name, (17, 0), 0) is not after_terrain
    assert seeder.seeds_asked == 4


def test_each_seeder_keeps_its_own_field_while_the_robot_travels() -> None:
    """Asked at every step, two seeders each build once per block entered, not once per step."""
    treat = _CountingSeeder("treat", (Seed((39, 39), 0.0),))
    explore = _CountingSeeder("explore", (Seed((0, 39), 0.0),))
    fields = FieldProvider(AgentMemory(1, _FIELD), flies=False, seeders=[treat, explore])

    for x in range(40):
        fields.field_for(treat.name, (x, 0), 0)
        fields.field_for(explore.name, (x, 0), 0)

    blocks_entered = 3
    assert treat.seeds_asked == blocks_entered
    assert explore.seeds_asked == blocks_entered


def test_no_seeds_means_no_field() -> None:
    seeder = _CountingSeeder("empty", ())
    fields = FieldProvider(AgentMemory(1, _FIELD), flies=False, seeders=[seeder])

    assert fields.field_for(seeder.name, (0, 0), 0) is None
