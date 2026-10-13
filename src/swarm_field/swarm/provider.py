"""The robot's field provider: its quadtree, kept in sync with memory, and the grid potential
fields of its seeders, built on it and cached one per seeder.

Behaviours read the fields to score their bids, and the brain hands the winner's field to the
``Navigator`` to move. Both go through :meth:`FieldProvider.field_for`, so a field is built
once and shared, whichever of them asks first.
"""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass

from swarm_field.contracts import CellId, Time
from swarm_field.swarm.field import Quadtree, build_grid_potential_field, traversable_cells
from swarm_field.swarm.field.quadtree import can_enter
from swarm_field.swarm.memory import AgentMemory
from swarm_field.swarm.seeder import FieldSeeder, SeededField


@dataclass(frozen=True)
class _CachedField:
    """A seeder's field kept for reuse, valid for as long as its key still matches.

    Attributes:
        key: The seeder's ``field_key`` when the field was built.
        version: The quadtree version it was built on.
        block: The robot's block of ``max_leaf_side`` cells. The field's window is made of
            whole blocks, so it stays exact for as long as the robot stays in this one.
        field: The field itself, ``None`` if the seeder had no targets.
    """

    key: Hashable
    version: int
    block: CellId
    field: SeededField | None

    def matches(self, key: Hashable, version: int, block: CellId) -> bool:
        """Whether this field still answers for the seeder's ``key``, on this terrain and block."""
        return self.key == key and self.version == version and self.block == block


class FieldProvider:
    """Builds and caches the grid potential fields of one robot's seeders, over its quadtree.

    The seeders are registered by name when the provider is created, and each has its own
    cache slot, so behaviours reading different fields at every decision point do not evict
    each other's.

    Attributes:
        quadtree: Where this robot can be and move, read from its memory's terrain.
    """

    def __init__(
        self, memory: AgentMemory, flies: bool, seeders: Iterable[FieldSeeder] = ()
    ) -> None:
        self._memory: AgentMemory = memory
        self._flies: bool = flies
        self._seeders: dict[str, FieldSeeder] = {seeder.name: seeder for seeder in seeders}
        self.quadtree: Quadtree = Quadtree(traversable_cells(memory, flies))
        self._slots: dict[str, _CachedField] = {}

    def update_terrain(self, cells: Iterable[CellId]) -> None:
        """Re-read the given cells from the terrain, after new terrain facts about them."""
        changes = {cell: can_enter(self._memory.terrain_at(cell), self._flies) for cell in cells}
        if changes:
            self.quadtree.update(changes)

    def field_for(self, seeder: str, robot: CellId, now: Time) -> SeededField | None:
        """The grid potential field of the seeder named ``seeder``, fine around the robot's cell.

        ``None`` means the seeder has no targets; an unknown name raises ``KeyError``. The
        field is kept in the seeder's slot, and handed back unchanged, for as long as the
        seeder's key, the terrain and the robot's block of the quadtree all stay the same; the
        seeder is only asked for its seeds when one of them changed. Following a plan therefore
        costs one build per block travelled rather than one per step, which matters because
        building takes tens to hundreds of milliseconds while reading the field back takes
        microseconds.
        """
        found = self._seeders[seeder]
        key = found.field_key(self._memory, robot, now)
        version = self.quadtree.version
        block = self._block_of(robot)
        cached = self._slots.get(seeder)
        if cached is not None and cached.matches(key, version, block):
            return cached.field

        seeds = found.seeds(self._memory, robot, now)
        field: SeededField | None = None
        if seeds:
            built = build_grid_potential_field(seeds, self.quadtree, robot, found.window_blocks)
            field = SeededField(seeds, built)
        self._slots[seeder] = _CachedField(key=key, version=version, block=block, field=field)
        return field

    def _block_of(self, robot: CellId) -> CellId:
        """The robot's block of ``max_leaf_side`` cells, the unit its field's window moves in."""
        side = self.quadtree.max_leaf_side
        return (robot[0] // side, robot[1] // side)
