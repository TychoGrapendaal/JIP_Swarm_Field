"""The robot's field provider: its quadtree, kept in sync with memory, and its objective fields.

A seeder's field is a grid potential field built on the quadtree and cached by its
``SeededObjectiveField``; the search graphs around the robot are kept per window size and shared
by all its seeded fields. Behaviours never read the fields. Only the brain asks
:meth:`FieldProvider.field_for`, for the winning candidate's field, which picks the robot's
target and which the ``Navigator`` follows.
"""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass

from swarm_field.contracts import CellId, Time
from swarm_field.swarm.field import (
    MixedGraph,
    Quadtree,
    solve_grid_potential_field,
    traversable_cells,
)
from swarm_field.swarm.field.quadtree import can_enter
from swarm_field.swarm.guidance import Guidance, ObjectiveField
from swarm_field.swarm.memory import AgentMemory, MemoryView
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


class _GraphCache:
    """The search graphs around one robot, one per window size, shared by its seeded fields.

    A graph depends only on the terrain and the window, not on the seeds, so a robot that stays
    in its block rebuilds a field without rebuilding the graph or its hops.
    """

    def __init__(self, quadtree: Quadtree) -> None:
        self._quadtree: Quadtree = quadtree
        self._graphs: dict[int, MixedGraph] = {}

    def around(self, robot: CellId, window_blocks: int) -> MixedGraph:
        """The search graph around the robot, kept per window size while it stays current."""
        graph = self._graphs.get(window_blocks)
        if graph is None or not graph.is_current_for(self._quadtree, robot, window_blocks):
            graph = MixedGraph.around(self._quadtree, robot, window_blocks)
            self._graphs[window_blocks] = graph
        return graph


class SeededObjectiveField:
    """The objective field of a ``FieldSeeder``: a grid potential field over the quadtree.

    The field is kept, and handed back unchanged, for as long as the seeder's key, the terrain
    and the robot's block of the quadtree all stay the same; the seeder is only asked for its
    seeds when one of them changed. Following a plan therefore costs one build per block
    travelled rather than one per step, which matters because building takes tens to hundreds
    of milliseconds while reading the field back takes microseconds.

    Attributes:
        name: The seeder's name.
    """

    def __init__(self, seeder: FieldSeeder, quadtree: Quadtree, graphs: _GraphCache) -> None:
        self.name: str = seeder.name
        self._seeder: FieldSeeder = seeder
        self._quadtree: Quadtree = quadtree
        self._graphs: _GraphCache = graphs
        self._cached: _CachedField | None = None

    def guidance_for(self, memory: MemoryView, robot: CellId, now: Time) -> SeededField | None:
        """The seeder's field, fine around the robot's cell; ``None`` if it has no targets."""
        key = self._seeder.field_key(memory, robot, now)
        version = self._quadtree.version
        block = self._block_of(robot)
        if self._cached is not None and self._cached.matches(key, version, block):
            return self._cached.field

        seeds = self._seeder.seeds(memory, robot, now)
        field: SeededField | None = None
        if seeds:
            graph = self._graphs.around(robot, self._seeder.window_blocks)
            built = solve_grid_potential_field(seeds, graph)
            field = SeededField(seeds, built)
        self._cached = _CachedField(key=key, version=version, block=block, field=field)
        return field

    def _block_of(self, robot: CellId) -> CellId:
        """The robot's block of ``max_leaf_side`` cells, the unit its field's window moves in."""
        side = self._quadtree.max_leaf_side
        return (robot[0] // side, robot[1] // side)


class FieldProvider:
    """One robot's quadtree, kept in sync with memory, and its objective fields by name.

    Every seeder becomes a ``SeededObjectiveField`` over the quadtree; other objective fields
    are registered as they are. Each field keeps its own cache, so a robot switching between
    objectives does not evict the other fields. A robot whose memory keeps no map has no
    quadtree, and so no seeders.

    Attributes:
        quadtree: Where this robot can be and move, read from its memory's terrain, or ``None``
            if its memory keeps no map.
    """

    def __init__(
        self,
        memory: AgentMemory,
        flies: bool,
        seeders: Iterable[FieldSeeder] = (),
        fields: Iterable[ObjectiveField] = (),
    ) -> None:
        """Build the robot's quadtree, if its memory keeps a map, and register its fields.

        Raises:
            ValueError: If a seeder is given for a memory without a map: a seeded field plans
                over the quadtree.
        """
        self._memory: AgentMemory = memory
        self._flies: bool = flies
        self.quadtree: Quadtree | None = None
        if memory.keeps_map:
            self.quadtree = Quadtree(traversable_cells(memory, flies))
        self._fields: dict[str, ObjectiveField] = {}
        graphs: _GraphCache | None = None
        if self.quadtree is not None:
            graphs = _GraphCache(self.quadtree)
        for seeder in seeders:
            if self.quadtree is None or graphs is None:
                raise ValueError(f"seeder {seeder.name} needs a memory that keeps a map")
            self._fields[seeder.name] = SeededObjectiveField(seeder, self.quadtree, graphs)
        for field in fields:
            self._fields[field.name] = field

    def names(self) -> frozenset[str]:
        """The names of the registered objective fields, seeded or not."""
        return frozenset(self._fields)

    def update_terrain(self, cells: Iterable[CellId]) -> None:
        """Re-read the given cells from the terrain, after new terrain facts about them."""
        if self.quadtree is None:
            return
        changes = {cell: can_enter(self._memory.terrain_at(cell), self._flies) for cell in cells}
        if changes:
            self.quadtree.update(changes)

    def field_for(self, name: str, robot: CellId, now: Time) -> Guidance | None:
        """The guidance of the objective field called ``name``, from the robot's cell.

        ``None`` means the field has nothing to guide to; an unknown name raises ``KeyError``.
        """
        return self._fields[name].guidance_for(self._memory, robot, now)
