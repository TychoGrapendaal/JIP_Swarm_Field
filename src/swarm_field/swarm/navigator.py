"""The robot's navigator: its quadtree, kept in sync with memory, and the fields built on it."""

from collections.abc import Iterable

from swarm_field.contracts import CellId
from swarm_field.swarm.field import (
    PotentialField,
    Quadtree,
    Seed,
    TerrainView,
    build_potential_field,
    traversable_cells,
)
from swarm_field.swarm.field.quadtree import can_enter


class Navigator:
    """Builds potential fields for one robot, over a quadtree of the cells it can enter.

    Attributes:
        quadtree: Where this robot can be and move, read from its memory's terrain.
    """

    def __init__(self, terrain: TerrainView, flies: bool) -> None:
        self._terrain: TerrainView = terrain
        self._flies: bool = flies
        self.quadtree: Quadtree = Quadtree(traversable_cells(terrain, flies))

    def update_terrain(self, cells: Iterable[CellId]) -> None:
        """Re-read the given cells from the terrain, after new terrain facts about them."""
        changes = {cell: can_enter(self._terrain.terrain_at(cell), self._flies) for cell in cells}
        if changes:
            self.quadtree.update(changes)

    def field_for(self, seeds: tuple[Seed, ...], robot: CellId) -> PotentialField:
        """The potential field towards ``seeds``, fine around the robot's cell."""
        # TODO: cache fields. A field can be reused while the seeds, the terrain and the
        # robot's block of the quadtree stay the same; for now every call builds a new one.
        return build_potential_field(seeds, self.quadtree, robot)
