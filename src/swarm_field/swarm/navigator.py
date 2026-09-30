"""The robot's navigator: its quadtree, kept in sync with memory, the grid potential fields
built on it, and the one seam that turns a field into a movement action.

The navigator holds the robot's belief about where it can move — a ``Quadtree`` over the
traversable cells it has learned from its memory — and the grid potential field an objective's
targets induce on that belief. It is also where movement is decided: :meth:`Navigator.step_towards`
follows the field, and hands over to the reactive ``Steerer`` when the field cannot guide the
robot, which is the fallback that keeps a robot moving over ground it has no plan for.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from swarm_field.contracts import CellId, Idle, Move
from swarm_field.swarm.field import (
    GridPotentialField,
    Quadtree,
    Seed,
    TerrainView,
    build_grid_potential_field,
    traversable_cells,
)
from swarm_field.swarm.field.quadtree import can_enter
from swarm_field.swarm.steering import QuadtreeSensing, Steerer, SteeringConfig

_PLAN_PATIENCE_TICKS = 3
"""Field-following ticks without the potential beating its minimum before the steerer takes over.

The field's window makes the potential approximate, so one tick of standstill is not a loop; a
few of them is.
"""


@dataclass(frozen=True)
class _CachedField:
    """A field kept for reuse, valid for as long as its key still matches.

    Attributes:
        seeds: The targets the field was built from.
        version: The quadtree version it was built on.
        block: The robot's block of ``max_leaf_side`` cells. The field's window is made of
            whole blocks, so it stays exact for as long as the robot stays in this one.
        field: The field itself.
    """

    seeds: tuple[Seed, ...]
    version: int
    block: CellId
    field: GridPotentialField

    def matches(self, seeds: tuple[Seed, ...], version: int, block: CellId) -> bool:
        """Whether this field still answers for ``seeds`` on terrain ``version``, in ``block``."""
        return self.seeds == seeds and self.version == version and self.block == block


class Navigator:
    """Builds grid potential fields for one robot, over a quadtree of the cells it can enter.

    Attributes:
        quadtree: Where this robot can be and move, read from its memory's terrain.
    """

    def __init__(
        self, terrain: TerrainView, flies: bool, steering: SteeringConfig | None = None
    ) -> None:
        self._terrain: TerrainView = terrain
        self._flies: bool = flies
        self._config: SteeringConfig = steering if steering is not None else SteeringConfig()
        self.quadtree: Quadtree = Quadtree(traversable_cells(terrain, flies))
        self._steerer: Steerer = Steerer(QuadtreeSensing(self.quadtree), self._config)
        self._cached: _CachedField | None = None
        self._best_value: float | None = None
        self._no_progress_ticks: int = 0

    def update_terrain(self, cells: Iterable[CellId]) -> None:
        """Re-read the given cells from the terrain, after new terrain facts about them."""
        changes = {cell: can_enter(self._terrain.terrain_at(cell), self._flies) for cell in cells}
        if changes:
            self.quadtree.update(changes)

    def field_for(self, seeds: tuple[Seed, ...], robot: CellId) -> GridPotentialField:
        """The grid potential field towards ``seeds``, fine around the robot's cell.

        The field is kept, and handed back unchanged, for as long as the seeds, the terrain and
        the robot's block of the quadtree all stay the same. Following a plan therefore costs
        one build per block travelled rather than one per step, which matters because building
        takes tens to hundreds of milliseconds while reading the field back takes microseconds.
        """
        block = self._block_of(robot)
        cached = self._cached
        if cached is not None and cached.matches(seeds, self.quadtree.version, block):
            return cached.field
        field = build_grid_potential_field(seeds, self.quadtree, robot)
        self._cached = _CachedField(
            seeds=seeds, version=self.quadtree.version, block=block, field=field
        )
        self._forget_plan()
        return field

    def step_towards(self, seeds: tuple[Seed, ...], robot: CellId) -> Move | Idle:
        """The one movement action of this decision point, on the way towards ``seeds``.

        A field guides the robot while it can: the robot steps into the neighbouring cell the
        field points at, which is one step of a route that is near-optimal around the obstacles
        the quadtree knows. The reactive ``Steerer`` takes over when the field cannot guide —
        no target is reachable, or following the field stopped bringing the robot closer, which
        is how the window's edge can make a robot turn back and forth. The steerer then walks
        towards the target the objective named, and gives way again as soon as the field points
        forward once more.

        Args:
            seeds: The targets of the objective being pursued, as the objective scored them.
            robot: The robot's current cell, the origin the step is taken from.
        """
        if not seeds:
            self._forget_plan()
            return Idle()

        field = self.field_for(seeds, robot)
        target = field.target_for(robot)
        if target == robot:
            self._forget_plan()
            return Idle()
        if target is None:
            # No wave reaches this robot, so the field has no route to offer. The objective's
            # targets are known all the same, so the steerer walks towards the most attractive
            # one and stalls if it is walled off for good, which is how the brain hears about it.
            return self._steer_towards(_attractive_seed(seeds).cell, robot)

        step = field.next_cell(robot)
        if step is not None and self._plan_progresses(field.value_at(robot)):
            self._steerer.reset()
            return Move(to=step)
        offset = (float(target[0] - robot[0]), float(target[1] - robot[1]))
        return self._steerer.steer(offset, robot)

    def _steer_towards(self, target: CellId, robot: CellId) -> Move | Idle:
        """The ``Steerer``'s step towards an absolute ``target``, as a relative offset."""
        offset = (float(target[0] - robot[0]), float(target[1] - robot[1]))
        return self._steerer.steer(offset, robot)

    def _block_of(self, robot: CellId) -> CellId:
        """The robot's block of ``max_leaf_side`` cells, the unit its field's window moves in."""
        side = self.quadtree.max_leaf_side
        return (robot[0] // side, robot[1] // side)

    def _plan_progresses(self, value: float) -> bool:
        """Whether following the field still brings the robot closer to its target.

        ``value`` is the potential at the robot's cell, in cells of travel left to the target,
        so a potential that keeps beating its running minimum is progress. A few ticks of
        anything else means the plan is no longer guiding the robot.
        """
        if self._best_value is None or value < self._best_value - self._config.progress_epsilon:
            self._best_value = value
            self._no_progress_ticks = 0
            return True
        self._no_progress_ticks += 1
        return self._no_progress_ticks < _PLAN_PATIENCE_TICKS

    def _forget_plan(self) -> None:
        """Drop what was tracked about the plan that is no longer in force."""
        self._best_value = None
        self._no_progress_ticks = 0


def _attractive_seed(seeds: tuple[Seed, ...]) -> Seed:
    """The seed the robot is most drawn to: the lowest start value, ties by position."""
    return min(seeds, key=lambda seed: (seed.value, seed.cell))
