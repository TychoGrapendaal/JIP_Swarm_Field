"""The field maps of the experiments: the field's size, which cells are blocked, and the crops.

A ``MapConfig`` is one scenario's field. ``river_and_rocks`` is the small field's map, one river
with a bridge and two rocks, and ``two_rivers_and_rocks`` the large field's, two rivers with
three bridges and eight rocks. An open field has no blocked cells, ``()``.

The four standard maps, to compose a ``SwarmScenario`` from: ``SMALL_OPEN_MAP``,
``LARGE_OPEN_MAP``, ``SMALL_RIVER_MAP`` and ``LARGE_RIVERS_MAP``.
"""

from __future__ import annotations

from dataclasses import dataclass

from swarm_field.contracts import CellId, CropType, FieldGeometry


@dataclass(frozen=True)
class MapConfig:
    """A square field of cells, and the cells blocked on it.

    Attributes:
        size: The field's side in cells.
        obstacles: The blocked cells; ``None`` for ``river_and_rocks``, ``()`` for an open field.
        cell_size: The side of a cell in metres.
        docks: The cells where robots recharge and killers restock, any number of robots at
            once. Explorers and seeded killers return to one when their battery runs low;
            ``()`` for none, so that a robot fails once its battery runs out.
        dock_range_m: How far a dock's own beacon reaches, for robots that relay beacons;
            ``None`` for the robots' radio range. A larger one is a stronger dock antenna.
    """

    size: int = 32
    obstacles: tuple[CellId, ...] | None = None
    cell_size: float = 10.0
    docks: tuple[CellId, ...] = ((0, 0),)
    dock_range_m: float | None = None

    def __post_init__(self) -> None:
        """Check that every dock is a free cell of the field.

        Raises:
            ValueError: If a dock is outside the field or blocked.
        """
        if self.dock_range_m is not None and self.dock_range_m < 0.0:
            raise ValueError("dock_range_m must not be negative")
        blocked = set(self.blocked_cells())
        for dock in self.docks:
            if not self.is_inside(dock) or dock in blocked:
                raise ValueError(f"dock {dock} is not a free cell of the field")

    def field_geometry(self) -> FieldGeometry:
        return FieldGeometry(
            width=self.size,
            height=self.size,
            cell_size=self.cell_size,
            row_sp=0.0,
            in_row_sp=0.0,
            headland=0.0,
        )

    def blocked_cells(self) -> tuple[CellId, ...]:
        """The obstacles, or ``river_and_rocks`` when none are given."""
        if self.obstacles is None:
            return river_and_rocks(self.size)
        return self.obstacles

    def free_cells(self) -> int:
        """How many cells are not blocked."""
        return self.size * self.size - len(set(self.blocked_cells()))

    def is_inside(self, cell: CellId) -> bool:
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size


def cropped_cells(field: FieldGeometry, obstacles: tuple[CellId, ...]) -> dict[CellId, CropType]:
    """A crop on every free cell, so that ``MetricsCollector`` measures the whole field."""
    blocked = set(obstacles)
    crops: dict[CellId, CropType] = {}
    for y in range(field.height):
        for x in range(field.width):
            if (x, y) not in blocked:
                crops[(x, y)] = CropType.MAIZE
    return crops


def river_and_rocks(size: int) -> tuple[CellId, ...]:
    """A river across the middle of the field with one bridge, and a few square rocks."""
    river_x = size // 2
    bridge_y = 3 * size // 4
    cells: list[CellId] = []
    for y in range(size):
        if abs(y - bridge_y) > 1:
            cells.extend([(river_x, y), (river_x + 1, y)])
    for corner_x, corner_y in ((size // 4, size // 3), (3 * size // 4, size // 5)):
        for dx in range(3):
            cells.extend((corner_x + dx, corner_y + dy) for dy in range(3))
    return tuple(cells)


def two_rivers_and_rocks(size: int) -> tuple[CellId, ...]:
    """Two rivers with three bridges between them, and eight square rocks.

    The rivers are two cells wide, at a third and two thirds of the width. The first has two
    bridges, at a quarter and three quarters of the height, the second one at half the height;
    each bridge is three cells wide. The rocks are 3 by 3 cells, spread over the three banks.
    """
    cells: list[CellId] = []
    for river_x, bridges in (
        (size // 3, (size // 4, 3 * size // 4)),
        (2 * size // 3, (size // 2,)),
    ):
        for y in range(size):
            on_bridge = False
            for bridge_y in bridges:
                if abs(y - bridge_y) <= 1:
                    on_bridge = True
            if not on_bridge:
                cells.extend([(river_x, y), (river_x + 1, y)])
    rocks = ((0.1, 0.15), (0.15, 0.6), (0.4, 0.12), (0.5, 0.4))
    rocks += ((0.42, 0.78), (0.8, 0.2), (0.85, 0.55), (0.75, 0.85))
    for fraction_x, fraction_y in rocks:
        corner_x = int(fraction_x * size)
        corner_y = int(fraction_y * size)
        for dx in range(3):
            cells.extend((corner_x + dx, corner_y + dy) for dy in range(3))
    return tuple(cells)


SMALL_OPEN_MAP = MapConfig(obstacles=())
"""The small open field: 32 x 32 cells of 10 m without obstacles, and a dock at ``(0, 0)``."""

LARGE_OPEN_MAP = MapConfig(size=64, obstacles=(), docks=((0, 0), (63, 63)))
"""The large open field: 64 x 64 cells of 10 m, four times the area, without obstacles, and a
dock in each of two opposite corners."""

SMALL_RIVER_MAP = MapConfig(obstacles=river_and_rocks(32), docks=((0, 0), (31, 31)))
"""The small river field: 32 x 32 cells of 10 m, one river with a bridge and two rocks
(``river_and_rocks``), and a dock on each bank of the river, at ``(0, 0)`` and ``(31, 31)``,
since a hop killer finds a dock across the river only slowly. ``MapConfig()`` is the same field
with the dock at ``(0, 0)`` only."""

LARGE_RIVERS_MAP = MapConfig(size=64, obstacles=two_rivers_and_rocks(64), docks=((0, 0), (63, 63)))
"""The large river field: 64 x 64 cells of 10 m, four times the area, with two rivers and three
bridges and eight rocks (``two_rivers_and_rocks``), and a dock in each of two opposite
corners."""
