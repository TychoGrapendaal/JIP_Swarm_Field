"""Field geometry and per-cell ground-truth state.

``CellState`` is ground truth owned by the simulation (B.1); it is never read directly by
the algorithm, only surfaced to visualization via ``WorldSnapshot``.
"""

from dataclasses import dataclass

from swarm_field.contracts.ids import CropType, RobotId


@dataclass
class FieldGeometry:
    """Physical layout of the field. All distances are in metres.

    Attributes:
        width: Field width in cells.
        height: Field height in cells.
        cell_size: Side length of one square cell.
        row_sp: Row spacing — distance between crop rows (the drivable aisle pitch).
        in_row_sp: In-row spacing — distance between plants along a single row.
        headland: Width of the turning strip at the ends of the rows.
    """

    width: int
    height: int
    cell_size: float
    row_sp: float
    in_row_sp: float
    headland: float


@dataclass
class CellState:
    """Ground-truth state of one grid cell.

    Attributes:
        crop: Crop growing here, or ``None`` if there is no plant.
        growth_stage: Crop maturity, 0..1.
        infestation: TRUE pest severity, 0..1.
        obstacle: Whether the cell blocks movement.
        occupant: The robot currently in the cell, if any.
    """

    crop: CropType | None
    growth_stage: float
    infestation: float
    obstacle: bool
    occupant: RobotId | None
