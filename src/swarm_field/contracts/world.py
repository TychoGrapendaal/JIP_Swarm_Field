"""Field geometry and per-cell ground-truth state.

A cell represents a patch or area of the field rather than one individual plant. Multiple
robots may occupy the same cell; robot poses are stored separately in ``WorldState``.
"""

from dataclasses import dataclass

from swarm_field.contracts.ids import CropType


@dataclass
class FieldGeometry:
    """Physical layout of the high-level field grid. All distances are in metres.

    Attributes:
        width: Field width in cells.
        height: Field height in cells.
        cell_size: Side length of one square field patch.
        row_sp: Optional crop-row spacing retained for scenario metadata.
        in_row_sp: Optional within-row spacing retained for scenario metadata.
        headland: Width of the field's headland in metres.
        connectivity: Candidate-neighbour connectivity, either 4 or 8.
    """

    width: int
    height: int
    cell_size: float
    row_sp: float
    in_row_sp: float
    headland: float
    connectivity: int = 8


@dataclass
class CellState:
    """Ground-truth state of one field patch.

    Attributes:
        crop: Crop species represented by this patch, or ``None`` when unset.
        growth_stage: Aggregate crop maturity for the patch, 0..1.
        infestation: TRUE pest severity for this patch, 0..1.
        obstacle: Whether the cell is blocked by a river, pond, fence, or similar obstacle.
    """

    crop: CropType | None
    growth_stage: float
    infestation: float
    obstacle: bool
