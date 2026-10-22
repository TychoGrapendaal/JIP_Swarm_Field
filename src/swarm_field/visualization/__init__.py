"""Component C — the visualization adapter.

Consumes read-only ``WorldSnapshot``s and renders them behind the ``Renderer`` port, which lives
in ``swarm_field.contracts`` so a run can be handed a renderer the way it is handed an agent.
``overlays`` draws what a study adds on top of a rendered snapshot: a cell overlay, trails,
targets, lines, arrows, markers and a side panel. The matplotlib renderers are imported lazily
(they only need matplotlib when instantiated), so this package stays importable without the
optional visualization technology installed.
"""

from swarm_field.contracts import Renderer
from swarm_field.visualization.matplotlib_renderer import (
    FrameWriter,
    MatplotlibRenderer,
    RobotStyle,
    save_frames,
)
from swarm_field.visualization.overlays import (
    Arrow,
    CellOverlay,
    Marker,
    PanelSource,
    Polyline,
    RobotTrails,
    SidePanel,
    draw_arrows,
    draw_cell_overlay,
    draw_markers,
    draw_polylines,
    draw_targets,
)
from swarm_field.visualization.renderer import TextRenderer

__all__ = [
    "Arrow",
    "CellOverlay",
    "FrameWriter",
    "Marker",
    "MatplotlibRenderer",
    "PanelSource",
    "Polyline",
    "Renderer",
    "RobotStyle",
    "RobotTrails",
    "SidePanel",
    "TextRenderer",
    "draw_arrows",
    "draw_cell_overlay",
    "draw_markers",
    "draw_polylines",
    "draw_targets",
    "save_frames",
]
