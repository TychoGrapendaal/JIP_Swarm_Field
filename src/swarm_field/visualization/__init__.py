"""Component C — the visualization adapter.

Consumes read-only ``WorldSnapshot``s and renders them behind the ``Renderer`` port, which lives
in ``swarm_field.contracts`` so a run can be handed a renderer the way it is handed an agent.
``SimulationViewer`` animates any running simulation with overlays, trails, goals and a panel.
The matplotlib renderers are imported lazily (they only need matplotlib when instantiated), so
this package stays importable without the optional visualization technology installed.
"""

from swarm_field.contracts import Renderer
from swarm_field.visualization.matplotlib_renderer import (
    FrameWriter,
    MatplotlibRenderer,
    RobotStyle,
    save_frames,
)
from swarm_field.visualization.renderer import TextRenderer
from swarm_field.visualization.viewer import Arrow, Marker, Polyline, SimulationViewer

__all__ = [
    "Arrow",
    "FrameWriter",
    "Marker",
    "MatplotlibRenderer",
    "Polyline",
    "Renderer",
    "RobotStyle",
    "SimulationViewer",
    "TextRenderer",
    "save_frames",
]
