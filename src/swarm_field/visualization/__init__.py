"""Component C — the visualization adapter.

Consumes read-only ``WorldSnapshot``s and renders them behind the ``Renderer`` port, which lives
in ``swarm_field.contracts`` so a run can be handed a renderer the way it is handed an agent.
The matplotlib renderers are imported lazily (they only need matplotlib when instantiated), so
this package stays importable without the optional visualization technology installed.
"""

from swarm_field.contracts import Renderer
from swarm_field.visualization.matplotlib_renderer import (
    FrameWriter,
    MatplotlibRenderer,
    save_frames,
)
from swarm_field.visualization.renderer import TextRenderer

__all__ = [
    "FrameWriter",
    "MatplotlibRenderer",
    "Renderer",
    "TextRenderer",
    "save_frames",
]
