"""Component C — the visualization adapter.

Consumes read-only ``WorldSnapshot``s and renders them behind the ``Renderer`` port;
``SimulationViewer`` animates any running simulation with overlays, trails, goals and a panel. The
matplotlib renderer is imported lazily (it only needs matplotlib when instantiated), so this
package stays importable without the optional visualization technology installed.
"""

from swarm_field.visualization.matplotlib_renderer import MatplotlibRenderer
from swarm_field.visualization.renderer import Renderer, TextRenderer
from swarm_field.visualization.viewer import SimulationViewer

__all__ = [
    "MatplotlibRenderer",
    "Renderer",
    "SimulationViewer",
    "TextRenderer",
]
