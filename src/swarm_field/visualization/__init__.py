"""Component C — the visualization adapter.

Consumes read-only ``WorldSnapshot``s and renders them behind the ``Renderer`` port. The
matplotlib renderer is imported lazily (it only needs matplotlib when instantiated), so this
package stays importable without the optional visualization technology installed.
"""

from swarm_field.visualization.matplotlib_renderer import MatplotlibRenderer
from swarm_field.visualization.renderer import Renderer, TextRenderer

__all__ = [
    "MatplotlibRenderer",
    "Renderer",
    "TextRenderer",
]
