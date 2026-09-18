"""Component C — the visualization adapter.

Consumes read-only ``WorldSnapshot``s and renders them behind the ``Renderer`` port. Kept
minimal for now: the port plus one example text renderer.
"""

from swarm_field.visualization.renderer import Renderer, TextRenderer

__all__ = [
    "Renderer",
    "TextRenderer",
]
