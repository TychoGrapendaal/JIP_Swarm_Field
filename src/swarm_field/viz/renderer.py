"""The ``Renderer`` port and an example renderer.

Visualization consumes read-only ``WorldSnapshot``s from the simulation and draws them; it
holds no simulation state and never mutates the world. The rendering technology stays behind
the ``Renderer`` port, so swapping it (a GUI, a web view, ...) touches only this package.
Interactive commands and robot inspection are added once those flows are built.
"""

from typing import Protocol

from swarm_field.contracts import WorldSnapshot


class Renderer(Protocol):
    """Draws a snapshot of the world. One implementation per rendering technology."""

    def render(self, snapshot: WorldSnapshot) -> None: ...


class TextRenderer:
    """A MOCK renderer that prints a one-line summary of each snapshot to stdout."""

    def render(self, snapshot: WorldSnapshot) -> None:
        print(self.summarize(snapshot))

    @staticmethod
    def summarize(snapshot: WorldSnapshot) -> str:
        field = snapshot.field
        robots = len(snapshot.robots)
        return f"tick {snapshot.tick}: {field.width}x{field.height} grid, {robots} robot(s)"
