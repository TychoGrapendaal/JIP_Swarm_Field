"""An example ``Renderer`` implementation and the mock one.

The ``Renderer`` port itself lives in ``swarm_field.contracts``, because a run can be handed a
renderer the same way it is handed a ``SwarmAgent``; importing it from here still works for the
demos. Visualization consumes read-only ``WorldSnapshot``s from the simulation and draws them;
it holds no simulation state and never mutates the world. The rendering technology stays behind
the port, so swapping it (a GUI, a web view, ...) touches only this package.
"""

from swarm_field.contracts import Renderer, WorldSnapshot

__all__ = ["Renderer", "TextRenderer"]


class TextRenderer:
    """A MOCK renderer that prints a one-line summary of each snapshot to stdout."""

    def render(self, snapshot: WorldSnapshot) -> None:
        print(self.summarize(snapshot))

    @staticmethod
    def summarize(snapshot: WorldSnapshot) -> str:
        field = snapshot.field
        robots = len(snapshot.robots)
        failed = sum(robot.failed for robot in snapshot.robots)
        active = robots - failed
        if failed:
            return (
                f"t={snapshot.time}: {field.width}x{field.height} grid, "
                f"{active}/{robots} robot(s) active"
            )
        return f"t={snapshot.time}: {field.width}x{field.height} grid, {robots} robot(s)"
