"""The ``SimStage`` port and a couple of example stages.

A stage does one thing to the shared ``SimContext``; the loop runs an ordered list of them.
The two examples below are still unimplemented: ``Decide`` shows the boundary with the
algorithm (calling each ``SwarmAgent``), and ``SpreadPests`` shows an uncoupled world
dynamic. Real stages (perceive, act, deliver messages, snapshot, ...) are added as they are
built, each as its own small class.
"""

from typing import Protocol

from swarm_field.sim.context import SimContext


class SimStage(Protocol):
    """One step of a simulation tick.

    Attributes:
        name: A short identifier, handy for logging and inspection.
    """

    name: str

    def run(self, ctx: SimContext) -> None: ...


class Decide:
    """Call ``SwarmAgent.decide`` for each robot and store the result in ``ctx.decisions``.

    Building each robot's ``Perception`` and the scheduling policy (all robots against a
    frozen snapshot vs. sequentially against a live world) live here, invisible to A.
    """

    name: str = "decide"

    def run(self, ctx: SimContext) -> None:
        raise NotImplementedError


class SpreadPests:
    """Advance infestation dynamics for one tick.

    Fully uncoupled from perception, decision and action: changing the pest model is
    changing this one stage. Draws any randomness from ``ctx.rng``.
    """

    name: str = "spread-pests"

    def run(self, ctx: SimContext) -> None:
        raise NotImplementedError
