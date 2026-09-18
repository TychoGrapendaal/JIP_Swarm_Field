"""The ``Behavior`` port and a couple of example behaviors.

A behavior is one composable piece of decision logic: given a read-only view of the robot
at a decision point, it proposes scored candidate actions. The brain combines several
behaviors and picks between their candidates.

The two examples below are still unimplemented: ``Explore`` drives field coverage and
``TreatInfestation`` reacts to sensed pests. More behaviors (return-to-charge,
avoid-crowding, ...) are added as they are built, each as its own small class.
"""

from dataclasses import dataclass
from random import Random
from typing import Protocol

from swarm_field.contracts import Action, Move, Observation, RobotSelfState, Time, Turn


@dataclass(frozen=True)
class ActionCandidate:
    """A scored action proposed by a behavior.

    Attributes:
        action: The action being proposed.
        score: How strongly the behavior wants it; the highest score wins.
        source: Name of the proposing behavior, for inspection / debugging.
    """

    action: Action
    score: float
    source: str


@dataclass(frozen=True)
class DecisionContext:
    """Read-only view handed to every behavior at a decision point.

    Minimal for now — a memory view (accumulated belief) and the message inbox are added
    when those parts are built.

    Attributes:
        time: The current simulation time.
        self_state: The robot's view of itself.
        sensed: What the robot's sensors picked up at this decision point.
    """

    time: Time
    self_state: RobotSelfState
    sensed: list[Observation]


class Behavior(Protocol):
    """A single, composable piece of decision logic.

    ``propose`` returns a single candidate, a tuple of candidates, or an empty tuple to
    propose nothing. Returning several candidates only matters to arbiters that
    fuse or fall back between them; the current brain simply takes the highest-scoring
    candidate across all behaviors, so returning your single best is usually enough.

    Attributes:
        name: A short identifier, used as the candidate's ``source``.
    """

    name: str

    def propose(self, ctx: DecisionContext) -> ActionCandidate | tuple[ActionCandidate, ...]: ...


class Explore:
    """Propose a move that improves field coverage.

    A real implementation would return several scored directions for a fusion arbiter (hence
    the tuple return type). This one is a MOCK that ignores coverage entirely.
    """

    name: str = "explore"

    def __init__(self, rng: Random) -> None:
        # The behavior holds its own RNG. For reproducible runs it should be seeded from the
        # simulation's single seeded source when the brain is assembled.
        self._rng: Random = rng

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        # MOCK: coin flip — 50% go straight ahead, 50% turn 45 degrees left. No real
        # exploration; replace with coverage-driven logic.
        if self._rng.random() < 0.5:
            action: Action = Move(direction=ctx.self_state.facing)
        else:
            action = Turn(facing=ctx.self_state.facing.ccw())
        return (ActionCandidate(action=action, score=1.0, source=self.name),)


class TreatInfestation:
    """Propose treating a sensed infestation, scored by severity and proximity.

    Returns a single candidate (narrowing the protocol's return type to just
    ``ActionCandidate``), since it commits to one target at a time.
    """

    name: str = "treat-infestation"

    def propose(self, ctx: DecisionContext) -> ActionCandidate:
        raise NotImplementedError
