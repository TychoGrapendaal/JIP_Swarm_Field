"""Composable robot behaviours for the high-level cell model."""

from dataclasses import dataclass
from random import Random
from typing import Protocol

from swarm_field.contracts import Action, Move, Observation, RobotSelfState, Time


@dataclass(frozen=True)
class ActionCandidate:
    """A scored action proposed by a behaviour."""

    action: Action
    score: float
    source: str


@dataclass(frozen=True)
class DecisionContext:
    """Read-only view handed to every behaviour at a decision point."""

    time: Time
    self_state: RobotSelfState
    sensed: list[Observation]


class Behavior(Protocol):
    """A single composable piece of decision logic."""

    name: str

    def propose(self, ctx: DecisionContext) -> ActionCandidate | tuple[ActionCandidate, ...]: ...


class Explore:
    """Mock exploration behaviour using direct target-cell moves."""

    name: str = "explore"

    def __init__(self, rng: Random) -> None:
        self._rng: Random = rng

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        """Pick a random adjacent target cell; boundary handling belongs to the world."""
        offsets = ((1, 0), (0, 1), (-1, 0), (0, -1))
        dx, dy = self._rng.choice(offsets)
        x, y = ctx.self_state.position
        action = Move(to=(x + dx, y + dy))
        return (ActionCandidate(action=action, score=1.0, source=self.name),)


class TreatInfestation:
    """Placeholder for the future pest-response behaviour."""

    name: str = "treat-infestation"

    def propose(self, ctx: DecisionContext) -> ActionCandidate:
        raise NotImplementedError
