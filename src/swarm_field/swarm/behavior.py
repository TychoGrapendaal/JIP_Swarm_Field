"""Composable robot behaviours for the high-level cell model."""

from dataclasses import dataclass
from random import Random
from typing import Protocol

from swarm_field.contracts import Action, Move, Observation, RobotSelfState, Time
from swarm_field.swarm.memory import MemoryView
from swarm_field.swarm.navigator import Navigator
from swarm_field.swarm.objective import Objective, TreatObjective


@dataclass(frozen=True)
class ActionCandidate:
    """A scored action proposed by a behaviour."""

    action: Action
    score: float
    source: str


@dataclass(frozen=True)
class DecisionContext:
    """Read-only view handed to every behaviour at a decision point.

    Attributes:
        time: The time of the decision.
        self_state: The robot's perception of itself.
        sensed: What the robot sensed at this decision point (already merged into memory).
        memory: The robot's facts about the field and its peers.
        navigator: Builds potential fields towards an objective's targets.
    """

    time: Time
    self_state: RobotSelfState
    sensed: list[Observation]
    memory: MemoryView
    navigator: Navigator


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
    """Skeleton of the pest-response behaviour: follow the field of the TREAT objective."""

    name: str = "treat-infestation"

    def __init__(self, objective: Objective | None = None) -> None:
        self._objective: Objective = TreatObjective()
        if objective is not None:
            self._objective = objective

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        seeds = self._objective.seeds(ctx.memory, ctx.time)
        if not seeds:
            return ()
        position = ctx.self_state.position
        field = ctx.navigator.field_for(seeds, position)
        step = field.next_cell(position)
        if step is None:
            # TODO: standing on the target: propose Treat(cell=position) instead of a move.
            return ()
        # TODO: a score comparable across objectives, e.g. from field.value_at(position) and a
        # weight per objective (see the arbiter in the field-level design). Lower potential is
        # better, so negate it for now.
        score = -field.value_at(position)
        return (ActionCandidate(action=Move(to=step), score=score, source=self.name),)
