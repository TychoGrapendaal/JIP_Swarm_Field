"""Composable robot behaviours for the high-level cell model."""

from dataclasses import dataclass, field
from typing import Protocol

from swarm_field.contracts import (
    Action,
    Idle,
    Message,
    Observation,
    RobotSelfState,
    Time,
)
from swarm_field.swarm.memory import MemoryView
from swarm_field.swarm.navigator import Navigator
from swarm_field.swarm.seeder import TreatSeeder


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
        navigator: The robot's navigator: it hands out the fields of the registered seeders, and
            turns a field into the cell to enter.
        inbox: The messages received at this decision point.
    """

    time: Time
    self_state: RobotSelfState
    sensed: list[Observation]
    memory: MemoryView
    navigator: Navigator
    inbox: list[Message] = field(default_factory=list)


class Behavior(Protocol):
    """A single composable piece of decision logic."""

    name: str

    def propose(self, ctx: DecisionContext) -> ActionCandidate | tuple[ActionCandidate, ...]: ...


class TreatInfestation:
    """Skeleton of the pest-response behaviour: follow the field of the TREAT seeder."""

    name: str = "treat-infestation"

    def __init__(self, seeder: str = TreatSeeder.name) -> None:
        self._seeder: str = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        position = ctx.self_state.position
        seeded = ctx.navigator.field_for(self._seeder, position, ctx.time)
        if seeded is None:
            return ()
        step = ctx.navigator.step_towards(seeded, position)
        if isinstance(step, Idle):
            # Either standing on the target, or the steerer gave up on a target it cannot reach.
            # TODO: standing on the target: propose Treat(cell=position) instead.
            return ()
        # TODO: a score comparable across objectives, e.g. from the field's potential at
        # `position` and a weight per objective (see the arbiter in the field-level design).
        # Lower potential is better, so negate it for now.
        score = -seeded.field.value_at(position)
        return (ActionCandidate(action=step, score=score, source=self.name),)
