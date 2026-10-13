"""The composed robot brain (A).

``ComposedAgent`` is the thin orchestrator that implements the ``SwarmAgent`` boundary. At each
decision point it merges what the robot sensed and received into its memory, keeps the field
provider's quadtree in sync with new terrain facts, gathers the scored candidates of its
behaviors, picks the best and turns its intent into one action: the navigator steps along the
field a ``FollowField`` names, and any other intent passes through. Knowledge broadcasts share the
robot memory, and a per-body capability gate filters candidates and validates the final action.
"""

import logging
from dataclasses import replace
from typing import cast

from swarm_field.contracts import (
    Action,
    AgentInspection,
    CellId,
    Decision,
    Idle,
    Message,
    MessageKind,
    Move,
    Observation,
    ObservationBase,
    Perception,
    RobotId,
    TerrainObservation,
    Time,
)
from swarm_field.swarm.capabilities import CapabilitySet, capabilities_for
from swarm_field.swarm.behavior import (
    ActionCandidate,
    Behavior,
    DecisionContext,
    FollowField,
    Intent,
)
from swarm_field.swarm.memory import AgentMemory
from swarm_field.swarm.navigator import Navigator
from swarm_field.swarm.provider import FieldProvider

logger = logging.getLogger(__name__)


class ComposedAgent:
    """A robot brain assembled from a memory, a field provider, a navigator and behaviors.

    Attributes:
        robot_id: The robot this brain drives.
        behaviors: The composable pieces of decision logic, combined at each decision point.
        memory: The robot's facts about the field and its peers.
        fields: The robot's quadtree, and the grid potential fields of its seeders, shared by
            the behaviors (to score) and the navigator (to move).
        navigator: Moves the robot along the winning candidate's field.
    """

    def __init__(
        self,
        robot_id: RobotId,
        behaviors: list[Behavior],
        memory: AgentMemory,
        fields: FieldProvider,
        navigator: Navigator,
        capabilities: CapabilitySet | None = None,
    ) -> None:
        self.robot_id: RobotId = robot_id
        self.behaviors: list[Behavior] = behaviors
        self.memory: AgentMemory = memory
        self.fields: FieldProvider = fields
        self.navigator: Navigator = navigator
        self.capabilities: CapabilitySet | None = capabilities
        self._current_goal: str | None = None

    def decide(self, perception: Perception) -> Decision:
        received = self._facts_received(perception.inbox)
        self.memory.ingest(perception)
        self.memory.merge(received)
        terrain_cells = [
            fact.cell
            for fact in [*perception.sensed, *received]
            if isinstance(fact, TerrainObservation)
        ]
        self.fields.update_terrain(terrain_cells)

        # Ground-truth body identity in the perception is authoritative. This also keeps
        # manually assembled agents safe if they were passed the wrong capability profile.
        canonical_capabilities = capabilities_for(perception.self_state.body)
        if self.capabilities is None or self.capabilities.body is not perception.self_state.body:
            capabilities = canonical_capabilities
        else:
            # Explicit profiles may make a canonical body stricter, never grant it extra
            # actions or physical abilities that the body type does not possess.
            capabilities = replace(
                canonical_capabilities,
                action_types=(
                    canonical_capabilities.action_types & self.capabilities.action_types
                )
                | {Idle},
            )

        ctx = DecisionContext(
            time=perception.time,
            self_state=perception.self_state,
            sensed=perception.sensed,
            memory=self.memory,
            fields=self.fields,
            inbox=perception.inbox,
            capabilities=capabilities,
        )
        candidates: list[ActionCandidate] = []
        for behavior in self.behaviors:
            for candidate in behavior.propose(ctx):
                if self._candidate_is_allowed(candidate, capabilities):
                    candidates.append(candidate)

        # TODO: an arbiter (fixed-priority overrides such as low battery, hysteresis against
        # switching objectives back and forth) instead of simply taking the highest score.
        best = max(candidates, key=lambda c: c.score, default=None)
        outbox = self._messages_to_send(perception.time)
        if best is None:
            self._current_goal = None
            return Decision(action=Idle(), outbox=outbox)

        self._current_goal = best.source
        action = self._action_for(best.action, perception.self_state.position, perception.time)
        # Validate the concrete action as well as the proposed intent: a FollowField intent
        # becomes a Move (or Idle) only after the navigator has resolved it.
        if not capabilities.can(type(action)):
            logger.warning(
                "Blocked disallowed action %s from robot %s (%s)",
                type(action).__name__,
                self.robot_id,
                capabilities.body.name,
            )
            action = Idle()
            self._current_goal = None
        return Decision(action=action, outbox=outbox)

    @staticmethod
    def _candidate_is_allowed(
        candidate: ActionCandidate, capabilities: CapabilitySet
    ) -> bool:
        """Apply the capability gate before candidate scoring.

        A ``FollowField`` is a movement intent, not an action crossing the simulation boundary,
        so it is checked against the capability for ``Move``. Other intents map directly to
        their action classes.
        """
        action_type = Move if isinstance(candidate.action, FollowField) else type(candidate.action)
        return capabilities.can(action_type)

    def inspect(self) -> AgentInspection:
        return AgentInspection(
            robot_id=self.robot_id,
            current_goal=self._current_goal,
            belief=self.memory.snapshot(),
        )

    def _action_for(self, intent: Intent, robot: CellId, now: Time) -> Action:
        """The action that carries out the winning candidate's intent."""
        if not isinstance(intent, FollowField):
            return intent
        field = self.fields.field_for(intent.seeder, robot, now)
        if field is None:
            logger.warning("FollowField couldn't find the corresponding field %s", intent.seeder)
            return Idle()
        return self.navigator.step_towards(field, robot)

    def _facts_received(self, inbox: list[Message]) -> list[Observation]:
        """The facts carried by the received messages."""
        facts: list[Observation] = []
        for message in inbox:
            if message.kind is not MessageKind.KNOWLEDGE:
                continue
            if not isinstance(message.payload, (tuple, list)):
                continue
            for fact in message.payload:
                if isinstance(fact, ObservationBase):
                    facts.append(cast(Observation, fact))
        return facts

    def _messages_to_send(self, now: Time) -> list[Message]:
        """The messages to broadcast at this decision point."""
        return [
            Message(
                sender=self.robot_id,
                sent_at=now,
                kind=MessageKind.KNOWLEDGE,
                payload=tuple(self.memory.records()),
            )
        ]
