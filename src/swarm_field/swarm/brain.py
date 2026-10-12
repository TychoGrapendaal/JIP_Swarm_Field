"""The composed robot brain (A).

``ComposedAgent`` is the thin orchestrator that implements the ``SwarmAgent`` boundary. At each
decision point it merges what the robot sensed and received into its memory, keeps the field
provider's quadtree in sync with new terrain facts, gathers the scored candidates of its
behaviors, picks the best and turns its intent into one action: the navigator steps along the
field a ``FollowField`` names, and any other intent passes through. Communication and capability
gating are added as those parts are built.
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
    Observation,
    ObservationBase,
    PeerTargetObservation,
    Perception,
    RobotId,
    TerrainObservation,
    Time,
)
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
    ) -> None:
        self.robot_id: RobotId = robot_id
        self.behaviors: list[Behavior] = behaviors
        self.memory: AgentMemory = memory
        self.fields: FieldProvider = fields
        self.navigator: Navigator = navigator
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

        ctx = DecisionContext(
            time=perception.time,
            self_state=perception.self_state,
            sensed=perception.sensed,
            memory=self.memory,
            fields=self.fields,
            inbox=perception.inbox,
        )
        candidates: list[ActionCandidate] = []
        for behavior in self.behaviors:
            candidates.extend(behavior.propose(ctx))

        # A fresh peer announcement should make the same target less attractive for the same
        # objective, encouraging the swarm to spread work across the field.
        candidates = self._discount_peer_claimed_targets(candidates, perception.time)

        # TODO: an arbiter (fixed-priority overrides such as low battery, hysteresis against
        # switching objectives back and forth) instead of simply taking the highest score.
        best = max(candidates, key=lambda c: c.score, default=None)
        outbox = self._messages_to_send(perception.time, best)
        if best is None:
            self._current_goal = None
            return Decision(action=Idle(), outbox=outbox)

        self._current_goal = best.source
        action = self._action_for(best.action, perception.self_state.position, perception.time)
        return Decision(action=action, outbox=outbox)

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

    def _discount_peer_claimed_targets(
        self, candidates: list[ActionCandidate], now: Time
    ) -> list[ActionCandidate]:
        """Reduce a candidate's score when a peer is already pursuing its target.

        The penalty is proportional to the decayed confidence of the peer announcement. It is
        applied only when both the target cell and objective match, so claims for another task
        do not suppress this candidate.
        """
        adjusted: list[ActionCandidate] = []
        for candidate in candidates:
            if candidate.target is None:
                adjusted.append(candidate)
                continue
            objective = self._objective_for(candidate)
            if objective is None:
                adjusted.append(candidate)
                continue
            confidence = self.memory.peer_target_confidence(candidate.target, objective, now)
            if confidence <= 0.0:
                adjusted.append(candidate)
                continue
            penalty = confidence * max(abs(candidate.score), 1.0)
            adjusted.append(replace(candidate, score=candidate.score - penalty))
        return adjusted

    @staticmethod
    def _objective_for(candidate: ActionCandidate) -> str | None:
        """Identify the behaviour/objective pursuing a candidate's target."""
        if candidate.target is None:
            return None
        return candidate.source

    def _facts_received(self, inbox: list[Message]) -> list[Observation]:
        """The facts carried by the received messages."""
        facts: list[Observation] = []
        for message in inbox:
            if message.kind is not MessageKind.KNOWLEDGE:
                continue
            if not isinstance(message.payload, (tuple, list)):
                continue
            for fact in message.payload:
                if isinstance(fact, (ObservationBase, PeerTargetObservation)):
                    facts.append(cast(Observation, fact))
        return facts

    def _messages_to_send(self, now: Time, best: ActionCandidate | None) -> list[Message]:
        """Broadcast remembered facts plus this robot's current target announcement."""
        target = best.target if best is not None else None
        objective = self._objective_for(best) if best is not None else None
        # Explicitly announcing a cleared target prevents a previous target from blocking a
        # region until its confidence happens to decay.
        current_target = PeerTargetObservation(
            source=self.robot_id,
            target=target,
            objective=objective if target is not None else None,
            confidence=1.0,
            observed_at=now,
        )
        return [
            Message(
                sender=self.robot_id,
                sent_at=now,
                kind=MessageKind.KNOWLEDGE,
                payload=(*tuple(self.memory.records()), current_target),
            )
        ]
