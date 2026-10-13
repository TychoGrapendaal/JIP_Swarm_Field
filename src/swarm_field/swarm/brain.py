"""The composed robot brain (A).

``ComposedAgent`` is the thin orchestrator that implements the ``SwarmAgent`` boundary. At each
decision point it merges what the robot sensed and received into its memory, keeps the field
provider's quadtree in sync with new terrain facts, gathers the scored candidates of its
behaviors, lets its arbiter pick one and turns its intent into one action: the navigator steps
along the field a ``FollowField`` names, and any other intent passes through. The robot
broadcasts its memory in a ``KNOWLEDGE`` message and merges the facts of the ``KNOWLEDGE``
messages it receives. Capability gating is added as that part is built.
"""

import logging
from typing import cast

from swarm_field.contracts import (
    Action,
    AgentInspection,
    AgentObjective,
    CellId,
    Decision,
    Idle,
    Message,
    MessageKind,
    Observation,
    ObservationBase,
    PeerTargetObservation,
    PeerObservation,
    Perception,
    RobotId,
    TerrainObservation,
    Time,
)
from swarm_field.swarm.arbiter import Arbiter
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
        arbiter: Picks the candidate the robot acts on and holds the robot to its choice. Every
            robot needs its own, since an arbiter remembers its robot's commitment.
    """

    def __init__(
        self,
        robot_id: RobotId,
        behaviors: list[Behavior],
        memory: AgentMemory,
        fields: FieldProvider,
        navigator: Navigator,
        arbiter: Arbiter,
    ) -> None:
        self.robot_id: RobotId = robot_id
        self.behaviors: list[Behavior] = behaviors
        self.memory: AgentMemory = memory
        self.fields: FieldProvider = fields
        self.navigator: Navigator = navigator
        self.arbiter: Arbiter = arbiter
        self._current_goal: str | None = None
        self._current_target: CellId | None = None

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

        # TODO: an arbiter (fixed-priority overrides such as low battery, hysteresis against
        # switching objectives back and forth) instead of simply taking the highest score.
        best = max(candidates, key=lambda c: c.score, default=None)
        outbox = self._messages_to_send(perception.time, best)
        if best is None:
        choice = self.arbiter.choose(candidates, perception.time)
        if choice is None:
            self._current_goal = None
            self._current_target = None
            return Decision(action=Idle(), outbox=self._messages_to_send(perception))

        best = choice.candidate
        self._current_goal = best.source
        action, target = self._action_for(
            best.action, perception.self_state.position, perception.time
        )
        self._current_target = target
        return Decision(action=action, outbox=self._messages_to_send(perception))

    def inspect(self) -> AgentInspection:
        return AgentInspection(
            robot_id=self.robot_id,
            current_goal=self._current_goal,
            belief=self.memory.snapshot(),
        )

    def _action_for(self, intent: Intent, robot: CellId, now: Time) -> tuple[Action, CellId | None]:
        """The action that carries out the winning candidate's intent."""
        if not isinstance(intent, FollowField):
            return intent, None
        field = self.fields.field_for(intent.seeder, robot, now)
        if field is None:
            logger.warning("FollowField couldn't find the corresponding field %s", intent.seeder)
            return Idle(), None
        return self.navigator.step_towards(field, robot)

    @staticmethod
    def _objective_for(candidate: ActionCandidate) -> AgentObjective | None:
        """Return the stable objective associated with a candidate target, if any."""
        if candidate.target is None:
            return None
        return candidate.objective

    def _facts_received(self, inbox: list[Message]) -> list[Observation]:
        """The facts carried by the received ``KNOWLEDGE`` messages of other robots."""
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
        objective = self._objective_for(best) if best is not None else None
        # A target is only meaningful when paired with a stable, typed objective. If a custom
        # behaviour omits its objective, do not publish an ambiguous target claim.
        target = best.target if best is not None and objective is not None else None
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
            if message.kind is MessageKind.KNOWLEDGE:
                if not isinstance(message.payload, (tuple, list)):
                    continue
                for fact in message.payload:
                    if isinstance(fact, ObservationBase):
                        facts.append(cast(Observation, fact))
            elif message.kind is MessageKind.STATUS:
                if isinstance(message.payload, PeerObservation):
                    facts.append(message.payload)
        return facts

    def _messages_to_send(self, perception: Perception) -> list[Message]:
        """The messages to broadcast at this decision point: the robot's status and its memory.

        The status tells peers where the robot is and which target it is heading for, the
        winning candidate's ``target``, so that they can avoid the same target.
        """
        status = PeerObservation(
            cell=perception.self_state.position,
            confidence=1.0,
            observed_at=perception.time,
            source=self.robot_id,
            peer=self.robot_id,
            body=perception.self_state.body,
            target=self._current_target,
        )
        status_message = Message(
            sender=self.robot_id,
            sent_at=perception.time,
            kind=MessageKind.STATUS,
            payload=status,
        )
        memory_message = Message(
            sender=self.robot_id,
            sent_at=perception.time,
            kind=MessageKind.KNOWLEDGE,
            payload=tuple(self.memory.records()),
        )
        return [status_message, memory_message]
