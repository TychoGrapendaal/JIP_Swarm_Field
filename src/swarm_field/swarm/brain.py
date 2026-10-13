"""The composed robot brain (A).

``ComposedAgent`` is the thin orchestrator that implements the ``SwarmAgent`` boundary. At each
decision point it merges what the robot sensed and received into its memory, keeps the field
provider's quadtree in sync with new terrain facts, gathers the scored candidates of its
behaviors, picks the best and turns its intent into one action: the navigator steps along the
field a ``FollowField`` names, and any other intent passes through. It broadcasts its memory and
its status (its cell and the winner's target), and merges what its peers broadcast. Capability
gating is added once it is built.
"""

import logging
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
    PeerObservation,
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
        fields: The robot's quadtree, and the grid potential fields of its seeders. Only the
            winner's field is built, to pick its target and for the navigator to follow.
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
            inbox=perception.inbox,
        )
        candidates: list[ActionCandidate] = []
        for behavior in self.behaviors:
            candidates.extend(behavior.propose(ctx))

        # TODO: an arbiter (fixed-priority overrides such as low battery, hysteresis against
        # switching objectives back and forth) instead of simply taking the highest score.
        best = max(candidates, key=lambda c: c.score, default=None)
        action: Action = Idle()
        self._current_goal = None
        self._current_target = None
        if best is not None:
            self._current_goal = best.source
            self._current_target = best.target
            action = self._action_for(best.action, perception.self_state.position, perception.time)
        return Decision(action=action, outbox=self._messages_to_send(perception))

    def inspect(self) -> AgentInspection:
        return AgentInspection(
            robot_id=self.robot_id,
            current_goal=self._current_goal,
            belief=self.memory.snapshot(),
            current_target=self._current_target,
        )

    def _action_for(self, intent: Intent, robot: CellId, now: Time) -> Action:
        """The action that carries out the winning candidate's intent.

        For a ``FollowField`` this builds the seeder's field, whose target becomes the robot's
        current target, and the navigator steps along it.
        """
        if not isinstance(intent, FollowField):
            return intent
        field = self.fields.field_for(intent.seeder, robot, now)
        if field is None:
            logger.warning("FollowField couldn't find the corresponding field %s", intent.seeder)
            return Idle()
        self._current_target = field.field.target_for(robot)
        return self.navigator.step_towards(field, robot)

    def _facts_received(self, inbox: list[Message]) -> list[Observation]:
        """The facts carried by the received messages: memory snapshots and status reports."""
        facts: list[Observation] = []
        for message in inbox:
            if message.kind is MessageKind.KNOWLEDGE:
                if not isinstance(message.payload, (tuple, list)):
                    continue
                for fact in message.payload:
                    if isinstance(fact, ObservationBase):
                        facts.append(cast(Observation, fact))
            elif message.kind is MessageKind.STATUS:
                if not isinstance(message.payload, PeerObservation):
                    logger.warning(
                        "Skipping a status message from %s with payload %r",
                        message.sender,
                        message.payload,
                    )
                    continue
                facts.append(message.payload)
        return facts

    def _messages_to_send(self, perception: Perception) -> list[Message]:
        """The messages to broadcast at this decision point: the memory and the robot's status.

        The status tells peers where the robot is and which target it is heading for, the
        target of the winner's field, or else the winning candidate's ``target``, so that they
        can avoid the same region.
        """
        knowledge = Message(
            sender=self.robot_id,
            sent_at=perception.time,
            kind=MessageKind.KNOWLEDGE,
            payload=tuple(self.memory.records()),
        )
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
        return [knowledge, status_message]
