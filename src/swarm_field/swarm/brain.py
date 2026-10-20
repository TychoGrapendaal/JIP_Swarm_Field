"""The composed robot brain (A).

``ComposedAgent`` is the thin orchestrator that implements the ``SwarmAgent`` boundary. At each
decision point it merges what the robot sensed and received into its memory, keeps the field
provider's quadtree in sync with new terrain facts, gathers the scored candidates of its
behaviors, lets its arbiter pick one and turns its intent into one action: the navigator steps
along the field a ``FollowField`` names, and any other intent passes through. It broadcasts its
status (its cell, and the winner's target and objective) and, depending on its
``KnowledgeSharing``, its memory, and merges what its peers broadcast. Capability gating is added
once it is built.
"""

import logging
from enum import Enum, auto
from typing import cast

from swarm_field.contracts import (
    Action,
    AgentInspection,
    AgentObjective,
    Beacon,
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
from swarm_field.swarm.arbiter import Arbiter
from swarm_field.swarm.beacon import BeaconConfig, treat_beacon
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


class KnowledgeSharing(Enum):
    """Which memory facts a robot broadcasts in its ``KNOWLEDGE`` message.

    ``EVERYTHING`` sends every fact: terrain, infestations and the relayed peers.
    ``INFESTATIONS`` sends only infestation facts, including cells observed to be clean.
    ``NOTHING`` sends no ``KNOWLEDGE`` message at all. The robot's status is broadcast at every
    level, so peers in range always learn where it is.
    """

    EVERYTHING = auto()
    INFESTATIONS = auto()
    NOTHING = auto()


class ComposedAgent:
    """A robot brain assembled from a memory, a field provider, a navigator and behaviors.

    Attributes:
        robot_id: The robot this brain drives.
        behaviors: The composable pieces of decision logic, combined at each decision point.
        memory: The robot's facts about the field and its peers.
        fields: The robot's quadtree, and its objective fields by name. Only the
            winner's field is built, to pick its target and for the navigator to follow.
        navigator: Moves the robot along the winning candidate's field.
        arbiter: Picks the candidate the robot acts on and holds the robot to its choice. Every
            robot needs its own, since an arbiter remembers its robot's commitment.
        sharing: Which memory facts the robot broadcasts to its peers.
        beacons: When heard beacons and the robot's own sightings count, or ``None`` for a robot
            that neither keeps nor broadcasts beacons.
    """

    def __init__(
        self,
        robot_id: RobotId,
        behaviors: list[Behavior],
        memory: AgentMemory,
        fields: FieldProvider,
        navigator: Navigator,
        arbiter: Arbiter,
        sharing: KnowledgeSharing = KnowledgeSharing.EVERYTHING,
        beacons: BeaconConfig | None = None,
    ) -> None:
        self.robot_id: RobotId = robot_id
        self.behaviors: list[Behavior] = behaviors
        self.memory: AgentMemory = memory
        self.fields: FieldProvider = fields
        self.navigator: Navigator = navigator
        self.arbiter: Arbiter = arbiter
        self.sharing: KnowledgeSharing = sharing
        self.beacons: BeaconConfig | None = beacons
        self._current_goal: str | None = None
        self._current_target: CellId | None = None
        self._current_objective: AgentObjective | None = None

    def decide(self, perception: Perception) -> Decision:
        received = self._facts_received(perception.inbox)
        self.memory.ingest(perception)
        self.memory.merge(received)
        if self.beacons is not None:
            self._hear_beacons(perception.inbox)
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

        choice = self.arbiter.choose(candidates, perception.time)
        if choice is None:
            self._current_goal = None
            self._current_target = None
            self._current_objective = None
            return Decision(action=Idle(), outbox=self._messages_to_send(perception))

        best = choice.candidate
        self._current_goal = best.source
        self._current_objective = best.objective
        action, field_target = self._action_for(
            best.action, perception.self_state.position, perception.time
        )
        self._current_target = best.target
        if field_target is not None:
            self._current_target = field_target
        return Decision(action=action, outbox=self._messages_to_send(perception))

    def inspect(self) -> AgentInspection:
        return AgentInspection(
            robot_id=self.robot_id,
            current_goal=self._current_goal,
            belief=self.memory.snapshot(),
            current_target=self._current_target,
        )

    def _action_for(self, intent: Intent, robot: CellId, now: Time) -> tuple[Action, CellId | None]:
        """The action that carries out the winning candidate's intent, and the field's target.

        For a ``FollowField`` this builds the seeder's field and the navigator steps along it;
        the target is the one the field leads to, or ``None`` for any other intent.
        """
        if not isinstance(intent, FollowField):
            return intent, None
        field = self.fields.field_for(intent.seeder, robot, now)
        if field is None:
            logger.warning("FollowField couldn't find the corresponding field %s", intent.seeder)
            return Idle(), None
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

    def _hear_beacons(self, inbox: list[Message]) -> None:
        """Store the beacons of every received beacon message."""
        for message in inbox:
            if message.kind is not MessageKind.BEACON:
                continue
            beacons: list[Beacon] = []
            if isinstance(message.payload, tuple):
                for item in cast(tuple[object, ...], message.payload):
                    if isinstance(item, Beacon):
                        beacons.append(item)
            if not beacons:
                logger.warning(
                    "Skipping a beacon message from %s with payload %r",
                    message.sender,
                    message.payload,
                )
                continue
            self.memory.hear_beacons(message.sender, message.sent_at, beacons)

    def _messages_to_send(self, perception: Perception) -> list[Message]:
        """The messages to broadcast at this decision point: the robot's status and its memory.

        The status tells peers where the robot is, which target it is heading for (the target
        of the winner's field, or else the winning candidate's ``target``) and for which
        objective, so that they can avoid the same region. The memory message carries the facts
        ``sharing`` selects, and is left out at ``KnowledgeSharing.NOTHING``. With ``beacons``
        the robot also broadcasts its TREAT beacon, if it has one.
        """
        status = PeerObservation(
            cell=perception.self_state.position,
            confidence=1.0,
            observed_at=perception.time,
            source=self.robot_id,
            peer=self.robot_id,
            body=perception.self_state.body,
            target=self._current_target,
            objective=self._current_objective,
        )
        status_message = Message(
            sender=self.robot_id,
            sent_at=perception.time,
            kind=MessageKind.STATUS,
            payload=status,
        )
        messages = [status_message]
        knowledge = self._knowledge_message(perception.time)
        if knowledge is not None:
            messages.append(knowledge)
        beacon = self._beacon_message(perception)
        if beacon is not None:
            messages.append(beacon)
        return messages

    def _knowledge_message(self, now: Time) -> Message | None:
        """The memory facts ``sharing`` selects, or ``None`` at ``KnowledgeSharing.NOTHING``."""
        match self.sharing:
            case KnowledgeSharing.EVERYTHING:
                facts: tuple[Observation, ...] = tuple(self.memory.records())
            case KnowledgeSharing.INFESTATIONS:
                facts = tuple(self.memory.infestations())
            case KnowledgeSharing.NOTHING:
                return None
        return Message(sender=self.robot_id, sent_at=now, kind=MessageKind.KNOWLEDGE, payload=facts)

    def _beacon_message(self, perception: Perception) -> Message | None:
        """The robot's TREAT beacon, or ``None`` without ``beacons`` or without a beacon."""
        if self.beacons is None:
            return None
        beacon = treat_beacon(
            self.memory,
            self.robot_id,
            perception.self_state.position,
            perception.time,
            self.beacons,
        )
        if beacon is None:
            return None
        return Message(
            sender=self.robot_id,
            sent_at=perception.time,
            kind=MessageKind.BEACON,
            payload=(beacon,),
        )
