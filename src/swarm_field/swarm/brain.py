"""The composed robot brain (A).

``ComposedAgent`` is the thin orchestrator that implements the ``SwarmAgent`` boundary. At each
decision point it merges what the robot sensed and received into its memory, keeps the field
provider's quadtree in sync with new terrain facts, gathers the scored candidates of its
behaviors, lets its arbiter pick one and turns its intent into one action: the navigator steps
along the field a ``FollowField`` names, and any other intent passes through. It broadcasts its
status (its cell, and the winner's target and objective) and what its ``KnowledgeExchange``
shares of its memory, and merges what its peers broadcast. Capability gating is added once it is
built.
"""

import logging

from swarm_field.contracts import (
    Action,
    AgentInspection,
    AgentObjective,
    CellId,
    Decision,
    Idle,
    Message,
    MessageKind,
    PeerObservation,
    Perception,
    RobotId,
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
from swarm_field.swarm.sharing import KnowledgeExchange, SharingConfig

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
        arbiter: Picks the candidate the robot acts on and holds the robot to its choice. Every
            robot needs its own, since an arbiter remembers its robot's commitment.
        exchange: Merges what the robot hears into its memory, and picks what it shares.
    """

    def __init__(
        self,
        robot_id: RobotId,
        behaviors: list[Behavior],
        memory: AgentMemory,
        fields: FieldProvider,
        navigator: Navigator,
        arbiter: Arbiter,
        sharing: SharingConfig | None = None,
    ) -> None:
        self.robot_id: RobotId = robot_id
        self.behaviors: list[Behavior] = behaviors
        self.memory: AgentMemory = memory
        self.fields: FieldProvider = fields
        self.navigator: Navigator = navigator
        self.arbiter: Arbiter = arbiter
        self.exchange: KnowledgeExchange = KnowledgeExchange(memory, sharing)
        self._current_goal: str | None = None
        self._current_target: CellId | None = None
        self._current_objective: AgentObjective | None = None

    def decide(self, perception: Perception) -> Decision:
        version = self.memory.version
        self.memory.ingest(perception)
        self.exchange.receive(perception.inbox)
        self.fields.update_terrain(self.memory.terrain_stored_after(version))

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

    def _messages_to_send(self, perception: Perception) -> list[Message]:
        """The messages to broadcast at this decision point: the robot's status and knowledge.

        The status tells peers where the robot is, which target it is heading for (the target
        of the winner's field, or else the winning candidate's ``target``) and for which
        objective, so that they can avoid the same region. The knowledge is what the exchange
        shares of the robot's memory, if anything.
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
        knowledge = self.exchange.knowledge_to_send()
        if knowledge is not None:
            messages.append(
                Message(
                    sender=self.robot_id,
                    sent_at=perception.time,
                    kind=MessageKind.KNOWLEDGE,
                    payload=knowledge,
                )
            )
        return messages
