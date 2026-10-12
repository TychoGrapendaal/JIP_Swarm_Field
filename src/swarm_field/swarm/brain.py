"""The composed robot brain (A).

``ComposedAgent`` is the thin orchestrator that implements the ``SwarmAgent`` boundary. At each
decision point it merges what the robot sensed and received into its memory, keeps the field
provider's quadtree in sync with new terrain facts, gathers the scored candidates of its
behaviors, lets its arbiter pick one (or takes the highest score without one) and turns its
intent into one action: the navigator steps along the field a ``FollowField`` names, and any
other intent passes through. The robot broadcasts its memory in a ``KNOWLEDGE`` message and
merges the facts of the ``KNOWLEDGE`` messages it receives. Capability gating is added as that
part is built.
"""

import logging

from swarm_field.contracts import (
    Action,
    AgentInspection,
    CellId,
    Decision,
    Idle,
    Message,
    MessageKind,
    Observation,
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
        arbiter: Picks the candidate the robot acts on and holds the robot to its choice.
            Without one, the brain takes the highest-scoring candidate outright, with no
            overrides or hysteresis.
    """

    def __init__(
        self,
        robot_id: RobotId,
        behaviors: list[Behavior],
        memory: AgentMemory,
        fields: FieldProvider,
        navigator: Navigator,
        arbiter: Arbiter | None = None,
    ) -> None:
        self.robot_id: RobotId = robot_id
        self.behaviors: list[Behavior] = behaviors
        self.memory: AgentMemory = memory
        self.fields: FieldProvider = fields
        self.navigator: Navigator = navigator
        self.arbiter: Arbiter | None = arbiter
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

        best = self._select(candidates, perception.time)
        outbox = self._messages_to_send(perception.time)
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

    def _select(self, candidates: list[ActionCandidate], now: Time) -> ActionCandidate | None:
        """The candidate to act on: the arbiter's choice, or the highest score without one."""
        if self.arbiter is None:
            if not candidates:
                return None
            return max(candidates, key=lambda candidate: candidate.score)
        choice = self.arbiter.choose(candidates, now)
        if choice is None:
            return None
        return choice.candidate

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
        """The facts carried by the received ``KNOWLEDGE`` messages of other robots."""
        facts: list[Observation] = []
        for message in inbox:
            if message.kind is not MessageKind.KNOWLEDGE or message.sender == self.robot_id:
                continue
            if not isinstance(message.payload, tuple):
                logger.warning("Ignoring KNOWLEDGE message with payload %r", message.payload)
                continue
            facts.extend(fact for fact in message.payload if isinstance(fact, Observation))
        return facts

    def _messages_to_send(self, now: Time) -> list[Message]:
        """The messages to broadcast at this decision point: the robot's memory, if any."""
        facts = self.memory.snapshot()
        if not facts:
            return []
        return [
            Message(sender=self.robot_id, sent_at=now, kind=MessageKind.KNOWLEDGE, payload=facts)
        ]
