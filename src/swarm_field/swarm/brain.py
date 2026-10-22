"""The composed robot brain.

``ComposedAgent`` implements the ``SwarmAgent`` boundary. At each decision point it merges local
and received observations into memory, updates the field provider with new terrain facts, gathers
behavior candidates, and lets its arbiter select one. The winning intent is resolved into a concrete
action, then checked against the robot body's capabilities. The agent broadcasts its status and
knowledge snapshot and merges the status and knowledge received from peers.
"""

import logging
from dataclasses import replace

from swarm_field.contracts import (
    Action,
    AgentInspection,
    AgentObjective,
    CellId,
    Decision,
    Idle,
    Message,
    MessageKind,
    Move,
    PeerObservation,
    Perception,
    PestId,
    RobotId,
    Time,
)
from swarm_field.swarm.arbiter import Arbiter, UtilityArbiter
from swarm_field.swarm.behavior import (
    ActionCandidate,
    Behavior,
    DecisionContext,
    FollowField,
    Intent,
)
from swarm_field.swarm.capabilities import CapabilitySet, capabilities_for
from swarm_field.swarm.field import GridPotentialField
from swarm_field.swarm.memory import AgentMemory
from swarm_field.swarm.navigator import Navigator
from swarm_field.swarm.provider import FieldProvider
from swarm_field.swarm.sharing import KnowledgeExchange, SharingConfig
from swarm_field.swarm.tether import TetherConfig, tethered_action

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
        tether: If set, every step is checked to keep the robot in radio contact with a peer
            (see ``swarm.tether``); ``None`` leaves steps unchecked.
    """

    def __init__(
        self,
        robot_id: RobotId,
        behaviors: list[Behavior],
        memory: AgentMemory,
        fields: FieldProvider,
        navigator: Navigator,
        arbiter: Arbiter | None = None,
        capabilities: CapabilitySet | None = None,
        sharing: SharingConfig | None = None,
        tether: TetherConfig | None = None,
    ) -> None:
        self.robot_id: RobotId = robot_id
        self.behaviors: list[Behavior] = behaviors
        self.memory: AgentMemory = memory
        self.fields: FieldProvider = fields
        self.navigator: Navigator = navigator
        self.capabilities: CapabilitySet | None = capabilities
        self.arbiter: Arbiter = arbiter if arbiter is not None else UtilityArbiter()
        self.exchange: KnowledgeExchange = KnowledgeExchange(memory, sharing)
        self.tether: TetherConfig | None = tether
        self._current_goal: str | None = None
        self._current_target: CellId | None = None
        self._current_objective: AgentObjective | None = None

    def decide(self, perception: Perception) -> Decision:
        version = self.memory.version
        self.memory.ingest(perception)
        self.exchange.receive(perception.inbox)
        self.fields.update_terrain(self.memory.terrain_stored_after(version))

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
                action_types=(canonical_capabilities.action_types & self.capabilities.action_types)
                | {Idle},
            )

        ctx = DecisionContext(
            time=perception.time,
            self_state=perception.self_state,
            sensed=perception.sensed,
            memory=self.memory,
            inbox=perception.inbox,
            capabilities=capabilities,
        )
        candidates: list[ActionCandidate] = []
        for behavior in self.behaviors:
            for candidate in behavior.propose(ctx):
                if self._candidate_is_allowed(candidate, capabilities):
                    candidates.append(candidate)

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
        # Peers learn the cell a robot is heading for; a pest known only by its id has no
        # reliable cell to report, so it is left out until the reports carry pest ids too.
        self._current_target = None
        if not isinstance(best.target, PestId):
            self._current_target = best.target
        if field_target is not None:
            self._current_target = field_target
        if self.tether is not None:
            action = tethered_action(
                action,
                perception.self_state.position,
                self._current_target,
                self.memory,
                perception.time,
                self.tether,
                self._followed_field(best.action),
            )

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
            self._current_target = None
            self._current_objective = None

        return Decision(action=action, outbox=self._messages_to_send(perception))

    @staticmethod
    def _candidate_is_allowed(candidate: ActionCandidate, capabilities: CapabilitySet) -> bool:
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

    def _followed_field(self, intent: Intent) -> GridPotentialField | None:
        """The field the winning intent follows, as just built, or ``None`` for no field."""
        if not isinstance(intent, FollowField):
            return None
        seeded = self.fields.cached_field(intent.seeder)
        if seeded is None:
            return None
        return seeded.field

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
            battery=perception.self_state.battery,
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
