"""The composed robot brain.

``ComposedAgent`` implements the ``SwarmAgent`` boundary. At each decision point it merges local
and received observations into memory, updates the field provider with new terrain facts, gathers
behavior candidates, and lets its arbiter select one. The winning intent is resolved into a concrete
action, then checked against the robot body's capabilities. The agent broadcasts its status and
knowledge snapshot and merges the status and knowledge received from peers. A robot of the hop
method also has a ``BeaconLink``, which hears, places and sends its beacons.
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
from swarm_field.swarm.guidance import Guidance, Heading
from swarm_field.swarm.hop.link import BeaconLink
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
        fields: The robot's quadtree, and its objective fields by name. Only the
            winner's field is built, to pick its target and for the navigator to follow.
        navigator: Moves the robot along the winning candidate's field.
        arbiter: Picks the candidate the robot acts on and holds the robot to its choice. Every
            robot needs its own, since an arbiter remembers its robot's commitment.
        exchange: Merges what the robot hears into its memory, and picks what it shares.
        beacon_link: The robot's beacons, for a robot of the hop method, or ``None`` for a
            robot that neither keeps nor broadcasts beacons.
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
        beacon_link: BeaconLink | None = None,
    ) -> None:
        self.robot_id: RobotId = robot_id
        self.behaviors: list[Behavior] = behaviors
        self.memory: AgentMemory = memory
        self.fields: FieldProvider = fields
        self.navigator: Navigator = navigator
        self.capabilities: CapabilitySet | None = capabilities
        self.arbiter: Arbiter = arbiter if arbiter is not None else UtilityArbiter()
        self.exchange: KnowledgeExchange = KnowledgeExchange(memory, sharing)
        self.beacon_link: BeaconLink | None = beacon_link
        self._current_goal: str | None = None
        self._current_target: CellId | None = None
        self._current_objective: AgentObjective | None = None
        self._heading: Heading | None = None
        self._guidance: Guidance | None = None

    def decide(self, perception: Perception) -> Decision:
        version = self.memory.version
        self.memory.ingest(perception)
        self.exchange.receive(perception.inbox)
        if self.beacon_link is not None:
            self.beacon_link.perceive(perception)
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
        self._current_goal = None
        self._current_target = None
        self._current_objective = None
        self._heading = None
        self._guidance = None
        action: Action = Idle()
        if choice is not None:
            best = choice.candidate
            self._current_goal = best.source
            self._current_objective = best.objective
            self._current_target = best.target
            action, field_target = self._action_for(
                best.action, perception.self_state.position, perception.time
            )
            if field_target is not None:
                self._current_target = field_target

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
        if self.beacon_link is not None:
            self.beacon_link.after_step(
                self._heading, action, perception.self_state.position, perception.time
            )
        return Decision(action=action, outbox=self._messages_to_send(perception))

    @property
    def guidance(self) -> Guidance | None:
        """The field's guidance the robot followed at its last decision, or ``None``."""
        return self._guidance

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
        the target is the one the field leads to, or ``None`` for any other intent. A heading
        is remembered in ``_heading`` for the ``BeaconLink``.
        """
        if not isinstance(intent, FollowField):
            return intent, None
        field = self.fields.field_for(intent.seeder, robot, now)
        if field is None:
            logger.warning("Field %s has nothing to guide the robot to; idling", intent.seeder)
            return Idle(), None
        self._guidance = field
        if isinstance(field, Heading):
            self._heading = field
        return self.navigator.step_towards(field, robot)

    def _messages_to_send(self, perception: Perception) -> list[Message]:
        """The messages to broadcast at this decision point: the robot's status and knowledge.

        The status tells peers where the robot is, which target it is heading for (the target
        of the winner's field, or else the winning candidate's ``target``) and for which
        objective, so that they can avoid the same region. The knowledge is what the exchange
        shares of the robot's memory, if anything. With a ``beacon_link`` the robot also
        broadcasts its TREAT beacon, if it has one.
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
        if self.beacon_link is not None:
            beacon = self.beacon_link.beacon_message(perception)
            if beacon is not None:
                messages.append(beacon)
        return messages
