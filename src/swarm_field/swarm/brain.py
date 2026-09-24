"""The composed robot brain (A).

``ComposedAgent`` is the thin orchestrator that implements the ``SwarmAgent`` boundary. At each
decision point it merges what the robot sensed and received into its memory, keeps the
navigator's quadtree in sync with new terrain facts, gathers the scored candidates of its
behaviors and picks the best. Communication and capability gating are added as those parts are
built.
"""

from swarm_field.contracts import (
    AgentInspection,
    Decision,
    Idle,
    Message,
    Observation,
    Perception,
    RobotId,
    TerrainObservation,
    Time,
)
from swarm_field.swarm.behavior import ActionCandidate, Behavior, DecisionContext
from swarm_field.swarm.memory import AgentMemory
from swarm_field.swarm.navigator import Navigator


class ComposedAgent:
    """A robot brain assembled from a memory, a navigator and a list of behaviors.

    Attributes:
        robot_id: The robot this brain drives.
        behaviors: The composable pieces of decision logic, combined at each decision point.
        memory: The robot's facts about the field and its peers.
        navigator: The robot's quadtree, and the potential fields built on it.
    """

    def __init__(
        self,
        robot_id: RobotId,
        behaviors: list[Behavior],
        memory: AgentMemory,
        navigator: Navigator,
    ) -> None:
        self.robot_id: RobotId = robot_id
        self.behaviors: list[Behavior] = behaviors
        self.memory: AgentMemory = memory
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
        self.navigator.update_terrain(terrain_cells)

        ctx = DecisionContext(
            time=perception.time,
            self_state=perception.self_state,
            sensed=perception.sensed,
            memory=self.memory,
            navigator=self.navigator,
        )
        candidates: list[ActionCandidate] = []
        for behavior in self.behaviors:
            proposal = behavior.propose(ctx)
            if isinstance(proposal, ActionCandidate):
                candidates.append(proposal)
            else:
                candidates.extend(proposal)

        # TODO: an arbiter (fixed-priority overrides such as low battery, hysteresis against
        # switching objectives back and forth) instead of simply taking the highest score.
        best = max(candidates, key=lambda c: c.score, default=None)
        outbox = self._messages_to_send(perception.time)
        if best is None:
            self._current_goal = None
            return Decision(action=Idle(), outbox=outbox)

        self._current_goal = best.source
        return Decision(action=best.action, outbox=outbox)

    def inspect(self) -> AgentInspection:
        return AgentInspection(
            robot_id=self.robot_id,
            current_goal=self._current_goal,
            belief=self.memory.snapshot(),
        )

    def _facts_received(self, inbox: list[Message]) -> list[Observation]:
        """The facts carried by the received messages."""
        # TODO: decode the facts of belief-sharing messages. No message kind carries facts
        # yet (MessageKind only has HELLO), so nothing is received for now.
        return []

    def _messages_to_send(self, now: Time) -> list[Message]:
        """The messages to broadcast at this decision point."""
        # TODO: share memory.records() with robots in range, in a belief-sharing message,
        # once that message kind exists. Nothing is sent for now.
        return []
