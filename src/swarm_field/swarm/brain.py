"""The composed robot brain (A).

``ComposedAgent`` is the thin orchestrator that implements the ``SwarmAgent`` boundary: it
holds a list of behaviors, gathers their scored candidates each tick and picks the best.
Memory, communication and capability gating are added as those parts are built.
"""

from swarm_field.contracts import AgentInspection, Decision, Idle, Perception, RobotId
from swarm_field.swarm.behavior import ActionCandidate, Behavior, DecisionContext


class ComposedAgent:
    """A robot brain assembled from a list of behaviors.

    Attributes:
        robot_id: The robot this brain drives.
        behaviors: The composable pieces of decision logic, combined each tick.
    """

    def __init__(self, robot_id: RobotId, behaviors: list[Behavior]) -> None:
        self.robot_id: RobotId = robot_id
        self.behaviors: list[Behavior] = behaviors
        self._current_goal: str | None = None

    def decide(self, perception: Perception) -> Decision:
        ctx = DecisionContext(
            tick=perception.tick,
            self_state=perception.self_state,
            sensed=perception.sensed,
        )
        candidates: list[ActionCandidate] = []
        for behavior in self.behaviors:
            proposal = behavior.propose(ctx)
            if isinstance(proposal, ActionCandidate):
                candidates.append(proposal)
            else:
                candidates.extend(proposal)

        best = max(candidates, key=lambda c: c.score, default=None)
        if best is None:
            self._current_goal = None
            return Decision(action=Idle())

        self._current_goal = best.source
        return Decision(action=best.action)

    def inspect(self) -> AgentInspection:
        belief: dict[str, object] = {}  # no memory model yet
        return AgentInspection(
            robot_id=self.robot_id,
            current_goal=self._current_goal,
            belief=belief,
        )
