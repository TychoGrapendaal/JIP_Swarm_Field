"""The composed robot brain (A).

``ComposedAgent`` is the thin orchestrator that implements the ``SwarmAgent`` boundary: it
holds a list of behaviors, gathers their scored candidates at each decision point and picks
the best. Memory, communication and capability gating are added as those parts are built.
"""

from swarm_field.contracts import AgentInspection, Decision, Idle, Perception, RobotId
from swarm_field.swarm.arbiter import ObjectiveArbiter, ObjectiveProposal
from swarm_field.swarm.behavior import ActionCandidate, Behavior, DecisionContext


class ComposedAgent:
    """A robot brain assembled from a list of behaviors.

    Attributes:
        robot_id: The robot this brain drives.
        behaviors: The composable pieces of decision logic, combined at each decision point.
        arbiter: Chooses between objective proposals and holds the robot to its choice. When
            absent, the brain falls back to taking the highest-scoring candidate outright,
            which has no hysteresis and is kept for behaviors that do not name an objective.
    """

    def __init__(
        self,
        robot_id: RobotId,
        behaviors: list[Behavior],
        arbiter: ObjectiveArbiter | None = None,
    ) -> None:
        self.robot_id: RobotId = robot_id
        self.behaviors: list[Behavior] = behaviors
        self.arbiter: ObjectiveArbiter | None = arbiter
        self._current_goal: str | None = None

    def decide(self, perception: Perception) -> Decision:
        ctx = DecisionContext(
            time=perception.time,
            self_state=perception.self_state,
            sensed=perception.sensed,
        )
        candidates = self._collect(ctx)

        best = self._select(candidates, perception.time)
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

    def _collect(self, ctx: DecisionContext) -> list[ActionCandidate]:
        """Gather every candidate the behaviors propose at this decision point."""
        candidates: list[ActionCandidate] = []
        for behavior in self.behaviors:
            proposal = behavior.propose(ctx)
            if isinstance(proposal, ActionCandidate):
                candidates.append(proposal)
            else:
                candidates.extend(proposal)
        return candidates

    def _select(self, candidates: list[ActionCandidate], now: int) -> ActionCandidate | None:
        """Reduce the candidates to one, through the arbiter when there is one."""
        if not candidates:
            return None
        if self.arbiter is None:
            return max(candidates, key=lambda candidate: candidate.score)

        proposals: list[ObjectiveProposal] = []
        for candidate in candidates:
            if isinstance(candidate, ObjectiveProposal):
                proposals.append(candidate)

        choice = self.arbiter.choose(proposals, now)
        if choice is None:
            return None
        return choice.proposal