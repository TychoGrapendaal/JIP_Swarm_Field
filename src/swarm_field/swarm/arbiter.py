"""Objective arbitration — which objective a robot pursues, and when it abandons one.

Every objective owns a potential field (see the field-level design). A behaviour evaluates its
field at the robot's cell and proposes an ``ObjectiveProposal``; the arbiter picks one. Both
terms of a proposal's score are measured in cells, so objectives measured in different things
(a pest, an unexplored tile, a dock) stay comparable:

    score = -potential + urgency

``potential`` is the objective's field value at the robot's cell, so lower is better and the
sign flips. ``urgency`` is an additive per-objective priority that the field itself cannot
express, such as a battery running down. Keeping the two apart is deliberate: the field's
value/distance exchange rate and the objective's priority are separate concerns, and folding
them into one number makes neither tunable on its own.

The arbiter is stateful because hysteresis requires it. New information arrives in bursts at
every encounter, so scores jump, and a robot that always follows the top score drives back and
forth without finishing anything. ``ArbiterConfig`` holds the three defences against that.
"""

from dataclasses import dataclass
from enum import Enum, auto

from swarm_field.contracts import Action, CellId, Time
from swarm_field.swarm.behavior import ActionCandidate


class Objective(Enum):
    """What a robot can currently be trying to do. One potential field per member."""

    TREAT = auto()
    EXPLORE = auto()
    MONITOR = auto()
    VERIFY = auto()
    SHARE = auto()
    RECHARGE = auto()
    STANDBY = auto()


@dataclass(frozen=True)
class ObjectiveProposal(ActionCandidate):
    """One objective's bid at a decision point.

    Attributes:
        objective: The objective this bid is for.
        target: The target cell the objective's field selected, if it has one.
        potential: The field value at the robot's cell, in cells. Lower is better.
        urgency: Additive priority for this objective, in cells. Higher is better.
        distance: Route cost from the robot to ``target``, in cells. The arbiter compares it
            against the distance when the target was taken on, to measure progress.
        is_override: Whether this bid skips comparison entirely (see ``ObjectiveArbiter``).
    """

    objective: Objective
    target: CellId | None
    potential: float
    urgency: float
    distance: float
    is_override: bool = False

    @classmethod
    def create(
        cls,
        objective: Objective,
        action: Action,
        potential: float,
        target: CellId | None = None,
        urgency: float = 0.0,
        distance: float = 0.0,
        is_override: bool = False,
    ) -> "ObjectiveProposal":
        """Build a proposal with ``score`` derived from ``potential`` and ``urgency``."""
        return cls(
            action=action,
            score=urgency - potential,
            source=objective.name.lower(),
            objective=objective,
            target=target,
            potential=potential,
            urgency=urgency,
            distance=distance,
            is_override=is_override,
        )

    def commitment(self) -> tuple[Objective, CellId | None]:
        """The pair identifying what this bid commits the robot to."""
        return (self.objective, self.target)


@dataclass(frozen=True)
class ArbiterConfig:
    """Hysteresis settings: how much better a rival must be, and how soon it may try.

    The required margin is ``floor + commitment_weight * progress``, in cells, where progress
    is the fraction of the original journey already driven.

    A flat margin protects every commitment equally, which is the wrong shape. What deserves
    protecting is the work that would be thrown away, and that grows as the robot approaches
    its target. A robot that has just set off has spent nothing, so redirecting it is cheap
    and the swarm should stay responsive to whatever the last encounter revealed; a robot two
    cells from a pest should need an override to be pulled away. A margin scaled by progress
    has that shape, and a margin scaled by the score does not — the score says how good the
    target is, which is unrelated to how much driving would be wasted.

    Counting progress at all means letting a sunk cost affect a decision, which is not
    rational. It is a deliberate trade of some myopic optimality for stability, and worth
    naming as such rather than dressing up as an optimum. Switching targets within an
    objective is cheaper than switching objectives, so its floor is smaller.

    Attributes:
        objective_floor: Margin for an objective switch at zero progress, in cells.
        target_floor: Margin for a target switch at zero progress, in cells.
        commitment_weight: Extra margin at full progress, in cells.
        min_dwell: How long a commitment is held before any switch is considered.
    """

    objective_floor: float = 5.0
    target_floor: float = 2.5
    commitment_weight: float = 15.0
    min_dwell: Time = 30_000


@dataclass(frozen=True)
class ArbiterChoice:
    """What the arbiter settled on.

    Attributes:
        proposal: The winning bid.
        switched: Whether this is a change from the previous commitment.
    """

    proposal: ObjectiveProposal
    switched: bool


class ObjectiveArbiter:
    """Picks one objective per decision point and holds the robot to it.

    Three rules, applied in order:

    1. **Overrides win.** A proposal marked ``is_override`` skips every comparison. This is for
       situations that must not be traded off against a good target, such as a battery low
       enough that the robot would strand itself. Among several, the highest score wins.
    2. **Commitments are held for ``min_dwell``.** Below that, the current commitment is kept
       if it is still on offer. This is the cheapest and most effective anti-thrash rule.
    3. **Switching needs a margin that grows with progress.** A rival must beat the current
       commitment by the margin described in ``ArbiterConfig``, which starts at a floor and
       rises as the robot closes on its target. Objective switches and target switches have
       different floors; without the second case, a robot keeps its objective and still turns
       around whenever an encounter reveals a marginally better pest.

    Behaviours should keep proposing the currently committed target while it remains valid, so
    that the arbiter can find it and compare against it. A commitment that disappears from the
    proposals is treated as finished, and the arbiter is free to switch immediately.
    """

    def __init__(self, config: ArbiterConfig | None = None) -> None:
        self.config: ArbiterConfig = config or ArbiterConfig()
        self._committed: tuple[Objective, CellId | None] | None = None
        self._committed_at: Time = 0
        self._distance_at_commit: float = 0.0

    @property
    def committed(self) -> tuple[Objective, CellId | None] | None:
        """The current commitment as an (objective, target) pair, or ``None`` if idle."""
        return self._committed

    def choose(
        self,
        proposals: list[ObjectiveProposal],
        now: Time,
    ) -> ArbiterChoice | None:
        """Pick one proposal, or ``None`` when there is nothing to do."""
        if not proposals:
            self._release()
            return None

        override = self._best_override(proposals)
        if override is not None:
            return self._commit(override, now)

        best = max(proposals, key=lambda proposal: proposal.score)
        current = self._find_committed(proposals)
        if current is None:
            return self._commit(best, now)

        if now - self._committed_at < self.config.min_dwell:
            return self._keep(current)
        if best.commitment() == current.commitment():
            return self._keep(current)
        if best.score < current.score + self._margin_against(current, best):
            return self._keep(current)

        return self._commit(best, now)

    def progress(self, current: ObjectiveProposal) -> float:
        """How much of the committed journey is already driven, as a fraction in [0, 1].

        Zero when nothing is known about the original distance, so an unset ``distance``
        leaves the margin at its floor rather than pinning it at either extreme.
        """
        if self._distance_at_commit <= 0.0:
            return 0.0
        driven = self._distance_at_commit - current.distance
        if driven <= 0.0:
            return 0.0
        return min(1.0, driven / self._distance_at_commit)

    def _margin_against(self, current: ObjectiveProposal, rival: ObjectiveProposal) -> float:
        """How much better ``rival`` must be to displace ``current``."""
        if rival.objective == current.objective:
            floor = self.config.target_floor
        else:
            floor = self.config.objective_floor
        return floor + self.config.commitment_weight * self.progress(current)

    def _find_committed(self, proposals: list[ObjectiveProposal]) -> ObjectiveProposal | None:
        """The proposal matching the current commitment, if it is still on offer."""
        if self._committed is None:
            return None
        for proposal in proposals:
            if proposal.commitment() == self._committed:
                return proposal
        return None

    @staticmethod
    def _best_override(proposals: list[ObjectiveProposal]) -> ObjectiveProposal | None:
        """The highest-scoring override, or ``None`` when there is none."""
        overrides: list[ObjectiveProposal] = []
        for proposal in proposals:
            if proposal.is_override:
                overrides.append(proposal)
        if not overrides:
            return None
        return max(overrides, key=lambda proposal: proposal.score)

    def _commit(self, proposal: ObjectiveProposal, now: Time) -> ArbiterChoice:
        switched = proposal.commitment() != self._committed
        self._committed = proposal.commitment()
        if switched:
            self._committed_at = now
            self._distance_at_commit = proposal.distance
        return ArbiterChoice(proposal=proposal, switched=switched)

    @staticmethod
    def _keep(proposal: ObjectiveProposal) -> ArbiterChoice:
        return ArbiterChoice(proposal=proposal, switched=False)

    def _release(self) -> None:
        self._committed = None
        self._distance_at_commit = 0.0