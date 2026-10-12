"""Arbitration: which candidate a robot acts on, and when it abandons what it is doing.

Every behaviour bids ``ActionCandidate``s on the shared ``[0, 1]`` score scale (see
``swarm_field.swarm.scoring``); the arbiter picks one. It sees only the bids and the time,
never memory, the robot's state or which behaviours exist, so any list of behaviours works with
it. A bid's ``source`` names its objective and its ``target`` the cell it commits to; both are
compared for equality only.

The arbiter is stateful because hysteresis requires it. New information arrives in bursts at
every encounter, so scores jump, and a robot that always follows the top score drives back and
forth without finishing anything. ``ArbiterConfig`` holds the defences against that.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isnan
from typing import Protocol

from swarm_field.contracts import CellId, Time, TimeUnit
from swarm_field.swarm.behavior import ActionCandidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArbiterChoice:
    """What the arbiter settled on.

    Attributes:
        candidate: The winning bid.
        switched: Whether the robot abandoned a commitment that was still on offer. Taking on
            a first commitment, or a new target once the old one is done, is not a switch, so
            switches per completed job can be counted from this.
    """

    candidate: ActionCandidate
    switched: bool


class Arbiter(Protocol):
    """Picks the one candidate a robot acts on at a decision point."""

    def choose(self, candidates: Sequence[ActionCandidate], now: Time) -> ArbiterChoice | None:
        """The winning candidate, or ``None`` when there is nothing to do."""
        ...


@dataclass(frozen=True)
class ArbiterConfig:
    """Overrides and hysteresis, in units of the shared score scale.

    A bid scoring at least ``override_threshold`` skips every comparison. Field-based scores
    stay below it (a target of value 1 under the robot scores about 0.73), so the band above is
    left to needs a behaviour scores directly, such as a battery about to run out.

    Otherwise a rival must beat the current commitment by ``floor + effort_weight * effort``.
    Effort is the time spent on the current target, as a fraction of ``effort_horizon``: it is
    what switching throws away. It is measured per target rather than per objective, because
    leaving an objective loses only the work towards the target being pursued; the cells a
    robot has already explored stay explored. Counting effort lets a sunk cost affect a
    decision, a deliberate trade of some myopic optimality for stability. Switching targets
    within an objective is cheaper than switching objectives, so its floor is smaller.

    The floors and the weight are the cell margins of the first arbiter (5, 2.5 and 15 cells)
    converted at the steepest point of the logistic, where one cell is worth
    ``1 / (4 * 10)`` = 0.025.

    Attributes:
        override_threshold: Score from which a bid overrides everything else.
        objective_floor: Margin for switching to another objective, with no effort spent.
        target_floor: Margin for switching to another target of the same objective, with no
            effort spent.
        effort_weight: Extra margin once ``effort_horizon`` has been spent on the target.
        effort_horizon: Time on a target after which its effort counts in full.
        min_dwell: How long a new objective is held before any switch is considered.
    """

    override_threshold: float = 0.9
    objective_floor: float = 0.125
    target_floor: float = 0.0625
    effort_weight: float = 0.375
    effort_horizon: Time = 5 * TimeUnit.MINUTE
    min_dwell: Time = 30 * TimeUnit.SECOND


@dataclass(frozen=True)
class Commitment:
    """What the robot is committed to, and since when.

    Attributes:
        source: The objective: the name of the behaviour whose bid won.
        target: The cell that bid committed the robot to, or ``None``.
        objective_since: When the robot took on this objective.
        target_since: When the robot took on this target.
    """

    source: str
    target: CellId | None
    objective_since: Time
    target_since: Time

    def is_for(self, candidate: ActionCandidate) -> bool:
        """Whether ``candidate`` bids for this commitment."""
        return (candidate.source, candidate.target) == (self.source, self.target)

    def effort(self, now: Time, horizon: Time) -> float:
        """The time spent on the target so far, as a fraction of ``horizon`` in [0, 1]."""
        if horizon <= 0:
            return 1.0
        return min(1.0, max(0, now - self.target_since) / horizon)

    def into_retargeted(self, candidate: ActionCandidate, now: Time) -> "Commitment":
        """This commitment taken on to ``candidate``'s target; a new objective starts afresh."""
        objective_since = self.objective_since
        if candidate.source != self.source:
            objective_since = now
        return Commitment(candidate.source, candidate.target, objective_since, now)


class UtilityArbiter:
    """Picks one candidate per decision point and holds the robot to its choice.

    Five rules, applied in order:

    1. **Overrides win.** A bid scoring at least ``override_threshold`` skips every comparison;
       among several, the highest score wins. This is for needs that must not be traded off
       against a good target, such as a battery low enough to strand the robot.
    2. **A finished objective frees the robot.** When no candidate bids for the committed
       objective any more, the highest score wins.
    3. **A finished target keeps the objective.** When the committed target is no longer on
       offer but its objective still bids, the robot takes on the objective's best target and
       stays on the objective, as an explorer moves on to the next unexplored cell.
    4. **A new objective is held for ``min_dwell``.** Before that, no rival is considered.
    5. **Switching needs a margin that grows with effort** (see ``ArbiterConfig``), with a
       smaller floor when the rival bids for the same objective.

    Scores outside ``[0, 1]`` are clamped, and bids without a number for a score are dropped,
    each with a warning.

    Attributes:
        config: The override threshold and the hysteresis settings.
        commitment: What the robot is committed to, or ``None`` if it has nothing to do.
    """

    def __init__(self, config: ArbiterConfig | None = None) -> None:
        self.config: ArbiterConfig = ArbiterConfig()
        if config is not None:
            self.config = config
        self.commitment: Commitment | None = None

    def choose(self, candidates: Sequence[ActionCandidate], now: Time) -> ArbiterChoice | None:
        bids = _on_scale(candidates)
        if not bids:
            self.commitment = None
            return None

        best = _highest_scoring(bids)
        overrides = [bid for bid in bids if bid.score >= self.config.override_threshold]
        if overrides:
            return self._take_on(_highest_scoring(overrides), now, self._is_on_offer(bids))

        current = self._current_bid(bids, now)
        if current is None:
            return self._take_on(best, now, abandons=False)
        if best is current or self._holds_against(current, best, now):
            return ArbiterChoice(current, switched=False)
        return self._take_on(best, now, abandons=True)

    def _current_bid(self, bids: Sequence[ActionCandidate], now: Time) -> ActionCandidate | None:
        """The bid for the robot's commitment, after moving on from a finished target.

        ``None`` when the robot has no commitment or its objective is no longer bid for.
        """
        commitment = self.commitment
        if commitment is None:
            return None
        same_objective: list[ActionCandidate] = []
        for bid in bids:
            if commitment.is_for(bid):
                return bid
            if bid.source == commitment.source:
                same_objective.append(bid)
        if not same_objective:
            return None
        next_target = _highest_scoring(same_objective)
        self.commitment = commitment.into_retargeted(next_target, now)
        return next_target

    def _holds_against(self, current: ActionCandidate, rival: ActionCandidate, now: Time) -> bool:
        """Whether the robot keeps ``current`` rather than switching to ``rival``."""
        commitment = self.commitment
        if commitment is None:
            return False
        if now - commitment.objective_since < self.config.min_dwell:
            return True
        floor = self.config.objective_floor
        if rival.source == current.source:
            floor = self.config.target_floor
        effort = commitment.effort(now, self.config.effort_horizon)
        margin = floor + self.config.effort_weight * effort
        return rival.score < current.score + margin

    def _is_on_offer(self, bids: Sequence[ActionCandidate]) -> bool:
        """Whether some bid is still for the robot's commitment."""
        commitment = self.commitment
        if commitment is None:
            return False
        return any(commitment.is_for(bid) for bid in bids)

    def _take_on(self, candidate: ActionCandidate, now: Time, abandons: bool) -> ArbiterChoice:
        """Commit to ``candidate``; ``abandons`` says whether the old commitment was on offer."""
        commitment = self.commitment
        if commitment is None:
            self.commitment = Commitment(candidate.source, candidate.target, now, now)
            return ArbiterChoice(candidate, switched=False)
        if commitment.is_for(candidate):
            return ArbiterChoice(candidate, switched=False)
        self.commitment = commitment.into_retargeted(candidate, now)
        return ArbiterChoice(candidate, switched=abandons)


def _highest_scoring(bids: Sequence[ActionCandidate]) -> ActionCandidate:
    """The highest-scoring bid; ties go to the earliest."""
    return max(bids, key=lambda bid: bid.score)


def _on_scale(candidates: Sequence[ActionCandidate]) -> list[ActionCandidate]:
    """The candidates with their scores on the shared ``[0, 1]`` scale."""
    bids: list[ActionCandidate] = []
    for candidate in candidates:
        if isnan(candidate.score):
            logger.warning("Dropping %s's bid: its score is not a number", candidate.source)
            continue
        if 0.0 <= candidate.score <= 1.0:
            bids.append(candidate)
            continue
        clamped = min(1.0, max(0.0, candidate.score))
        logger.warning(
            "%s bid a score of %s, outside [0, 1]; clamped to %s",
            candidate.source,
            candidate.score,
            clamped,
        )
        bids.append(replace(candidate, score=clamped))
    return bids
