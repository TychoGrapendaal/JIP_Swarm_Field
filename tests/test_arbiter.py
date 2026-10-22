"""Tests for the arbiter, with mock behaviours bidding fixed candidates."""

from swarm_field.contracts import (
    BodyType,
    Charge,
    FieldGeometry,
    Perception,
    PestId,
    RobotSelfState,
    Treat,
)
from swarm_field.swarm import (
    ActionCandidate,
    AgentMemory,
    ArbiterConfig,
    Behavior,
    ComposedAgent,
    DecisionContext,
    FieldProvider,
    FollowField,
    Navigator,
    UtilityArbiter,
)

_HORIZON = 300_000
"""Five minutes on a target makes its effort count in full, in milliseconds."""


def _bid(source: str, score: float, target: tuple[int, int] | None = (1, 1)) -> ActionCandidate:
    return ActionCandidate(FollowField(source), score, source, target)


def _arbiter(min_dwell: int = 0) -> UtilityArbiter:
    """0.125 to change objective, 0.0625 to change target, plus up to 0.375 with effort."""
    return UtilityArbiter(
        ArbiterConfig(
            override_threshold=0.9,
            objective_floor=0.125,
            target_floor=0.0625,
            effort_weight=0.375,
            effort_horizon=_HORIZON,
            min_dwell=min_dwell,
        )
    )


def test_highest_score_wins_when_uncommitted() -> None:
    choice = _arbiter().choose([_bid("explore", 0.3), _bid("treat", 0.6)], now=0)

    assert choice is not None
    assert choice.candidate.source == "treat"
    assert not choice.switched


def test_a_rival_must_beat_the_commitment_by_the_objective_floor() -> None:
    arbiter = _arbiter()
    treat = _bid("treat", 0.4)
    arbiter.choose([treat], now=0)

    # Right after committing there is no effort yet, so the margin is the 0.125 floor.
    kept = arbiter.choose([treat, _bid("explore", 0.5, (9, 9))], now=0)
    switched = arbiter.choose([treat, _bid("explore", 0.6, (9, 9))], now=0)

    assert kept is not None
    assert kept.candidate == treat
    assert switched is not None
    assert (switched.candidate.source, switched.switched) == ("explore", True)


def test_another_target_of_the_same_objective_needs_only_the_smaller_floor() -> None:
    arbiter = _arbiter()
    first = _bid("treat", 0.4, (1, 1))
    arbiter.choose([first], now=0)

    # 0.1 better: under the objective floor (0.125) but over the target floor (0.0625).
    choice = arbiter.choose([first, _bid("treat", 0.5, (4, 4))], now=0)

    assert choice is not None
    assert choice.candidate.target == (4, 4)


def test_effort_spent_on_the_target_makes_it_harder_to_abandon() -> None:
    """A rival 0.3 better wins after ten seconds on the target, and loses after five minutes."""
    early = _arbiter()
    late = _arbiter()
    treat = _bid("treat", 0.4)
    explore = _bid("explore", 0.7, (9, 9))
    early.choose([treat], now=0)
    late.choose([treat], now=0)

    after_ten_seconds = early.choose([treat, explore], now=10_000)
    after_five_minutes = late.choose([treat, explore], now=_HORIZON)

    assert after_ten_seconds is not None
    assert after_ten_seconds.candidate == explore
    assert after_five_minutes is not None
    assert after_five_minutes.candidate == treat


def test_effort_grows_with_time_on_the_target_and_is_capped() -> None:
    arbiter = _arbiter()
    arbiter.choose([_bid("treat", 0.4)], now=0)
    commitment = arbiter.commitment

    assert commitment is not None
    assert commitment.effort(0, _HORIZON) == 0.0
    assert commitment.effort(_HORIZON // 2, _HORIZON) == 0.5
    assert commitment.effort(10 * _HORIZON, _HORIZON) == 1.0


def test_a_commitment_without_a_target_builds_no_effort() -> None:
    """Exploring for five minutes does not lock the robot in: a pest still pulls it away (#60)."""
    arbiter = _arbiter()
    explore = _bid("explore", 0.5, None)
    arbiter.choose([explore], now=0)

    # 0.15 better beats the 0.125 objective floor; with effort it would need 0.5 more.
    choice = arbiter.choose([explore, _bid("treat", 0.65, None)], now=_HORIZON)

    assert choice is not None
    assert (choice.candidate.source, choice.switched) == ("treat", True)


def test_a_pest_known_by_its_id_keeps_its_effort_while_its_estimate_moves() -> None:
    """With hop-based navigation the pest's id, not its estimated cell, is the commitment."""
    arbiter = _arbiter()
    trip = ActionCandidate(FollowField("treat"), 0.5, "treat", PestId(finder=3, number=1))
    arbiter.choose([trip], now=0)

    # After five minutes on the trip a rival 0.3 better is held off by the full margin.
    choice = arbiter.choose([trip, _bid("explore", 0.8, (9, 9))], now=_HORIZON)

    assert choice is not None
    assert (choice.candidate.source, choice.switched) == ("treat", False)


def test_a_finished_target_keeps_the_objective_and_restarts_its_effort() -> None:
    """An explorer moves on to the next cell without losing the objective's floor."""
    arbiter = _arbiter()
    arbiter.choose([_bid("explore", 0.45, (1, 1))], now=0)

    next_cell = _bid("explore", 0.45, (2, 1))
    choice = arbiter.choose([next_cell, _bid("treat", 0.5, (9, 9))], now=_HORIZON)

    assert choice is not None
    assert (choice.candidate, choice.switched) == (next_cell, False)
    assert arbiter.commitment is not None
    assert arbiter.commitment.effort(_HORIZON, _HORIZON) == 0.0


def test_dwell_time_blocks_switching_objective() -> None:
    arbiter = _arbiter(min_dwell=30_000)
    treat = _bid("treat", 0.4)
    arbiter.choose([treat], now=0)
    explore = _bid("explore", 0.8, (9, 9))

    during = arbiter.choose([treat, explore], now=10_000)
    after = arbiter.choose([treat, explore], now=40_000)

    assert during is not None
    assert during.candidate == treat
    assert after is not None
    assert after.candidate == explore


def test_an_override_skips_every_comparison() -> None:
    arbiter = _arbiter(min_dwell=30_000)
    treat = _bid("treat", 0.7)
    arbiter.choose([treat], now=0)
    recharge = ActionCandidate(Charge(), 0.95, "recharge", None)

    choice = arbiter.choose([treat, recharge], now=_HORIZON)

    assert choice is not None
    assert (choice.candidate, choice.switched) == (recharge, True)


def test_a_finished_objective_frees_the_robot_immediately() -> None:
    arbiter = _arbiter(min_dwell=30_000)
    arbiter.choose([_bid("treat", 0.7)], now=0)
    explore = _bid("explore", 0.2, (9, 9))

    choice = arbiter.choose([explore], now=1_000)

    assert choice is not None
    assert (choice.candidate, choice.switched) == (explore, False)


def test_no_candidates_releases_the_commitment() -> None:
    arbiter = _arbiter()
    arbiter.choose([_bid("treat", 0.7)], now=0)

    assert arbiter.choose([], now=1_000) is None
    assert arbiter.commitment is None


def test_scores_off_the_scale_are_clamped_and_non_numbers_dropped() -> None:
    choice = _arbiter().choose(
        [_bid("broken", float("nan")), _bid("too-low", -3.0), _bid("too-high", 0.85)], now=0
    )

    assert choice is not None
    assert choice.candidate.source == "too-high"
    clamped = _arbiter().choose([_bid("way-too-high", 7.0)], now=0)
    assert clamped is not None
    assert clamped.candidate.score == 1.0


class _FixedBids:
    """A mock behaviour bidding the same candidates at every decision point."""

    def __init__(self, name: str, candidates: tuple[ActionCandidate, ...]) -> None:
        self.name: str = name
        self._candidates: tuple[ActionCandidate, ...] = candidates

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        return self._candidates


def test_the_brain_acts_on_the_arbiters_choice() -> None:
    field = FieldGeometry(width=5, height=5, cell_size=1.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)
    memory = AgentMemory(1, field)
    fields = FieldProvider(memory, flies=False)
    behaviors: list[Behavior] = [
        _FixedBids("treat", (ActionCandidate(Treat((0, 0)), 0.6, "treat", (0, 0)),)),
        _FixedBids("recharge", (ActionCandidate(Charge(), 0.95, "recharge", None),)),
    ]
    agent = ComposedAgent(
        1, behaviors, memory, fields, Navigator(fields.quadtree), UtilityArbiter()
    )
    robot = RobotSelfState(1, (0, 0), 0.05, BodyType.KILLER, 1.0, 1.0, 1.0)

    decision = agent.decide(Perception(time=0, self_state=robot, sensed=[], inbox=[]))

    assert decision.action == Charge()
    assert agent.inspect().current_goal == "recharge"
