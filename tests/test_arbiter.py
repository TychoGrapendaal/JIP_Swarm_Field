"""Tests for objective arbitration."""

from swarm_field.contracts import BodyType, Move, Perception, RobotSelfState, Treat
from swarm_field.swarm import (
    ArbiterConfig,
    ComposedAgent,
    DecisionContext,
    Objective,
    ObjectiveArbiter,
    ObjectiveProposal,
)

FAR = 40.0
NEAR = 5.0


def _proposal(
    objective: Objective,
    potential: float,
    target: tuple[int, int] | None = (0, 0),
    urgency: float = 0.0,
    distance: float = 0.0,
    is_override: bool = False,
) -> ObjectiveProposal:
    return ObjectiveProposal.create(
        objective=objective,
        action=Move(to=target or (0, 0)),
        potential=potential,
        target=target,
        urgency=urgency,
        distance=distance,
        is_override=is_override,
    )


def _arbiter(min_dwell: int = 0) -> ObjectiveArbiter:
    """5 cells to change objective, 2.5 to change target, rising by 15 as the robot closes."""
    return ObjectiveArbiter(
        ArbiterConfig(
            objective_floor=5.0,
            target_floor=2.5,
            commitment_weight=15.0,
            min_dwell=min_dwell,
        )
    )


def test_score_is_urgency_minus_potential() -> None:
    proposal = _proposal(Objective.TREAT, potential=8.0, urgency=3.0)

    assert proposal.score == -5.0


def test_lowest_potential_wins_when_uncommitted() -> None:
    arbiter = _arbiter()
    treat = _proposal(Objective.TREAT, potential=NEAR, target=(1, 1))
    explore = _proposal(Objective.EXPLORE, potential=FAR, target=(9, 9))

    choice = arbiter.choose([explore, treat], now=0)

    assert choice is not None
    assert choice.proposal.objective is Objective.TREAT
    assert choice.switched


def test_marginal_rival_does_not_displace_the_commitment() -> None:
    arbiter = _arbiter()
    treat = _proposal(Objective.TREAT, potential=50.0, target=(1, 1), distance=40.0)
    arbiter.choose([treat], now=0)

    # 4 better, under the 5-cell floor.
    explore = _proposal(Objective.EXPLORE, potential=46.0, target=(9, 9), distance=30.0)
    choice = arbiter.choose([treat, explore], now=100_000)

    assert choice is not None
    assert choice.proposal.objective is Objective.TREAT
    assert not choice.switched


def test_clearly_better_rival_displaces_the_commitment() -> None:
    arbiter = _arbiter()
    treat = _proposal(Objective.TREAT, potential=50.0, target=(1, 1), distance=40.0)
    arbiter.choose([treat], now=0)

    explore = _proposal(Objective.EXPLORE, potential=30.0, target=(9, 9), distance=30.0)
    choice = arbiter.choose([treat, explore], now=100_000)

    assert choice is not None
    assert choice.proposal.objective is Objective.EXPLORE
    assert choice.switched


def test_target_switch_uses_the_smaller_floor() -> None:
    arbiter = _arbiter()
    first = _proposal(Objective.TREAT, potential=50.0, target=(1, 1), distance=40.0)
    arbiter.choose([first], now=0)

    # 4 better: under the objective floor (5) but over the target floor (2.5).
    second = _proposal(Objective.TREAT, potential=46.0, target=(4, 4), distance=38.0)
    choice = arbiter.choose([first, second], now=100_000)

    assert choice is not None
    assert choice.proposal.target == (4, 4)


def test_a_fresh_commitment_is_cheap_to_abandon() -> None:
    """Nothing is driven yet, so the margin is only the floor and the robot redirects."""
    arbiter = _arbiter()
    treat = _proposal(Objective.TREAT, potential=50.0, target=(1, 1), distance=40.0)
    arbiter.choose([treat], now=0)

    explore = _proposal(Objective.EXPLORE, potential=42.0, target=(9, 9), distance=30.0)
    choice = arbiter.choose([treat, explore], now=100_000)

    assert choice is not None
    assert choice.proposal.objective is Objective.EXPLORE


def test_an_almost_finished_commitment_is_expensive_to_abandon() -> None:
    """The same 8-cell rival loses once most of the journey has been driven."""
    arbiter = _arbiter()
    arbiter.choose(
        [_proposal(Objective.TREAT, potential=50.0, target=(1, 1), distance=40.0)],
        now=0,
    )

    nearly_there = _proposal(Objective.TREAT, potential=50.0, target=(1, 1), distance=2.0)
    explore = _proposal(Objective.EXPLORE, potential=42.0, target=(9, 9), distance=30.0)
    choice = arbiter.choose([nearly_there, explore], now=100_000)

    assert choice is not None
    assert choice.proposal.objective is Objective.TREAT


def test_progress_grows_as_the_robot_closes_on_its_target() -> None:
    arbiter = _arbiter()
    start = _proposal(Objective.TREAT, potential=50.0, target=(1, 1), distance=40.0)
    arbiter.choose([start], now=0)

    assert arbiter.progress(start) == 0.0
    assert arbiter.progress(_proposal(Objective.TREAT, 50.0, (1, 1), distance=10.0)) == 0.75
    assert arbiter.progress(_proposal(Objective.TREAT, 50.0, (1, 1), distance=0.0)) == 1.0


def test_progress_is_zero_when_the_target_moved_further_away() -> None:
    """A re-routed target can be further than at commit; that is not negative progress."""
    arbiter = _arbiter()
    arbiter.choose(
        [_proposal(Objective.TREAT, potential=50.0, target=(1, 1), distance=40.0)],
        now=0,
    )

    detoured = _proposal(Objective.TREAT, potential=50.0, target=(1, 1), distance=55.0)

    assert arbiter.progress(detoured) == 0.0


def test_dwell_time_blocks_switching() -> None:
    arbiter = _arbiter(min_dwell=30_000)
    treat = _proposal(Objective.TREAT, potential=50.0, target=(1, 1), distance=40.0)
    arbiter.choose([treat], now=0)

    explore = _proposal(Objective.EXPLORE, potential=1.0, target=(9, 9), distance=30.0)
    during = arbiter.choose([treat, explore], now=10_000)
    after = arbiter.choose([treat, explore], now=40_000)

    assert during is not None
    assert during.proposal.objective is Objective.TREAT
    assert after is not None
    assert after.proposal.objective is Objective.EXPLORE


def test_override_skips_every_comparison() -> None:
    arbiter = _arbiter(min_dwell=30_000)
    treat = _proposal(Objective.TREAT, potential=1.0, target=(1, 1))
    arbiter.choose([treat], now=0)

    recharge = _proposal(Objective.RECHARGE, potential=FAR, target=(0, 0), is_override=True)
    choice = arbiter.choose([treat, recharge], now=1_000)

    assert choice is not None
    assert choice.proposal.objective is Objective.RECHARGE


def test_vanished_commitment_frees_the_robot_immediately() -> None:
    arbiter = _arbiter(min_dwell=30_000)
    treat = _proposal(Objective.TREAT, potential=1.0, target=(1, 1))
    arbiter.choose([treat], now=0)

    explore = _proposal(Objective.EXPLORE, potential=FAR, target=(9, 9))
    choice = arbiter.choose([explore], now=1_000)

    assert choice is not None
    assert choice.proposal.objective is Objective.EXPLORE


def test_no_proposals_releases_the_commitment() -> None:
    arbiter = _arbiter()
    arbiter.choose([_proposal(Objective.TREAT, potential=1.0)], now=0)

    assert arbiter.choose([], now=1_000) is None
    assert arbiter.committed is None


class _ScriptedBehavior:
    """Emits a different set of proposals on each successive decision point."""

    def __init__(self, rounds: tuple[tuple[ObjectiveProposal, ...], ...]) -> None:
        self.name: str = "scripted"
        self._rounds: tuple[tuple[ObjectiveProposal, ...], ...] = rounds
        self._index: int = 0

    def propose(self, ctx: DecisionContext) -> tuple[ObjectiveProposal, ...]:
        round_index = min(self._index, len(self._rounds) - 1)
        self._index += 1
        return self._rounds[round_index]


def _perception(time: int) -> Perception:
    return Perception(
        time=time,
        self_state=RobotSelfState(
            robot_id=1,
            position=(0, 0),
            battery=1.0,
            body=BodyType.KILLER,
            speed_mps=1.0,
            consumable=1.0,
            health=1.0,
        ),
        sensed=[],
        inbox=[],
    )


def test_brain_holds_its_objective_through_a_marginally_better_offer() -> None:
    treat = ObjectiveProposal.create(
        objective=Objective.TREAT,
        action=Treat(cell=(1, 1)),
        potential=50.0,
        target=(1, 1),
        distance=40.0,
    )
    explore = _proposal(Objective.EXPLORE, potential=46.0, target=(9, 9), distance=30.0)
    agent = ComposedAgent(
        robot_id=1,
        behaviors=[_ScriptedBehavior(((treat,), (treat, explore)))],
        arbiter=_arbiter(),
    )

    first = agent.decide(_perception(0))
    second = agent.decide(_perception(100_000))

    assert first.action == Treat(cell=(1, 1))
    assert second.action == Treat(cell=(1, 1))
    assert agent.inspect().current_goal == "treat"


def test_brain_without_an_arbiter_still_takes_the_top_score() -> None:
    treat = _proposal(Objective.TREAT, potential=50.0, target=(1, 1))
    explore = _proposal(Objective.EXPLORE, potential=42.0, target=(9, 9))
    agent = ComposedAgent(robot_id=1, behaviors=[_ScriptedBehavior(((treat, explore),))])

    decision = agent.decide(_perception(0))

    assert decision.action == Move(to=(9, 9))