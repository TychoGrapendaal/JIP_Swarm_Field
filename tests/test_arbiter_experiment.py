"""Tests for reading which arbiter rule settled a choice, from outside ``UtilityArbiter``."""

from swarm_field.contracts import TimeUnit
from swarm_field.experiments import ArbiterLog, ArbiterRule, ArbiterScenario, LiveRun
from swarm_field.experiments.arbiter import RecordingArbiter
from swarm_field.swarm import (
    ActionCandidate,
    ArbiterConfig,
    FollowField,
    RebalanceConfig,
    UtilityArbiter,
)


def _bid(source: str, score: float, target: tuple[int, int] | None = (1, 1)) -> ActionCandidate:
    return ActionCandidate(FollowField(source), score, source, target)


def test_every_choice_is_read_as_the_rule_that_settled_it() -> None:
    log = ArbiterLog()
    arbiter = RecordingArbiter(UtilityArbiter(ArbiterConfig(min_dwell=10_000)), 1, log)
    explore = _bid("explore", 0.5, None)

    arbiter.choose([explore], now=0)
    arbiter.choose([explore, _bid("dock", 0.8)], now=5_000)
    arbiter.choose([explore, _bid("dock", 0.6)], now=20_000)
    arbiter.choose([explore, _bid("dock", 0.8)], now=20_000)
    arbiter.choose([explore, _bid("treat", 0.95)], now=21_000)
    arbiter.choose([explore], now=22_000)

    assert log.rules == {
        ArbiterRule.FIRST_COMMITMENT: 1,
        ArbiterRule.DWELL_HOLD: 1,
        ArbiterRule.MARGIN_HOLD: 1,
        ArbiterRule.MARGIN_SWITCH: 1,
        ArbiterRule.OVERRIDE: 1,
        ArbiterRule.OBJECTIVE_FINISHED: 1,
    }


def test_a_finished_target_is_read_as_a_retarget() -> None:
    log = ArbiterLog()
    arbiter = RecordingArbiter(UtilityArbiter(), 1, log)
    arbiter.choose([_bid("treat", 0.5, (1, 1))], now=0)

    arbiter.choose([_bid("treat", 0.4, (5, 5))], now=1_000)

    assert (log.retargets, log.last[1].rule) == (1, ArbiterRule.UNCONTESTED)


def test_a_rebalancing_swarm_of_ten_records_every_robot() -> None:
    scenario = ArbiterScenario(
        scouts=6, killers=4, rebalance=RebalanceConfig(), until=2 * TimeUnit.MINUTE
    )
    run = LiveRun.from_config(scenario.into_experiment())
    log = ArbiterLog()
    log.attach(run)

    run.advance_to(scenario.until)

    assert len(log.rules_by_robot) == 10
