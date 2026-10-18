"""Tests for the battery behaviour, and a short run of the arbiter scenario."""

from swarm_field.contracts import (
    AgentObjective,
    BodyType,
    Charge,
    FieldGeometry,
    PeerObservation,
    RobotSelfState,
    TimeUnit,
)
from swarm_field.experiments import ArbiterLog, ArbiterScenario, LiveRun
from swarm_field.swarm import (
    AgentMemory,
    DecisionContext,
    FollowField,
    RechargeConfig,
    ReturnToDock,
)

_DOCKS = ((1, 1), (8, 8))


def _ctx(
    battery: float, position: tuple[int, int] = (2, 2), peers_at_far_dock: int = 0
) -> DecisionContext:
    memory = AgentMemory(1, FieldGeometry(10, 10, 10.0, 0.0, 0.0, 0.0))
    memory.merge(
        PeerObservation(
            cell=(5, 5),
            confidence=1.0,
            observed_at=0,
            source=peer,
            peer=peer,
            body=BodyType.SCOUT,
            target=_DOCKS[1],
            objective=AgentObjective.RECHARGE,
        )
        for peer in range(2, 2 + peers_at_far_dock)
    )
    self_state = RobotSelfState(1, position, battery, BodyType.SCOUT, 1.0, 1.0, 1.0)
    return DecisionContext(time=0, self_state=self_state, sensed=[], memory=memory)


def _behaviour() -> ReturnToDock:
    return ReturnToDock(_DOCKS, ("dock-0", "dock-1"), RechargeConfig(low=0.5, critical=0.2))


def test_a_charged_robot_does_not_bid() -> None:
    assert _behaviour().propose(_ctx(battery=0.8)) == ()


def test_a_low_battery_bids_for_every_dock_less_for_a_crowded_one() -> None:
    bids = _behaviour().propose(_ctx(battery=0.35, peers_at_far_dock=2))

    assert [bid.target for bid in bids] == list(_DOCKS)
    assert 0.0 < bids[1].score < bids[0].score < 0.9


def test_a_critical_battery_overrides_towards_the_nearest_dock() -> None:
    (bid,) = _behaviour().propose(_ctx(battery=0.1, position=(7, 6)))

    assert bid.action == FollowField("dock-1")
    assert bid.score >= 0.9


def test_a_robot_on_a_dock_charges() -> None:
    (bid,) = _behaviour().propose(_ctx(battery=0.3, position=(1, 1)))

    assert bid.action == Charge()


def test_a_ten_robot_swarm_records_a_rule_for_every_choice() -> None:
    scenario = ArbiterScenario(scouts=6, killers=4, until=2 * TimeUnit.MINUTE)
    run = LiveRun.from_config(scenario.into_experiment())
    log = ArbiterLog()
    log.attach(run)

    run.advance_to(scenario.until)

    assert len(log.rules_by_robot) == 10
    assert log.decisions() > 0
