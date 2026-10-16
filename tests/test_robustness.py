"""Tests for failure injection and position error (B.5 robustness)."""

from random import Random

import pytest

from swarm_field.contracts import (
    AgentInspection,
    BodyType,
    CellId,
    Decision,
    FailureMode,
    FailureSpec,
    FieldGeometry,
    Idle,
    Move,
    Perception,
    PositionErrorSpec,
    RobotId,
    ScenarioId,
)
from swarm_field.runner import (
    ExperimentConfig,
    RobustnessCase,
    build_run,
    default_robustness_cases,
    robustness_sweep,
)
from swarm_field.simulation import (
    FailureCheck,
    FailureConfig,
    RobotDecision,
    RobotState,
    SimulationContext,
    SimulationLoop,
    WorldState,
)
from swarm_field.simulation.scenario import build_scenario


class RecordingAgent:
    """A brain that stands still and remembers the cell it was told it occupies."""

    def __init__(self) -> None:
        self.believed: list[CellId] = []

    def decide(self, perception: Perception) -> Decision:
        self.believed.append(perception.self_state.position)
        return Decision(action=Idle())

    def inspect(self) -> AgentInspection:
        return AgentInspection(robot_id=0, current_goal=None, belief=None)


def _field(width: int = 5, height: int = 5) -> FieldGeometry:
    return FieldGeometry(
        width=width,
        height=height,
        cell_size=10.0,
        row_sp=0.0,
        in_row_sp=0.0,
        headland=0.0,
        connectivity=4,
    )


def _world(*robots: RobotState) -> WorldState:
    return WorldState(_field(), robots=robots)


def _robot(robot_id: RobotId = 1, at: CellId = (2, 2)) -> RobotState:
    return RobotState(robot_id=robot_id, position=at, body=BodyType.SCOUT, speed_mps=1.0)


def _ctx(world: WorldState, seed: int = 1) -> SimulationContext:
    return SimulationContext(now=0, rng=Random(seed), agents={}, world=world, noise=Random(seed))


def _one_decision(world: WorldState, agent: RecordingAgent, ctx: SimulationContext) -> None:
    """Let a single robot decide once, and record where it thought it was."""
    ctx.agents = {1: agent}
    ctx.schedule(RobotDecision(time=0, robot=1, decision_interval=1_000))
    SimulationLoop().run(ctx, until=1)


def test_a_dead_battery_failure_empties_the_battery_and_stops_the_robot() -> None:
    world = _world(_robot())

    world.fail_robot(1, FailureMode.DEAD_BATTERY)

    assert world.robots[1].failed is True
    assert world.robots[1].battery == 0.0


def test_a_disabled_failure_keeps_the_robot_observable_but_stops_it() -> None:
    world = _world(_robot())

    world.fail_robot(1, FailureMode.DISABLED)

    assert world.robots[1].health == 0.0
    assert world.robots[1].battery == pytest.approx(1.0)


def test_an_offline_failure_silences_the_radio_only() -> None:
    world = _world(_robot())
    world.robots[1].radio_range_m = 20.0

    world.fail_robot(1, FailureMode.OFFLINE)

    assert world.robots[1].radio_range_m == 0.0
    assert world.robots[1].health == pytest.approx(1.0)


def test_a_failed_robot_stays_failed() -> None:
    world = _world(_robot())

    world.fail_robot(1, FailureMode.DEAD_BATTERY)
    world.robots[1].battery = 1.0
    world.fail_robot(1, FailureMode.DEAD_BATTERY)

    assert world.robots[1].battery == 1.0


def test_live_robots_excludes_the_failed_in_id_order() -> None:
    world = _world(_robot(1), _robot(2), _robot(3))
    world.fail_robot(2, FailureMode.DISABLED)

    assert world.live_robots() == (1, 3)


def test_a_failed_robot_stops_deciding_and_stops_rescheduling() -> None:
    world = _world(_robot())
    ctx = _ctx(world)
    world.fail_robot(1, FailureMode.DISABLED)
    ctx.schedule(RobotDecision(time=0, robot=1))

    SimulationLoop().run(ctx, until=10_000)

    assert len(ctx.queue) == 0


def test_a_move_under_way_still_lands_after_a_failure() -> None:
    """A failed robot must not be stranded mid-cell; its arrival is already committed."""
    world = _world(_robot())
    world.apply_action(1, Move(to=(3, 2)), time=0)
    world.fail_robot(1, FailureMode.DISABLED)

    world.advance_to(30_000)

    assert world.robots[1].position == (3, 2)


def test_a_zero_probability_never_fails_a_robot() -> None:
    world = _world(_robot(1), _robot(2), _robot(3))
    ctx = _ctx(world, seed=11)
    ctx.schedule(FailureCheck(time=0, config=FailureConfig(period=1_000, probability=0.0)))

    SimulationLoop().run(ctx, until=10_000)

    assert world.live_robots() == (1, 2, 3)


def test_certain_failures_take_every_live_robot() -> None:
    world = _world(_robot(1), _robot(2), _robot(3))
    ctx = _ctx(world, seed=5)
    ctx.schedule(FailureCheck(time=0, config=FailureConfig(period=100, probability=1.0)))

    SimulationLoop().run(ctx, until=1_000)

    assert world.live_robots() == ()


def test_the_failure_check_reschedules_itself() -> None:
    world = _world(_robot())
    ctx = _ctx(world)
    ctx.schedule(FailureCheck(time=0, config=FailureConfig(period=1_000, probability=0.0)))

    SimulationLoop().run(ctx, until=3_500)

    assert len(ctx.queue) == 1


def test_an_intermediate_probability_fails_some_but_not_all() -> None:
    world = _world(_robot(1), _robot(2), _robot(3), _robot(4))
    ctx = _ctx(world, seed=3)
    ctx.noise = Random(0)
    ctx.schedule(FailureCheck(time=0, config=FailureConfig(period=1_000, probability=0.5)))

    SimulationLoop().run(ctx, until=1)

    assert 0 < len(world.live_robots()) < 4


def test_faults_are_drawn_from_the_noise_source_not_the_world() -> None:
    """The pest dynamics must see the same draws with faults on, or a sweep compares nothing."""
    scenario = build_scenario(ScenarioId.S2, seed=101)

    def world_stream(with_failures: bool) -> list[float]:
        config = ExperimentConfig(label="x", scenario=scenario, until=180_000)
        config.scenario.failures = (
            FailureSpec(period=1_000, probability=0.5) if with_failures else FailureSpec()
        )
        ctx, _ = build_run(config)
        return [ctx.rng.random() for _ in range(5)]

    assert world_stream(with_failures=True) == world_stream(with_failures=False)


def test_the_same_holds_for_position_error() -> None:
    scenario = build_scenario(ScenarioId.S2, seed=101)

    def world_stream(error: float) -> list[float]:
        config = ExperimentConfig(label="x", scenario=scenario, until=180_000)
        config.scenario.position_error = PositionErrorSpec(probability=error)
        ctx, _ = build_run(config)
        return [ctx.rng.random() for _ in range(5)]

    assert world_stream(0.5) == world_stream(0.0)


def test_a_robot_believes_its_true_cell_when_position_error_is_off() -> None:
    world = _world(_robot(1, at=(1, 3)))
    agent = RecordingAgent()

    _one_decision(world, agent, _ctx(world))

    assert agent.believed == [(1, 3)]


def test_position_error_displaces_the_position_the_robot_believes() -> None:
    world = _world(_robot(1, at=(2, 2)))
    ctx = _ctx(world, seed=7)
    ctx.position_error = PositionErrorSpec(probability=1.0, max_offset=1)
    agent = RecordingAgent()

    _one_decision(world, agent, ctx)

    assert agent.believed[0] != (2, 2)
    for reported in agent.believed:
        assert abs(reported[0] - 2) <= 1
        assert abs(reported[1] - 2) <= 1


def test_position_error_is_clamped_inside_the_field() -> None:
    world = _world(_robot(1, at=(0, 0)))
    ctx = _ctx(world, seed=2)
    ctx.position_error = PositionErrorSpec(probability=1.0, max_offset=3)
    agent = RecordingAgent()

    _one_decision(world, agent, ctx)

    for reported in agent.believed:
        assert world.is_inside(reported)


def test_position_error_only_perturbs_the_belief_not_the_truth() -> None:
    world = _world(_robot(1, at=(2, 2)))
    ctx = _ctx(world, seed=7)
    ctx.position_error = PositionErrorSpec(probability=1.0, max_offset=1)
    agent = RecordingAgent()

    _one_decision(world, agent, ctx)

    assert world.robots[1].position == (2, 2)


def test_failure_spec_rejects_an_impossible_probability() -> None:
    with pytest.raises(ValueError):
        FailureSpec(probability=1.5)


def test_failure_spec_rejects_a_non_positive_period() -> None:
    with pytest.raises(ValueError):
        FailureSpec(period=0)


def test_position_error_spec_rejects_a_negative_offset() -> None:
    with pytest.raises(ValueError):
        PositionErrorSpec(max_offset=-1)


def test_a_robustness_sweep_runs_every_case_from_the_same_seed() -> None:
    cases = (
        RobustnessCase(label="baseline"),
        RobustnessCase(label="half-dead", failures=FailureSpec(period=1_000, probability=0.5)),
    )

    runs = robustness_sweep(ScenarioId.S2, cases, seed=101, until=120_000)

    assert [run.label for run in runs] == ["S2/baseline", "S2/half-dead"]
    assert {run.seed for run in runs} == {101}


def test_failing_robots_show_up_as_fewer_active_robots() -> None:
    baseline, failing = robustness_sweep(
        ScenarioId.S2,
        (
            RobustnessCase(label="baseline"),
            RobustnessCase(
                label="always-dead", failures=FailureSpec(period=1_000, probability=1.0)
            ),
        ),
        seed=101,
        until=120_000,
    )

    assert baseline.active_robots == 15
    assert failing.active_robots == 0


def test_a_baseline_sweep_keeps_every_robot() -> None:
    (only,) = robustness_sweep(
        ScenarioId.S2, (RobustnessCase(label="baseline"),), seed=101, until=120_000
    )

    assert only.active_robots == 15


def test_the_default_cases_include_a_baseline_and_perturb_one_axis_at_a_time() -> None:
    cases = default_robustness_cases()

    assert cases[0] == RobustnessCase(label="baseline")
    for case in cases[1:]:
        faults = case.failures.probability > 0.0
        errors = case.position_error.probability > 0.0
        assert faults != errors


def test_a_robustness_sweep_is_reproducible() -> None:
    cases = default_robustness_cases(
        failure_probabilities=(0.01,), position_error_probabilities=(0.2,)
    )

    first = robustness_sweep(ScenarioId.S2, cases, seed=101, until=60_000)
    second = robustness_sweep(ScenarioId.S2, cases, seed=101, until=60_000)

    assert [run.to_row() for run in first] == [run.to_row() for run in second]


def test_sweeping_every_failure_mode_keeps_the_baseline_intact() -> None:
    failing = tuple(
        RobustnessCase(
            label=mode.name.lower(),
            failures=FailureSpec(period=1_000, probability=1.0, mode=mode),
        )
        for mode in FailureMode
    )
    cases = (RobustnessCase(label="baseline"), *failing)

    runs = robustness_sweep(ScenarioId.S2, cases, seed=101, until=60_000)

    assert [run.active_robots for run in runs] == [15, 0, 0, 0]
