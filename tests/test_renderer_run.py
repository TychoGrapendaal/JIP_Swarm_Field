"""Tests for rendering a run driven by the runner (C.1 over B.5)."""

from swarm_field.contracts import FailureMode, FailureSpec, ScenarioId, WorldSnapshot
from swarm_field.experiments import build_named_experiment
from swarm_field.runner import (
    ExperimentConfig,
    RobustnessCase,
    build_experiment,
    build_run,
    robustness_case_config,
    robustness_sweep,
    run_experiment,
)
from swarm_field.simulation import MetricsConfig, SimulationLoop

HOUR = 3_600_000


def _s2() -> ExperimentConfig:
    """The S2 preset from seed 101 for one hour, the base of the robustness conditions."""
    return build_named_experiment("S2", 101, HOUR)


BASE_STEP_MS = 500


class RecordingRenderer:
    """A ``Renderer`` that keeps every snapshot it is handed, for assertions."""

    def __init__(self) -> None:
        self.snapshots: list[WorldSnapshot] = []

    def render(self, snapshot: WorldSnapshot) -> None:
        self.snapshots.append(snapshot)


def _config(
    label: str = "test",
    scenario: ScenarioId = ScenarioId.S2,
    seed: int = 101,
    until: int = 2 * HOUR,
    period: int = HOUR,
) -> ExperimentConfig:
    return build_experiment(
        label, scenario, seed, until=until, metrics=MetricsConfig(period=period)
    )


def _all_fail(
    label: str = "all-fail", mode: FailureMode = FailureMode.DEAD_BATTERY
) -> RobustnessCase:
    """A condition in which every robot fails on the first check."""
    return RobustnessCase(
        label=label, failures=FailureSpec(period=1_000, probability=1.0, mode=mode)
    )


def _failed_count(snapshot: WorldSnapshot) -> int:
    return sum(robot.failed for robot in snapshot.robots)


def test_a_renderer_is_handed_one_snapshot_per_metric_tick() -> None:
    renderer = RecordingRenderer()

    run = run_experiment(_config(), renderer)

    assert [snapshot.time for snapshot in renderer.snapshots] == [0, HOUR, 2 * HOUR]
    assert len(renderer.snapshots) == len(run.series)


def test_each_snapshot_carries_the_metrics_sample_for_its_own_time() -> None:
    renderer = RecordingRenderer()

    run = run_experiment(_config(), renderer)

    for snapshot, sample in zip(renderer.snapshots, run.series, strict=True):
        assert snapshot.metrics is not None
        assert snapshot.metrics.time == snapshot.time
        assert snapshot.metrics == sample


def test_a_snapshot_shows_robots_the_run_has_lost() -> None:
    renderer = RecordingRenderer()
    config = _config()
    config.scenario.failures = FailureSpec(
        period=1_000, probability=1.0, mode=FailureMode.DEAD_BATTERY
    )

    run = run_experiment(config, renderer)

    assert len(renderer.snapshots[-1].robots) == 15
    assert _failed_count(renderer.snapshots[-1]) == 15
    assert run.active_robots == 0


def test_a_snapshot_distinguishes_a_fault_condition_from_the_baseline() -> None:
    """The frames of a fault condition must show the swarm shrinking, not only report it."""
    baseline = RecordingRenderer()
    faulty = RecordingRenderer()

    run_experiment(
        robustness_case_config(_s2(), RobustnessCase(label="baseline")),
        baseline,
    )
    run_experiment(
        robustness_case_config(_s2(), _all_fail()),
        faulty,
    )

    # Physical battery depletion can produce natural failures in the baseline. The injected
    # all-fail condition must still fail every robot and remain distinguishable from baseline.
    baseline_failures = _failed_count(baseline.snapshots[-1])
    assert baseline_failures < len(baseline.snapshots[-1].robots)
    assert _failed_count(faulty.snapshots[-1]) == 15
    assert _failed_count(faulty.snapshots[-1]) > baseline_failures


def test_a_run_with_a_renderer_measures_exactly_the_same_as_one_without() -> None:
    """Rendering only reads ground truth, so it cannot move the numbers it is drawn from."""
    plain = run_experiment(_config())
    drawn = run_experiment(_config(), RecordingRenderer())

    assert plain.to_row() == drawn.to_row()


def test_a_run_still_renders_when_it_records_no_metrics() -> None:
    config = _config()
    renderer = RecordingRenderer()
    ctx, _collector = build_run(config, renderer)
    ctx.metrics = None

    SimulationLoop().run(ctx, config.until)

    assert [snapshot.time for snapshot in renderer.snapshots] == [0, HOUR, 2 * HOUR]
    assert all(snapshot.metrics is None for snapshot in renderer.snapshots)


def test_a_robustness_sweep_passes_its_renderer_to_every_case() -> None:
    renderer = RecordingRenderer()
    cases = (RobustnessCase(label="baseline"), _all_fail())

    runs = robustness_sweep(_s2(), cases, renderer=renderer)

    # Two cases, each rendering at time zero and one hour: the renderer is shared, so the frames
    # arrive as one stream rather than being attributable to a condition.
    assert len(renderer.snapshots) == 4
    # The baseline may naturally lose robots to battery depletion; the injected all-fail case
    # must lose the complete swarm.
    assert runs[0].active_robots > 0
    assert runs[1].active_robots == 0


def test_a_case_configured_again_produces_the_same_run() -> None:
    """The basis for rendering one condition after the sweep has tabled it."""
    case = _all_fail()
    swept = robustness_sweep(_s2(), (case,))[0]

    redone = run_experiment(robustness_case_config(_s2(), case))

    assert swept.to_row() == redone.to_row()


def test_driving_a_run_in_slices_measures_exactly_the_same() -> None:
    """Slicing the clock to animate a run cannot change what the run measures.

    An interactive viewer advances the loop one render frame at a time rather than jumping to the
    horizon, so this pins that the two routes through the loop agree. ``World.advance_to`` settles
    each completed move once and clears it, which is what makes stopping on the slice boundary
    harmless.
    """
    config = _config()
    jumped_ctx, jumped = build_run(config)
    SimulationLoop().run(jumped_ctx, config.until)

    sliced_ctx, sliced = build_run(config)
    moment = 0
    while moment < config.until:
        moment += BASE_STEP_MS
        SimulationLoop().run(sliced_ctx, until=min(moment, config.until))

    assert sliced.series() == jumped.series()
