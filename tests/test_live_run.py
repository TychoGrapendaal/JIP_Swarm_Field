"""Tests for stepping a runner-built run, and recording and replaying it."""

from swarm_field.contracts import InjectInfestation, TimeUnit
from swarm_field.experiments import LiveRun, build_named_experiment

_HORIZON = 10 * TimeUnit.MINUTE


def test_a_recorded_run_replays_to_the_same_world_and_metrics() -> None:
    config = build_named_experiment("pests", 4, _HORIZON)
    live = LiveRun.from_config(config)
    live.advance_to(TimeUnit.MINUTE)
    live.apply_command(InjectInfestation(cell=(20, 20), severity=0.9))
    live.step()
    live.apply_command(InjectInfestation(cell=(5, 25), severity=0.5))
    live.advance_to(_HORIZON)

    replayed = LiveRun.from_config(config, live.command_log())
    replayed.advance_to(_HORIZON)

    assert replayed.context.world.cells == live.context.world.cells
    assert replayed.context.world.positions() == live.context.world.positions()
    assert replayed.metrics.series() == live.metrics.series()
    assert replayed.context.world.cell((20, 20)).infestation > 0.0


def test_the_log_names_the_scenario_and_seed_the_run_came_from() -> None:
    run = LiveRun.from_config(build_named_experiment("explore", 9, TimeUnit.MINUTE))
    run.advance_to(TimeUnit.MINUTE)

    log = run.command_log()

    assert (log.scenario, log.seed, log.until, log.commands) == ("explore", 9, TimeUnit.MINUTE, ())
    assert not run.replaying
    assert LiveRun.from_config(run.config, log).replaying


def test_a_snapshot_carries_the_latest_metrics_sample() -> None:
    run = LiveRun.from_config(build_named_experiment("S1", 1, TimeUnit.MINUTE))

    snapshot = run.snapshot_at(30 * TimeUnit.SECOND)

    assert snapshot.time == 30 * TimeUnit.SECOND
    assert snapshot.metrics == run.metrics.latest()
    assert snapshot.metrics is not None
