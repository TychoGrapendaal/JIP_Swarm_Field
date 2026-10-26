"""Smoke tests of the pest pipeline: the swarm, and the grid-search baseline it is compared with.

Sensing, memory, sharing, the TREAT field, navigation and ``Treat`` all have to work for this to
pass, so a regression in any of them shows up here.
"""

from dataclasses import replace

from studies.baselines import grid_sweep_brain
from swarm_field.contracts import InjectInfestation, TimeUnit
from swarm_field.experiments import SEEDED_KILLER, SMOKE, LiveRun

_PEST = (2, 1)
"""A cell next to where the explorers start, so it is sensed at once whatever route they take."""


def test_the_killers_clear_a_pest_the_explorers_find() -> None:
    # One injected pest and no others: no new pests appear, and the pest does not grow.
    pests = replace(SMOKE.pests, pests_per_hour=0.0, growth_rate=0.0)
    scenario = replace(SMOKE.with_pests(pests), seed=0, until=15 * TimeUnit.MINUTE)
    run = LiveRun.from_config(scenario.into_experiment("smoke-test"))
    run.apply_command(InjectInfestation(cell=_PEST, severity=0.1))

    run.advance_to(run.horizon)

    (pest,) = run.metrics.pests()
    assert pest.detected_at is not None
    assert pest.known_at is not None
    assert pest.cleared_at is not None
    assert run.context.world.cell(_PEST).infestation == 0.0


def test_a_grid_sweep_robot_clears_a_pest_on_its_strip() -> None:
    # Ten utility robots that share nothing, one injected pest, no new pests and no growth.
    pests = replace(SMOKE.pests, pests_per_hour=0.0, growth_rate=0.0)
    scenario = SMOKE.with_pests(pests).without_robots().add_robots(SEEDED_KILLER, 10)
    scenario = scenario.with_robot_config(replace(scenario.robots, radio_range_m=0.0))
    scenario = replace(scenario, seed=0, until=15 * TimeUnit.MINUTE)
    config = replace(scenario.into_experiment("grid-sweep-test"), brain=grid_sweep_brain(10))
    run = LiveRun.from_config(config)
    # The first robot sweeps the leftmost strip, up column 0 first.
    run.apply_command(InjectInfestation(cell=(0, 5), severity=0.1))

    run.advance_to(run.horizon)

    (pest,) = run.metrics.pests()
    assert pest.cleared_at is not None
