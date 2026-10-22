"""The smoke test of the whole pest pipeline: scouts find a pest, killers hear of it and clear it.

Sensing, memory, sharing, the TREAT field, navigation and ``Treat`` all have to work for this to
pass, so a regression in any of them shows up here.
"""

from dataclasses import replace

from swarm_field.contracts import InjectInfestation, TimeUnit
from swarm_field.experiments import LiveRun, PestScenario

_PEST = (2, 1)
"""A cell next to where the scouts start, so it is sensed at once whatever route they explore."""


def test_the_killers_clear_a_pest_the_scouts_find() -> None:
    # Treating takes an hour per unit of severity, so the pest does not grow and starts small.
    scenario = replace(
        PestScenario.smoke(seed=0),
        pests_per_hour=0.0,
        growth_rate=0.0,
        until=15 * TimeUnit.MINUTE,
    )
    run = LiveRun.from_config(scenario.into_experiment("smoke-test"))
    run.apply_command(InjectInfestation(cell=_PEST, severity=0.1))

    run.advance_to(run.horizon)

    (pest,) = run.metrics.pests()
    assert pest.detected_at is not None
    assert pest.known_at is not None
    assert pest.cleared_at is not None
    assert run.context.world.cell(_PEST).infestation == 0.0
