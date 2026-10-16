"""Scenario tests: three robots exploring a small field with the EXPLORE seeder and field-aware
exploration behavior. Runs are measured from ground truth; ``tests/benchmark_explore.py`` compares
more variants on larger fields.
"""

from dataclasses import replace
from functools import cache

from swarm_field.contracts import MetricsSample, TimeUnit
from swarm_field.experiments import ExploreScenario
from swarm_field.runner import run_experiment
from swarm_field.swarm import ExploreConfig
from tests.baselines import RANDOM_WAYPOINT_BRAIN

_SCENARIO = ExploreScenario(
    width=24,
    height=24,
    robots=3,
    measure_from=45 * TimeUnit.MINUTE,
    until=90 * TimeUnit.MINUTE,
)
"""Idleness is measured once the field has been covered, which takes about 40 minutes."""
_SPREAD = ExploreConfig()
_PLAIN = ExploreConfig(spread_peers=False)


@cache
def _final(explore: ExploreConfig | None) -> MetricsSample:
    """Run the scenario and return its last metrics; ``None`` runs the random-waypoint baseline."""
    if explore is None:
        experiment = replace(_SCENARIO.into_experiment(), brain=RANDOM_WAYPOINT_BRAIN)
    else:
        experiment = replace(_SCENARIO, explore=explore).into_experiment()
    return run_experiment(experiment).series[-1]


def test_explorers_cover_the_field_and_keep_it_fresher_than_random_waypoints() -> None:
    explore = _final(_SPREAD)
    baseline = _final(None)

    assert explore.coverage == 1.0
    assert explore.mean_idleness_ms < baseline.mean_idleness_ms


def test_the_peer_discount_cuts_redundant_visits_and_idleness() -> None:
    plain = _final(_PLAIN)
    spread = _final(_SPREAD)

    assert spread.redundancy < plain.redundancy
    assert spread.mean_idleness_ms < plain.mean_idleness_ms
