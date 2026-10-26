"""Scenario tests: three robots exploring a small field with the EXPLORE seeder.

Each run is a ``swarm_field.experiments`` exploration run by ``swarm_field.runner``, measured
from ground truth. ``studies/explore_variants.py`` compares many more variants on larger fields.
"""

from dataclasses import replace
from functools import cache

from studies.baselines import RANDOM_WAYPOINT_BRAIN
from swarm_field.contracts import MetricsSample, TimeUnit
from swarm_field.experiments import EXPLORE, EXPLORER, ExplorerPreset, MapConfig
from swarm_field.runner import run_experiment
from swarm_field.swarm import ExploreConfig

_FIELD = replace(
    EXPLORE.without_robots()
    .with_map(MapConfig(size=24, obstacles=()))
    .with_metrics(replace(EXPLORE.metrics, measure_from=45 * TimeUnit.MINUTE)),
    until=90 * TimeUnit.MINUTE,
)
"""Idleness is measured over the second half of the run, after the first exploration."""
_SPREAD = ExploreConfig()
_PLAIN = ExploreConfig(spread_peers=False)


@cache
def _final(explore: ExploreConfig | None) -> MetricsSample:
    """Run the scenario and return its last metrics; ``None`` runs the random-waypoint baseline."""
    if explore is None:
        scenario = _FIELD.add_robots(EXPLORER, 3)
        experiment = replace(scenario.into_experiment(), brain=RANDOM_WAYPOINT_BRAIN)
    else:
        experiment = _FIELD.add_robots(ExplorerPreset(explore=explore), 3).into_experiment()
    return run_experiment(experiment)[-1]


def test_explorers_cover_more_of_the_field_and_keep_it_fresher_than_random_waypoints() -> None:
    # A scout senses only the cell it is in, so three scouts do not cover the whole field in the
    # run; following the EXPLORE field they cover clearly more of it than random waypoints.
    explore = _final(_SPREAD)
    baseline = _final(None)

    assert explore.coverage > baseline.coverage
    assert explore.mean_idleness_ms < baseline.mean_idleness_ms


def test_the_peer_discount_cuts_idleness() -> None:
    # Redundancy is left out: idleness is what the discount is meant to improve.
    plain = _final(_PLAIN)
    spread = _final(_SPREAD)

    assert spread.mean_idleness_ms < plain.mean_idleness_ms
