"""Scenario tests: three robots exploring a small field with the EXPLORE seeder.

Each run is a ``swarm_field.experiments`` exploration on the real simulation loop, measured from
ground truth. ``tests/benchmark_explore.py`` compares many more variants on larger fields.
"""

from functools import cache

from swarm_field.contracts import RobotId, TimeUnit
from swarm_field.experiments import ExploreScenario, RandomWaypointSeeder, build_explore_run
from swarm_field.simulation import CoverageSummary
from swarm_field.swarm import ExploreConfig, ExploreSeeder, FieldSeeder

_SCENARIO = ExploreScenario(width=24, height=24, robots=3, measure_from=45 * TimeUnit.MINUTE)
"""Idleness is measured once the field has been covered, which takes about 40 minutes."""
_END = 90 * TimeUnit.MINUTE
_SPREAD = ExploreConfig()
_PLAIN = ExploreConfig(spread_peers=False)


@cache
def _summary(config: ExploreConfig | None) -> CoverageSummary:
    """Run the scenario; ``None`` runs the random-waypoint baseline."""

    def seeder_for(robot: RobotId) -> FieldSeeder:
        if config is None:
            return RandomWaypointSeeder(_SCENARIO.field_geometry(), (), seed=robot)
        return ExploreSeeder(config)

    run = build_explore_run(_SCENARIO, seeder_for)
    run.advance_to(_END)
    return run.summary()


def test_explorers_cover_the_field_and_keep_it_fresher_than_random_waypoints() -> None:
    explore = _summary(_SPREAD)
    baseline = _summary(None)

    assert explore.time_to_full is not None
    assert explore.mean_idleness < baseline.mean_idleness


def test_the_peer_discount_cuts_redundant_visits_and_idleness() -> None:
    plain = _summary(_PLAIN)
    spread = _summary(_SPREAD)

    assert spread.redundancy < plain.redundancy
    assert spread.mean_idleness < plain.mean_idleness
