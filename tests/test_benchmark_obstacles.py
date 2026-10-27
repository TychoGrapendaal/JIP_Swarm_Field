"""Tests for the obstacle benchmark's pest responses."""

from studies.benchmark_obstacles import pest_response_configurations
from swarm_field.experiments import HOP_KILLER, RELAY, RELAYED_KILLERS


def test_the_pest_responses_keep_the_number_of_killers() -> None:
    configurations = pest_response_configurations(RELAYED_KILLERS, seeded_no_yield=True)

    expected = [
        "seeded",
        "seeded-best",
        "seeded-no-yield",
        "hop",
        "hop-repel",
        "hop-tuned",
        "hop-best",
    ]
    assert list(configurations) == expected
    for configured in configurations.values():
        assert configured.killers == RELAYED_KILLERS.killers
        assert configured.count(RELAY.name) == RELAYED_KILLERS.count(RELAY.name)
    assert configurations["hop-tuned"].count(HOP_KILLER.name) == RELAYED_KILLERS.killers
