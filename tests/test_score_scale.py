"""Tests for the shared score scale: a field's potential as a score in [0, 1]."""

from math import inf

from swarm_field.swarm import score_from_potential


def test_the_score_keeps_the_order_of_potentials_within_zero_and_one() -> None:
    potentials = [-30.0, -10.0, 0.0, 5.0, 40.0, 2_000.0, inf]

    scores = [score_from_potential(potential) for potential in potentials]

    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert score_from_potential(0.0) == 0.5
    assert score_from_potential(inf) == 0.0
