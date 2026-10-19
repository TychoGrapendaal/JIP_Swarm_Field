"""Tests for the pest experiment's scenario description."""

from dataclasses import replace

from swarm_field.experiments import KILLERS_ONLY


def test_lattice_starts_are_distinct_inside_the_field_and_reproducible() -> None:
    starts = KILLERS_ONLY.starts()

    assert len(set(starts)) == KILLERS_ONLY.killers
    for x, y in starts:
        assert 0 <= x < KILLERS_ONLY.size
        assert 0 <= y < KILLERS_ONLY.size
    assert KILLERS_ONLY.starts() == starts
    assert replace(KILLERS_ONLY, seed=1).starts() != starts
