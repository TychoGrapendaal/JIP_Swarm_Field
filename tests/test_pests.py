"""Tests for the pest experiment's scenario description."""

from dataclasses import replace

from swarm_field.experiments import RELAYED_KILLERS


def test_lattice_starts_are_inside_the_field_and_reproducible() -> None:
    starts = RELAYED_KILLERS.starts()

    assert len(starts) == RELAYED_KILLERS.scouts + RELAYED_KILLERS.killers
    killers = starts[RELAYED_KILLERS.scouts :]
    assert len(set(killers)) == RELAYED_KILLERS.killers
    for x, y in starts:
        assert 0 <= x < RELAYED_KILLERS.size
        assert 0 <= y < RELAYED_KILLERS.size
    assert RELAYED_KILLERS.starts() == starts
    assert replace(RELAYED_KILLERS, seed=1).starts() != starts


def test_lattice_starts_avoid_obstacles() -> None:
    scenario = replace(RELAYED_KILLERS, obstacles=None, scouts=64, killers=1)
    blocked = set(scenario.blocked_cells())

    for cell in scenario.starts():
        assert cell not in blocked
