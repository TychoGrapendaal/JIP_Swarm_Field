"""Tests for small behaviour that lives in the contracts."""

from swarm_field.contracts import Direction


def test_direction_offsets() -> None:
    assert Direction.N.offset == (0, 1)
    assert Direction.E.offset == (1, 0)
    assert Direction.SW.offset == (-1, -1)


def test_all_directions_have_distinct_unit_offsets() -> None:
    offsets = {direction.offset for direction in Direction}
    assert len(offsets) == 8
    assert all(dx in (-1, 0, 1) and dy in (-1, 0, 1) for dx, dy in offsets)
