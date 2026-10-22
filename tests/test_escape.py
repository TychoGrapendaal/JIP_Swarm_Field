"""Tests for escaping a wall by trial and error."""

from swarm_field.swarm import EscapeConfig
from swarm_field.swarm.escape import EscapeBeacon, StuckEscape


def _beacons(escape: StuckEscape, cells: list[tuple[int, int]]) -> list[EscapeBeacon]:
    found: list[EscapeBeacon] = []
    for cell in cells:
        beacon = escape.check(cell, (cell[0] + 1, cell[1]))
        if beacon is not None:
            found.append(beacon)
    return found


def test_a_robot_stuck_against_a_wall_escapes_harder_until_it_leaves_the_cell() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=3))

    stuck = _beacons(escape, [(3, 3)] * 9)
    moved = _beacons(escape, [(2, 3)] * 3)

    assert stuck == [
        EscapeBeacon((4, 3), 1.0),
        EscapeBeacon((4, 3), 2.0),
        EscapeBeacon((4, 3), 4.0),
    ]
    assert moved == [EscapeBeacon((3, 3), 1.0)]


def test_a_robot_without_a_wall_ahead_places_nothing() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=3))

    found: list[EscapeBeacon | None] = []
    for _ in range(6):
        found.append(escape.check((3, 3), None))

    assert found == [None] * 6
