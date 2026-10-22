"""Tests for escaping a wall by trial and error."""

from swarm_field.contracts import CellId
from swarm_field.swarm import EscapeConfig
from swarm_field.swarm.escape import EscapeBeacon, StuckEscape


def _beacons(
    escape: StuckEscape, cells: list[CellId], wall_push: float = 0.0
) -> list[EscapeBeacon]:
    found: list[EscapeBeacon] = []
    for cell in cells:
        beacon = escape.check(cell, (cell[0] + 1, cell[1]), wall_push)
        if beacon is not None:
            found.append(beacon)
    return found


def test_a_robot_standing_against_a_wall_escapes_once_per_window() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=3))

    assert _beacons(escape, [(3, 3)] * 9) == [EscapeBeacon((4, 3), 1.0)] * 3


def test_an_escape_pushes_harder_than_the_push_still_on_its_wall() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=3))

    assert _beacons(escape, [(3, 3)] * 3, wall_push=1.5) == [EscapeBeacon((4, 3), 3.0)]


def test_a_robot_shuttling_between_two_cells_is_stuck() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=4))

    assert len(_beacons(escape, [(3, 3), (3, 4), (3, 3), (3, 4)])) == 1


def test_a_killer_moving_diagonally_is_not_stuck() -> None:
    escape = StuckEscape(EscapeConfig())
    cells: list[CellId] = []
    for step in range(6):
        cells.extend([(step, step)] * 10)

    assert _beacons(escape, cells) == []


def test_a_robot_without_a_wall_ahead_places_nothing() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=3))

    found: list[EscapeBeacon | None] = []
    for _ in range(6):
        found.append(escape.check((3, 3), None, 0.0))

    assert found == [None] * 6
