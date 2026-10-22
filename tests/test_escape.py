"""Tests for escaping by trial and error when a robot circles in place."""

from swarm_field.contracts import CellId
from swarm_field.swarm import EscapeConfig
from swarm_field.swarm.escape import EscapeBeacon, StuckEscape


def _no_push(cell: CellId) -> float:
    return 0.0


def _beacons(escape: StuckEscape, cells: list[CellId]) -> list[EscapeBeacon]:
    """The escape beacons of a robot whose heading points east at every cell it is in."""
    found: list[EscapeBeacon] = []
    for cell in cells:
        beacon = escape.check(cell, (cell[0] + 1, cell[1]), _no_push)
        if beacon is not None:
            found.append(beacon)
    return found


def test_a_robot_standing_still_escapes_onto_the_cell_ahead_once_per_window() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=3))

    assert _beacons(escape, [(3, 3)] * 9) == [EscapeBeacon((4, 3), 1.0)] * 3


def test_a_robot_shuttling_between_two_cells_escapes_from_where_it_circles() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=5))
    there_and_back: list[CellId] = [(3, 3), (3, 4), (3, 4), (3, 4), (3, 3)]

    assert _beacons(escape, there_and_back) == [EscapeBeacon((3, 4), 1.0)]


def test_an_escape_pushes_harder_than_the_recent_beacon_on_its_cell() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=3))

    beacon: EscapeBeacon | None = None
    for _ in range(3):
        beacon = escape.check((3, 3), (4, 3), lambda cell: 1.5)

    assert beacon == EscapeBeacon((4, 3), 3.0)


def test_a_killer_moving_diagonally_is_not_stuck() -> None:
    escape = StuckEscape(EscapeConfig())
    cells: list[CellId] = []
    for step in range(10):
        cells.extend([(step, step)] * 10)

    assert _beacons(escape, cells) == []


def test_a_decision_without_a_heading_starts_the_window_over() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=3))

    found = _beacons(escape, [(3, 3)] * 2)
    escape.forget()
    found.extend(_beacons(escape, [(3, 3)] * 2))

    assert found == []
