"""Tests for escaping by trial and error when a robot circles in place."""

from swarm_field.contracts import CellId, Time, TimeUnit
from swarm_field.swarm import EscapeConfig
from swarm_field.swarm.hop import EscapeBeacon, StuckEscape

_DECAY = EscapeConfig().escape_decay


def _beacons(escape: StuckEscape, cells: list[CellId], start: Time = 0) -> list[EscapeBeacon]:
    """The escape beacons of a robot heading east from each cell, one decision a second."""
    found: list[EscapeBeacon] = []
    for index, cell in enumerate(cells):
        beacon = escape.check(cell, (cell[0] + 1, cell[1]), start + index * TimeUnit.SECOND)
        if beacon is not None:
            found.append(beacon)
    return found


def test_a_robot_standing_still_escapes_onto_the_cell_ahead() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=3))

    assert _beacons(escape, [(3, 3)] * 3) == [EscapeBeacon((4, 3), 1.0, _DECAY)]


def test_a_robot_shuttling_between_two_cells_escapes_from_where_it_circles() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=5))
    there_and_back: list[CellId] = [(3, 3), (3, 4), (3, 4), (3, 4), (3, 3)]

    assert _beacons(escape, there_and_back) == [EscapeBeacon((3, 4), 1.0, _DECAY)]


def test_an_escape_in_the_same_area_pushes_harder_and_lasts_longer() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=3))

    found = _beacons(escape, [(3, 3)] * 6 + [(3, 4)] * 3)

    assert found == [
        EscapeBeacon((4, 3), 1.0, _DECAY),
        EscapeBeacon((4, 3), 2.0, 2 * _DECAY),
        EscapeBeacon((4, 3), 4.0, 4 * _DECAY),
    ]


def test_an_escape_in_a_new_area_or_after_the_last_one_faded_starts_over() -> None:
    escape = StuckEscape(EscapeConfig(stuck_decisions=3))

    elsewhere = _beacons(escape, [(3, 3)] * 3 + [(9, 9)] * 3)
    later = _beacons(escape, [(9, 9)] * 3, start=2 * _DECAY)

    assert [beacon.strength for beacon in elsewhere + later] == [1.0, 1.0, 1.0]


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
