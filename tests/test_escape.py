"""Tests for escaping by trial and error when a robot circles in place."""

from swarm_field.contracts import CellId, Time, TimeUnit
from swarm_field.swarm import EscapeConfig
from swarm_field.swarm.hop import EscapeBeacon, StuckEscape

_DECAY = EscapeConfig().escape_decay
_DECISION = 15 * TimeUnit.SECOND
_KILLER_SPEED = 0.01
"""A ground killer's top speed in cells per second: 0.1 m/s over 10 m cells."""
_SHORT = EscapeConfig(window=2 * _DECISION)
"""An escape that judges the robot over three decisions."""


def _beacons(
    escape: StuckEscape,
    cells: list[CellId],
    start: Time = 0,
    on_heading: list[bool] | None = None,
) -> list[EscapeBeacon]:
    """The escape beacons of a robot heading east from each cell, one decision every 15 s.

    ``on_heading`` says which decisions followed a heading; by default all of them.
    """
    found: list[EscapeBeacon] = []
    for index, cell in enumerate(cells):
        ahead: CellId | None = (cell[0] + 1, cell[1])
        if on_heading is not None and not on_heading[index]:
            ahead = None
        beacon = escape.check(cell, ahead, start + index * _DECISION, _KILLER_SPEED)
        if beacon is not None:
            found.append(beacon)
    return found


def test_a_robot_standing_still_escapes_onto_the_cell_ahead() -> None:
    escape = StuckEscape(_SHORT)

    assert _beacons(escape, [(3, 3)] * 3) == [EscapeBeacon((4, 3), 1.0, _DECAY)]


def test_a_robot_shuttling_between_two_cells_escapes_from_where_it_circles() -> None:
    escape = StuckEscape(EscapeConfig(window=4 * _DECISION))
    there_and_back: list[CellId] = [(3, 3), (3, 4), (3, 4), (3, 4), (3, 3)]

    assert _beacons(escape, there_and_back) == [EscapeBeacon((3, 4), 1.0, _DECAY)]


def test_an_escape_in_the_same_area_pushes_harder_and_lasts_longer() -> None:
    escape = StuckEscape(_SHORT)

    found = _beacons(escape, [(3, 3)] * 6 + [(3, 4)] * 3)

    assert found == [
        EscapeBeacon((4, 3), 1.0, _DECAY),
        EscapeBeacon((4, 3), 2.0, 2 * _DECAY),
        EscapeBeacon((4, 3), 4.0, 4 * _DECAY),
    ]


def test_an_escape_in_a_new_area_or_after_the_last_one_faded_starts_over() -> None:
    escape = StuckEscape(_SHORT)

    elsewhere = _beacons(escape, [(3, 3)] * 3 + [(9, 9)] * 3)
    later = _beacons(escape, [(9, 9)] * 3, start=2 * _DECAY)

    assert [beacon.strength for beacon in elsewhere + later] == [1.0, 1.0, 1.0]


def test_a_killer_moving_diagonally_is_not_stuck() -> None:
    escape = StuckEscape(EscapeConfig())
    cells: list[CellId] = []
    for step in range(10):
        cells.extend([(step, step)] * 10)

    assert _beacons(escape, cells) == []


def test_a_killer_that_steps_once_and_then_stands_is_stuck() -> None:
    escape = StuckEscape(EscapeConfig())

    found = _beacons(escape, [(3, 3)] + [(4, 3)] * 40)

    assert found == [EscapeBeacon((5, 3), 1.0, _DECAY)]


def test_a_shuttle_that_loses_its_heading_on_the_way_is_still_stuck() -> None:
    escape = StuckEscape(EscapeConfig())
    cells: list[CellId] = []
    on_heading: list[bool] = []
    for _ in range(4):
        cells.extend([(3, 3)] * 7 + [(3, 4)] * 8)
        on_heading.extend([True] * 6 + [False] + [True] * 5 + [False] * 3)

    found = _beacons(escape, cells, on_heading=on_heading)

    assert len(found) >= 1


def test_a_robot_that_waited_for_work_is_not_stuck_on_its_first_heading() -> None:
    escape = StuckEscape(EscapeConfig())

    waited = _beacons(escape, [(3, 3)] * 50, on_heading=[False] * 50)
    first = _beacons(escape, [(3, 3)], start=50 * _DECISION)

    assert waited + first == []


def test_a_treatment_starts_the_record_over() -> None:
    escape = StuckEscape(_SHORT)

    found = _beacons(escape, [(3, 3)] * 2)
    escape.forget()
    found.extend(_beacons(escape, [(3, 3)] * 2, start=2 * _DECISION))

    assert found == []
