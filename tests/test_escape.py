"""Tests for escaping by trial and error."""

from swarm_field.contracts import TimeUnit
from swarm_field.swarm import EscapeConfig
from swarm_field.swarm.escape import EscapeBeacon, StuckEscape

_MINUTE = TimeUnit.MINUTE


def test_a_robot_whose_hop_number_stays_put_escapes_until_it_comes_down() -> None:
    escape = StuckEscape(EscapeConfig(stuck_window=5 * _MINUTE))

    found: list[EscapeBeacon | None] = []
    for minute, cell in ((0, (3, 3)), (5, (6, 3)), (10, (3, 3)), (11, (4, 3)), (16, (5, 3))):
        hop = 2.0
        if minute == 16:
            hop = 1.5
        found.append(escape.check(cell, hop, minute * _MINUTE, None))

    assert found == [None, EscapeBeacon((6, 3), 1.0), None, None, None]


def test_the_boost_grows_while_the_robot_stays_stuck() -> None:
    escape = StuckEscape(EscapeConfig(stuck_window=5 * _MINUTE))

    strengths: list[float] = []
    for minute in range(0, 31, 5):
        beacon = escape.check((3, 3), 2.0, minute * _MINUTE, (4, 3))
        if beacon is not None:
            strengths.append(beacon.strength)

    assert strengths == [1.0, 2.0, 4.0]


def test_an_escape_at_the_wall_marks_the_blocked_cell() -> None:
    escape = StuckEscape(EscapeConfig(stuck_window=5 * _MINUTE, escape_at_wall=True))

    escape.check((3, 3), 2.0, 0, (4, 3))

    assert escape.check((3, 3), 2.0, 5 * _MINUTE, (4, 3)) == EscapeBeacon((4, 3), 1.0)
