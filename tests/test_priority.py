"""Tests for the order robots give way in."""

from swarm_field.contracts import TimeUnit
from swarm_field.swarm.priority import PRIORITY_PERIOD, priority_of, ranks_before


def test_the_order_is_the_same_for_every_robot_within_a_period() -> None:
    early = sorted(range(10), key=lambda robot: priority_of(robot, 0))
    late = sorted(range(10), key=lambda robot: priority_of(robot, PRIORITY_PERIOD - 1))

    assert early == late
    assert ranks_before(early[0], early[1], 0)
    assert not ranks_before(early[1], early[0], 0)


def test_the_order_is_reshuffled_so_no_robot_always_goes_first() -> None:
    firsts = set()
    for window in range(20):
        now = window * PRIORITY_PERIOD + TimeUnit.MINUTE
        firsts.add(min(range(10), key=lambda robot: priority_of(robot, now)))

    assert len(firsts) > 3
