"""Which of two robots goes first when they must not both do the same thing.

Robots that tie (two on one cell heading for the same target, two anchors on one pest, two
killers equally near one pest) must agree, without talking it over, on which one gives way.
Comparing robot ids would do, but the same low-id robots would then always win and always
carry the same burden. So the order is a hash of the robot's id and the current time window:
every robot computes the same order from what it knows, and the order is reshuffled every
``PRIORITY_PERIOD``.
"""

from swarm_field.contracts import RobotId, Time, TimeUnit

PRIORITY_PERIOD: Time = 10 * TimeUnit.MINUTE
"""How long one order of the robots holds before it is reshuffled."""

_MASK = 0xFFFFFFFF


def priority_of(robot: RobotId, now: Time, period: Time = PRIORITY_PERIOD) -> tuple[int, RobotId]:
    """The robot's place in the order at ``now``: smaller goes first; ties go to the lower id."""
    window = now // period
    mixed = (robot * 2654435761 + window * 2246822519 + 374761393) & _MASK
    mixed = ((mixed ^ (mixed >> 15)) * 2246822519) & _MASK
    mixed = ((mixed ^ (mixed >> 13)) * 3266489917) & _MASK
    return (mixed ^ (mixed >> 16), robot)


def ranks_before(robot: RobotId, other: RobotId, now: Time) -> bool:
    """Whether ``robot`` goes before ``other`` at ``now``."""
    return priority_of(robot, now) < priority_of(other, now)
