"""The hop field: an objective field that sums the attraction of downhill beacons.

The robot's own value is the lowest fresh beacon it heard plus the hop to it (``own_value``). Every
fresh beacon below that value attracts the robot towards its cell, with its
``beacon_strength``. The sum of those attractions is the robot's ``Heading``. The field reads
only beacons, never a pest's cell.
"""

from math import hypot

from swarm_field.contracts import AgentObjective, CellId, Time
from swarm_field.swarm.beacon import BeaconConfig, beacon_strength, fresh_beacons, own_value
from swarm_field.swarm.guidance import Heading
from swarm_field.swarm.memory import HeardBeacon, MemoryView


class HopField:
    """The TREAT objective field of the hop beacons.

    Attributes:
        name: How a behaviour asks for this field.
    """

    name: str = "treat-hop"

    def __init__(self, config: BeaconConfig) -> None:
        self._config: BeaconConfig = config

    def guidance_for(self, memory: MemoryView, robot: CellId, now: Time) -> Heading | None:
        """The summed attraction of the downhill beacons, or ``None`` if none pulls the robot.

        The heading reaches as far as the strongest downhill beacon, whose cell is the target.
        Beacons on the robot's own cell pull in no direction, so they are left out.
        """
        fresh = fresh_beacons(memory, AgentObjective.TREAT, now, self._config)
        if not fresh:
            return None
        here = own_value(fresh, robot, self._config)

        pull_x = 0.0
        pull_y = 0.0
        strongest: HeardBeacon | None = None
        strongest_strength = 0.0
        for heard in fresh:
            if heard.beacon.value >= here:
                continue
            dx = heard.beacon.cell[0] - robot[0]
            dy = heard.beacon.cell[1] - robot[1]
            distance = hypot(dx, dy)
            if distance == 0:
                continue
            strength = beacon_strength(heard, now, self._config)
            pull_x += strength * dx / distance
            pull_y += strength * dy / distance
            if strength > strongest_strength:
                strongest = heard
                strongest_strength = strength

        pull = hypot(pull_x, pull_y)
        if strongest is None or pull == 0:
            return None
        cell = strongest.beacon.cell
        reach = hypot(cell[0] - robot[0], cell[1] - robot[1])
        return Heading(offset=(pull_x / pull * reach, pull_y / pull * reach), target=cell)
