"""The hop field: an objective field that sums the attraction of downhill beacons.

The robot's own value is the lowest fresh beacon it heard plus the hop to it (``own_value``). Every
fresh beacon below that value attracts the robot towards its cell, with its
``beacon_strength``. The sum of those attractions is the robot's ``Heading``. The field reads
only beacons, never a pest's cell. With ``killer_repulsion``, a killer's beacon pushes
instead.
"""

from dataclasses import dataclass
from math import hypot

from swarm_field.contracts import BodyType, CellId, RobotId, Time, Vector2
from swarm_field.swarm.beacon import BeaconConfig, beacon_strength, downhill_beacons
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
        """The ``hop_pull`` as a heading, or ``None`` if nothing pulls the robot.

        The heading reaches as far as the strongest attracting beacon, whose cell is the target.
        """
        pull = hop_pull(memory, robot, now, self._config)
        if pull is None:
            return None
        cell = pull.strongest.beacon.cell
        reach = hypot(cell[0] - robot[0], cell[1] - robot[1])
        length = hypot(pull.vector[0], pull.vector[1])
        offset = (pull.vector[0] / length * reach, pull.vector[1] / length * reach)
        return Heading(offset=offset, target=cell)


@dataclass(frozen=True)
class HopPull:
    """The summed pull of the downhill beacons on a robot.

    Attributes:
        vector: The sum of every beacon's pull, or push for a killer's beacon.
        strongest: The strongest attracting beacon.
        strength: That beacon's strength.
    """

    vector: Vector2
    strongest: HeardBeacon
    strength: float


def hop_pull(memory: MemoryView, robot: CellId, now: Time, config: BeaconConfig) -> HopPull | None:
    """The pull of the ``downhill_beacons``, or ``None`` if no beacon attracts the robot.

    A killer's beacon pushes with ``killer_repulsion`` times its strength instead, so that a
    killer closer to the pest counters the pull towards it. Pulls that cancel out exactly
    count as no pull.
    """
    killers: set[RobotId] = set()
    if config.killer_repulsion > 0:
        killers = _killers(memory)
    pull_x = 0.0
    pull_y = 0.0
    strongest: HeardBeacon | None = None
    strongest_strength = 0.0
    for heard in downhill_beacons(memory, robot, now, config):
        dx = heard.beacon.cell[0] - robot[0]
        dy = heard.beacon.cell[1] - robot[1]
        distance = hypot(dx, dy)
        strength = beacon_strength(heard, now, config)
        if heard.sender in killers:
            strength *= -config.killer_repulsion
        pull_x += strength * dx / distance
        pull_y += strength * dy / distance
        if strength > strongest_strength:
            strongest = heard
            strongest_strength = strength
    if strongest is None or hypot(pull_x, pull_y) == 0:
        return None
    return HopPull(vector=(pull_x, pull_y), strongest=strongest, strength=strongest_strength)


def _killers(memory: MemoryView) -> set[RobotId]:
    """The peers the robot knows to be killers."""
    killers: set[RobotId] = set()
    for peer in memory.peers():
        if peer.body is BodyType.KILLER and peer.peer is not None:
            killers.add(peer.peer)
    return killers
