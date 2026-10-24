"""The hop field: the objective field that turns the robot's ``hop_pull`` into a heading."""

from math import hypot

from swarm_field.contracts import CellId, Time
from swarm_field.swarm.guidance import Heading
from swarm_field.swarm.hop.memory import BeaconView
from swarm_field.swarm.hop.pull import hop_pull
from swarm_field.swarm.hop.relay import BeaconConfig
from swarm_field.swarm.memory import MemoryView


class HopField:
    """The TREAT objective field of the hop beacons, for one robot.

    It reads only the robot's beacons, never a pest's cell or the terrain.

    Attributes:
        name: How a behaviour asks for this field.
    """

    name: str = "treat-hop"

    def __init__(self, config: BeaconConfig, beacons: BeaconView) -> None:
        self._config: BeaconConfig = config
        self._beacons: BeaconView = beacons

    def guidance_for(self, memory: MemoryView, robot: CellId, now: Time) -> Heading | None:
        """The ``hop_pull`` as a heading, or ``None`` if nothing pulls the robot.

        The heading reaches as far as the strongest attracting beacon, whose cell is the target.
        ``memory`` is not read: the field has the robot's beacons of its own.
        """
        pull = hop_pull(self._beacons, robot, now, self._config)
        if pull is None:
            return None
        cell = pull.strongest.beacon.cell
        reach = hypot(cell[0] - robot[0], cell[1] - robot[1])
        length = hypot(pull.vector[0], pull.vector[1])
        offset = (pull.vector[0] / length * reach, pull.vector[1] / length * reach)
        return Heading(offset=offset, target=cell)
