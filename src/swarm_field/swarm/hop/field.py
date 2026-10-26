"""The hop fields: the objective fields that turn the robot's ``hop_pull`` into a heading.

``HopField`` follows the TREAT beacons to a pest, ``HopDockField`` the DOCK beacons to a dock.
"""

import logging
from math import hypot
from typing import override

from swarm_field.contracts import BeaconKind, CellId, Time
from swarm_field.swarm.guidance import Heading
from swarm_field.swarm.hop.memory import BeaconView
from swarm_field.swarm.hop.pull import hop_pull
from swarm_field.swarm.hop.relay import BeaconConfig
from swarm_field.swarm.memory import MemoryView

logger = logging.getLogger(__name__)


class HopField:
    """The TREAT objective field of the hop beacons, for one robot.

    It reads only the robot's beacons, never a pest's cell or the terrain.

    Attributes:
        name: How a behaviour asks for this field.
        kind: The beacons it follows.
    """

    name: str = "treat-hop"
    kind: BeaconKind = BeaconKind.TREAT

    def __init__(self, config: BeaconConfig, beacons: BeaconView) -> None:
        self._config: BeaconConfig = config
        self._beacons: BeaconView = beacons

    def guidance_for(self, memory: MemoryView, robot: CellId, now: Time) -> Heading | None:
        """The ``hop_pull`` as a heading, or ``None`` if nothing pulls the robot.

        The heading reaches as far as the strongest attracting beacon, whose cell is the target.
        ``memory`` is not read: the field has the robot's beacons of its own.
        """
        pull = hop_pull(self._beacons, robot, now, self._config, self.kind)
        if pull is None:
            return None
        cell = pull.strongest.beacon.cell
        reach = hypot(cell[0] - robot[0], cell[1] - robot[1])
        length = hypot(pull.vector[0], pull.vector[1])
        offset = (pull.vector[0] / length * reach, pull.vector[1] / length * reach)
        return Heading(offset=offset, target=cell)


class HopDockField(HopField):
    """The DOCK objective field of the hop beacons: a heading down the dock gradient.

    It reads the robot's beacons and the configured dock cells, never the terrain. The heading
    reaches as far as the strongest attracting DOCK beacon, whose cell is the target. When no
    DOCK beacon pulls, it heads straight for the nearest dock cell instead, which the robot's
    steerer walks by what it senses around itself: a robot low on battery or doses keeps going
    until it hears a beacon, rather than idling where it stands.

    Attributes:
        name: How a behaviour asks for this field.
        kind: The beacons it follows.
    """

    name: str = "dock-hop"
    kind: BeaconKind = BeaconKind.DOCK

    def __init__(
        self, config: BeaconConfig, beacons: BeaconView, docks: tuple[CellId, ...] = ()
    ) -> None:
        super().__init__(config, beacons)
        self._docks: tuple[CellId, ...] = docks
        self._heading_straight: bool = False

    @override
    def guidance_for(self, memory: MemoryView, robot: CellId, now: Time) -> Heading | None:
        """The pull of the DOCK beacons as a heading, else a heading to the nearest dock.

        ``None`` without docks, or on a dock. The straight heading is logged once each time it
        takes over from a pull, or the first time, not at every decision.
        """
        pulled = super().guidance_for(memory, robot, now)
        if pulled is not None:
            self._heading_straight = False
            return pulled
        if not self._docks:
            return None
        nearest = min(self._docks, key=lambda dock: hypot(dock[0] - robot[0], dock[1] - robot[1]))
        if nearest == robot:
            return None
        if not self._heading_straight:
            logger.info(
                "No DOCK beacon pulls the robot at %s; heading straight for the dock at %s",
                robot,
                nearest,
            )
            self._heading_straight = True
        offset = (float(nearest[0] - robot[0]), float(nearest[1] - robot[1]))
        return Heading(offset=offset, target=nearest)
