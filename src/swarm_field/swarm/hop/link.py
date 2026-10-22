"""A robot's link to the beacon network: what it hears, what it places, and what it sends.

``ComposedAgent`` holds one ``BeaconLink`` per hop robot and calls it at three points of a
decision: ``perceive`` with the perception, ``after_step`` with the heading the robot followed,
and ``beacon_message`` for the outbox. Everything the hop method adds to a brain lives here, so
the brain itself knows nothing of beacons.
"""

import logging
from typing import cast

from swarm_field.contracts import (
    Beacon,
    CellId,
    FieldGeometry,
    Message,
    MessageKind,
    PeerObservation,
    Perception,
    RobotId,
    Time,
)
from swarm_field.swarm.guidance import Heading
from swarm_field.swarm.hop.escape import StuckEscape, cell_ahead
from swarm_field.swarm.hop.memory import BeaconMemory
from swarm_field.swarm.hop.pull import obstacle_strength
from swarm_field.swarm.hop.relay import BeaconConfig, treat_beacon
from swarm_field.swarm.memory import MemoryView
from swarm_field.swarm.steering import SensedSurroundings

logger = logging.getLogger(__name__)


class BeaconLink:
    """One robot's beacons, the surroundings it steers by, and its escape.

    Attributes:
        config: How beacons are kept, relayed and weighed.
        memory: The robot's beacons, heard and placed; the hop rules read it as a
            ``BeaconView``.
        surroundings: What the robot sensed around itself at its last decision, and nothing
            older; the navigator steers headings by it, without a map.
    """

    def __init__(
        self, robot_id: RobotId, config: BeaconConfig, field: FieldGeometry, sightings: MemoryView
    ) -> None:
        """Link one robot to the beacon network.

        Args:
            robot_id: The robot the link belongs to.
            config: How beacons are kept, relayed and weighed.
            field: The field, for the surroundings' bounds.
            sightings: Where the robot's own pest sightings are read, for its source beacon.
        """
        self.config: BeaconConfig = config
        self.memory: BeaconMemory = BeaconMemory()
        self.surroundings: SensedSurroundings = SensedSurroundings(field.width, field.height)
        self._robot_id: RobotId = robot_id
        self._sightings: MemoryView = sightings
        self._escape: StuckEscape | None = None
        if config.escape is not None:
            self._escape = StuckEscape(config.escape)

    def perceive(self, perception: Perception) -> None:
        """Take in a decision's sensing and the beacons and statuses it received."""
        self.surroundings.sense(perception.sensed)
        for message in perception.inbox:
            if message.kind is MessageKind.BEACON:
                self._hear(message)
            elif message.kind is MessageKind.STATUS and isinstance(
                message.payload, PeerObservation
            ):
                self.memory.note_body(message.sender, message.payload.body)

    def after_step(self, heading: Heading | None, robot: CellId, now: Time) -> None:
        """Mark the wall the decision's heading runs into, and escape when the robot circles.

        Whether the cell ahead is a wall comes from ``surroundings``, this decision's sensing.
        The wall gets a beacon with ``BeaconConfig.obstacle_repulsion``; a robot that circles in
        place gets an escape beacon from its ``StuckEscape``. A decision without a heading starts
        the escape's window over.

        Args:
            heading: The heading the robot followed at this decision, or ``None``.
            robot: The robot's cell.
            now: The time of the decision.
        """
        if heading is None:
            if self._escape is not None:
                self._escape.forget()
            return
        ahead = cell_ahead(robot, heading.offset)
        if self.config.obstacle_repulsion > 0 and self.surroundings.blocks(ahead):
            self._place_stronger(
                ahead, now, self.config.obstacle_repulsion, self.config.obstacle_decay
            )
        if self._escape is not None:
            escape = self._escape.check(robot, ahead, now)
            if escape is not None:
                self._place_stronger(escape.cell, now, escape.strength, escape.decay)

    def beacon_message(self, perception: Perception) -> Message | None:
        """The robot's TREAT beacon as a message, or ``None`` if it has no beacon."""
        beacon = treat_beacon(
            self.memory,
            self._sightings,
            self._robot_id,
            perception.self_state.position,
            perception.time,
            self.config,
        )
        if beacon is None:
            return None
        return Message(
            sender=self._robot_id,
            sent_at=perception.time,
            kind=MessageKind.BEACON,
            payload=(beacon,),
        )

    def _hear(self, message: Message) -> None:
        """Store the beacons of a beacon message, or warn about a malformed one."""
        beacons: list[Beacon] = []
        if isinstance(message.payload, tuple):
            for item in cast(tuple[object, ...], message.payload):
                if isinstance(item, Beacon):
                    beacons.append(item)
        if not beacons:
            logger.warning(
                "Skipping a beacon message from %s with payload %r",
                message.sender,
                message.payload,
            )
            return
        self.memory.hear(message.sender, message.sent_at, beacons)

    def _place_stronger(self, cell: CellId, now: Time, strength: float, decay: Time) -> None:
        """Place a beacon on ``cell``, unless the push already there is as strong."""
        push = 0.0
        placed = self.memory.placed_beacon_at(cell)
        if placed is not None:
            push = obstacle_strength(placed, now)
        if strength > push:
            self.memory.place(cell, now, strength, decay)
