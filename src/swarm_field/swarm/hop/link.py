"""A robot's link to the beacon network: what it hears, what it places, and what it sends.

``ComposedAgent`` holds one ``BeaconLink`` per hop robot and calls it at three points of a
decision: ``perceive`` with the perception, ``after_step`` with the heading the robot followed
and the action it took, and ``beacon_message`` for the outbox. Everything the hop method adds
to a brain lives here, so the brain itself knows nothing of beacons.
"""

import logging
from typing import cast

from swarm_field.contracts import (
    Action,
    Beacon,
    CellId,
    FieldGeometry,
    InfestationObservation,
    Message,
    MessageKind,
    PeerObservation,
    Perception,
    RobotId,
    Time,
    Treat,
)
from swarm_field.swarm.guidance import Heading
from swarm_field.swarm.hop.escape import StuckEscape, cell_ahead
from swarm_field.swarm.hop.memory import BeaconMemory
from swarm_field.swarm.hop.pull import obstacle_strength
from swarm_field.swarm.hop.relay import BeaconConfig, treat_beacon
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

    def __init__(self, robot_id: RobotId, config: BeaconConfig, field: FieldGeometry) -> None:
        """Link one robot to the beacon network.

        Args:
            robot_id: The robot the link belongs to.
            config: How beacons are kept, relayed and weighed.
            field: The field, for the surroundings' bounds.
        """
        self.config: BeaconConfig = config
        self.memory: BeaconMemory = BeaconMemory()
        self.surroundings: SensedSurroundings = SensedSurroundings(field.width, field.height)
        self._robot_id: RobotId = robot_id
        self._cell_size: float = field.cell_size
        self._speed_mps: float = 0.0
        self._escape: StuckEscape | None = None
        if config.escape is not None:
            self._escape = StuckEscape(config.escape)

    def perceive(self, perception: Perception) -> None:
        """Take in a decision's sensing and the beacons and statuses it received.

        First the memory forgets what no longer counts: beacons older than ``fresh_for``,
        sightings older than ``source_for``, and placed beacons that faded.
        """
        now = perception.time
        self._speed_mps = perception.self_state.speed_mps
        self.memory.forget(now - self.config.fresh_for, now - self.config.source_for, now)
        self.surroundings.sense(perception.sensed)
        for fact in perception.sensed:
            if isinstance(fact, InfestationObservation):
                self.memory.see(fact.cell, fact.observed_at, fact.severity > 0)
        for message in perception.inbox:
            if message.kind is MessageKind.BEACON:
                self._hear(message)
            elif message.kind is MessageKind.STATUS and isinstance(
                message.payload, PeerObservation
            ):
                self.memory.note_body(message.sender, message.payload.body, message.sent_at)

    def after_step(self, heading: Heading | None, action: Action, robot: CellId, now: Time) -> None:
        """Escape when the robot circles in place: place the beacon its ``StuckEscape`` hands out.

        A treatment starts the escape's record over: the robot stands still on purpose.

        Args:
            heading: The heading the robot followed at this decision, or ``None``.
            action: The action the robot took at this decision.
            robot: The robot's cell.
            now: The time of the decision.
        """
        if self._escape is None:
            return
        if isinstance(action, Treat):
            self._escape.forget()
            return
        ahead: CellId | None = None
        if heading is not None:
            ahead = cell_ahead(robot, heading.offset)
        speed = self._speed_mps / self._cell_size
        escape = self._escape.check(robot, ahead, now, speed)
        if escape is not None:
            self._place_stronger(escape.cell, now, escape.strength, escape.decay)

    def beacon_message(self, perception: Perception) -> Message | None:
        """The robot's TREAT beacon as a message, or ``None`` if it has no beacon."""
        beacon = treat_beacon(
            self.memory, perception.self_state.position, perception.time, self.config
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
