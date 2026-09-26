"""The communication channel (B.3): range-based delivery of broadcast messages.

A handler hands a robot's outbox to ``Channel.broadcast``, which appends each message to the
inbox of every robot within the sender's radio range; recipients read their inbox at their
next decision.

The payload is opaque here — delivery reads only the envelope's ``sender``.
"""

from collections.abc import Iterable
from math import dist

from swarm_field.contracts import Message, RobotId
from swarm_field.simulation.world import WorldState


def robots_in_range_of(sender: RobotId, world: WorldState) -> list[RobotId]:
    """Every robot that can hear ``sender`` broadcast: all others within ITS radio range.

    Only the sender's range counts — a receiver's own range governs what it transmits, not
    what it picks up. Distance is measured in metres between whole-cell positions: radio
    range is fuzzy, so sub-cell position error is irrelevant.
    """
    origin = world.position_of(sender)
    reach = world.radio_range_of(sender)

    heard_by: list[RobotId] = []
    for robot in world.robot_ids():
        if robot == sender:
            continue
        metres = dist(origin, world.position_of(robot)) * world.field.cell_size
        if metres <= reach:
            heard_by.append(robot)
    return heard_by


class Channel:
    """Carries broadcasts between robots, delivering each by its sender's range.

    Delivered messages wait in the recipient's inbox until it reads them at its next
    decision.
    """

    def __init__(self) -> None:
        self._inboxes: dict[RobotId, list[Message]] = {}

    def broadcast(self, world: WorldState, outbox: Iterable[Message]) -> None:
        """Deliver a robot's outbox now, to the inbox of every robot in the sender's range."""
        for message in outbox:
            for recipient in robots_in_range_of(message.sender, world):
                self._inboxes.setdefault(recipient, []).append(message)

    def take_inbox(self, robot: RobotId) -> list[Message]:
        """Remove and return the messages waiting for ``robot``, oldest first."""
        return self._inboxes.pop(robot, [])
