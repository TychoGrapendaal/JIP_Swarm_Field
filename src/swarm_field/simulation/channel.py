"""The communication channel (B.3): range-based delivery of broadcast messages.

A handler hands a robot's outbox to ``Channel.broadcast``, which appends each message to the
inbox of every robot within the sender's radio range; recipients read their inbox at their
next decision.

The payload is opaque here — delivery reads only the envelope's ``sender``.
"""

from collections.abc import Iterable
from math import dist

from swarm_field.contracts import Message, RobotId, Time
from swarm_field.simulation.world import WorldState


def robots_in_range_of(sender: RobotId, world: WorldState, time: Time) -> list[RobotId]:
    """Every robot that can hear ``sender`` broadcast: all others within ITS radio range.

    Only the sender's range counts — a receiver's own range governs what it transmits, not
    what it picks up. Distance is measured in metres between whole-cell positions: radio
    range is fuzzy, so sub-cell position error is irrelevant.
    """
    origin = world.continuous_position(sender, time)
    reach = world.robots[sender].radio_range_m

    heard_by: list[RobotId] = []
    for robot in world.robots:
        if robot == sender:
            continue
        metres = dist(origin, world.continuous_position(robot, time)) * world.field.cell_size
        if metres <= reach:
            heard_by.append(robot)
    return heard_by


class Channel:
    """Carries broadcasts between robots, delivering each by its sender's range.

    Delivered messages wait in the recipient's inbox until it reads them at its next
    decision.

    Attributes:
        sent: How many messages have been handed to the channel since the run began. The
            payload is never read, so this counts broadcasts rather than deliveries.
    """

    def __init__(self) -> None:
        self._inboxes: dict[RobotId, list[Message]] = {}
        self.sent: int = 0

    def broadcast(self, world: WorldState, time: Time, outbox: Iterable[Message]) -> None:
        """Deliver a robot's outbox now, to the inbox of every robot in the sender's range."""
        for message in outbox:
            self.sent += 1
            for recipient in robots_in_range_of(message.sender, world, time):
                self._inboxes.setdefault(recipient, []).append(message)

    def take_inbox(self, robot: RobotId) -> list[Message]:
        """Remove and return the messages waiting for ``robot``."""
        return self._inboxes.pop(robot, [])
