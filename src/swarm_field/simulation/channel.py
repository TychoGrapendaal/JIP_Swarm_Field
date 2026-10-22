"""The communication channel (B.3): range-based delivery of broadcast messages.

A handler hands a robot's outbox to ``Channel.broadcast``, which appends each message to the
inbox of every robot within the sender's radio range; recipients read their inbox at their
next decision. ``Channel.broadcast_from_cell`` delivers a message sent from a cell instead, for
the mock pest sources.

The payload is opaque here — delivery reads only the envelope's ``sender``.
"""

from collections.abc import Iterable
from math import dist

from swarm_field.contracts import CellId, Message, RobotId, Time
from swarm_field.simulation.world import WorldState

_MAX_MOVE_CELLS = 1.5
"""How far a moving robot can be from its cell's centre, in cells, with a margin: a diagonal
step is about 1.41 cells."""


def robots_in_range_of(sender: RobotId, world: WorldState, time: Time) -> list[RobotId]:
    """Every robot that can hear ``sender`` broadcast: all others within ITS radio range.

    Only the sender's range counts — a receiver's own range governs what it transmits, not
    what it picks up.
    """
    origin = world.continuous_position(sender, time)
    return robots_within(origin, world.robots[sender].radio_range_m, world, time, exclude=sender)


def robots_within(
    origin: tuple[float, float],
    reach_m: float,
    world: WorldState,
    time: Time,
    exclude: RobotId | None = None,
) -> list[RobotId]:
    """Every robot within ``reach_m`` metres of ``origin``, a continuous position in cells.

    Distance is measured in metres between whole-cell positions: radio range is fuzzy, so
    sub-cell position error is irrelevant. A robot is never more than one diagonal step from
    the centre of its own cell, so a robot whose cell is further than the reach plus that step
    is skipped without working out where along its move it is.
    """
    cell_size = world.field.cell_size
    reach_cells = reach_m / cell_size + _MAX_MOVE_CELLS
    heard_by: list[RobotId] = []
    for robot, state in world.robots.items():
        if robot == exclude:
            continue
        x, y = state.position
        if max(abs(x + 0.5 - origin[0]), abs(y + 0.5 - origin[1])) > reach_cells:
            continue
        metres = dist(origin, world.continuous_position(robot, time)) * cell_size
        if metres <= reach_m:
            heard_by.append(robot)
    return heard_by


class Channel:
    """Carries broadcasts between robots, delivering each by its sender's range.

    Delivered messages wait in the recipient's inbox until it reads them at its next
    decision.

    Attributes:
        sent: How many messages have been handed to the channel since the run began. Every
            robot broadcasts at every decision, so this follows the number of decisions.
        received: How many times a message reached a robot in range since the run began: a
            broadcast heard by three robots counts three times. This measures how much the
            robots actually communicated.
    """

    def __init__(self) -> None:
        self._inboxes: dict[RobotId, list[Message]] = {}
        self.sent: int = 0
        self.received: int = 0

    def broadcast(self, world: WorldState, time: Time, outbox: Iterable[Message]) -> None:
        """Deliver a robot's outbox now, to the inbox of every robot in the sender's range.

        Who is in range is worked out once per sender, since every message of the outbox is
        sent at the same time.
        """
        in_range: dict[RobotId, list[RobotId]] = {}
        for message in outbox:
            self.sent += 1
            if message.sender not in in_range:
                in_range[message.sender] = robots_in_range_of(message.sender, world, time)
            for recipient in in_range[message.sender]:
                self._inboxes.setdefault(recipient, []).append(message)
                self.received += 1

    def broadcast_from_cell(
        self, world: WorldState, time: Time, cell: CellId, reach_m: float, message: Message
    ) -> None:
        """Deliver a message sent from ``cell`` itself rather than by a robot.

        Not counted in ``sent`` or ``received``, which measure the robots' own communication.
        """
        origin = (cell[0] + 0.5, cell[1] + 0.5)
        for recipient in robots_within(origin, reach_m, world, time):
            self._inboxes.setdefault(recipient, []).append(message)

    def take_inbox(self, robot: RobotId) -> list[Message]:
        """Remove and return the messages waiting for ``robot``."""
        return self._inboxes.pop(robot, [])
