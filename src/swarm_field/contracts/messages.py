"""The message envelope exchanged between robots.

The simulation's channel reads only the header (sender, time) to decide delivery; the
``payload`` is opaque to the simulation and interpreted only inside the algorithm.
"""

from dataclasses import dataclass
from enum import Enum, auto

from swarm_field.contracts.ids import CellId, RobotId, Time


class MessageKind(Enum):
    """Discriminates message payloads.

    More specialised kinds (stigmergy, claims, ...) can be added later without changing the
    transport layer.

    Attributes:
        HELLO: A placeholder, to test that messages are delivered.
        KNOWLEDGE: What a robot shares of its memory, in a shape the algorithm owns, or a tuple
            of facts from a mock source.
        STATUS: A robot's report of itself: a ``PeerObservation`` with its cell, body, remaining
            battery fraction, and optional target/objective.
        BEACON: A robot's hop values, as a tuple of ``Beacon``s, one per kind.
    """

    HELLO = auto()
    KNOWLEDGE = auto()
    STATUS = auto()
    BEACON = auto()


class BeaconKind(Enum):
    """What a beacon marks.

    Attributes:
        TREAT: Guides killers to a pest. A robot that saw the pest is its source, and robots
            relay it.
        DOCK: Guides robots to a dock. Every dock broadcasts it from its own cell, and robots
            relay it.
    """

    TREAT = auto()
    DOCK = auto()


@dataclass(frozen=True)
class Beacon:
    """A hop value of one kind, which robots relay to guide each other.

    Attributes:
        kind: What the beacon marks.
        value: How far the marked place is, in hops: whole hops, or distances in radio ranges
            with metric hops. 0 is the place itself.
        cell: Where the value holds: the sender's cell, or the marked cell for a source.
    """

    kind: BeaconKind
    value: float
    cell: CellId


@dataclass(frozen=True)
class Message:
    """One broadcast message. The channel delivers it; only the algorithm reads its payload.

    Attributes:
        sender: The robot that sent the message.
        sent_at: Time at which the message was sent.
        kind: Discriminates how to interpret the payload.
        payload: Opaque to the simulation; the algorithm owns its shape.
    """

    sender: RobotId
    sent_at: Time
    kind: MessageKind
    payload: object
