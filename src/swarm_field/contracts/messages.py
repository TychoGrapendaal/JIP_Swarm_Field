"""The message envelope exchanged between robots.

The simulation's channel reads only the header (sender, time) to decide delivery; the
``payload`` is opaque to the simulation and interpreted only inside the algorithm.
"""

from dataclasses import dataclass
from enum import Enum, auto

from swarm_field.contracts.ids import AgentObjective, CellId, RobotId, Time


class MessageKind(Enum):
    """Discriminates message payloads.

    More specialised kinds (stigmergy, claims, ...) can be added later without changing the
    transport layer.

    Attributes:
        HELLO: A placeholder, to test that messages are delivered.
        KNOWLEDGE: A robot's current memory snapshot, as a tuple of facts.
        STATUS: A robot's report of itself: a ``PeerObservation`` of its own cell and the
            target it is heading for.
        BEACON: A robot's hop values, as a tuple of ``Beacon``s, one per objective.
    """

    HELLO = auto()
    KNOWLEDGE = auto()
    STATUS = auto()
    BEACON = auto()


@dataclass(frozen=True)
class Beacon:
    """A hop value for one objective, which robots relay to guide each other towards it.

    Attributes:
        objective: The objective the value guides to.
        value: How far the objective is, in hops. 0 is the objective itself.
        cell: Where the value holds: the sender's cell, or the objective's cell for a source.
    """

    objective: AgentObjective
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
