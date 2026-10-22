"""The message envelope exchanged between robots.

The simulation's channel reads only the header (sender, time) to decide delivery; the
``payload`` is opaque to the simulation and interpreted only inside the algorithm.
"""

from dataclasses import dataclass
from enum import Enum, auto

from swarm_field.contracts.ids import CellId, PestId, RobotId, Time


class MessageKind(Enum):
    """Discriminates message payloads.

    More specialised kinds (stigmergy, claims, ...) can be added later without changing the
    transport layer.

    Attributes:
        HELLO: A placeholder, to test that messages are delivered.
        KNOWLEDGE: What a robot shares of its memory, in a shape the algorithm owns.
        STATUS: A robot's report of itself: a ``PeerObservation`` with its cell, body, remaining
            battery fraction, and optional target/objective.
        BEACON: Pest news passed on hop by hop: a tuple of ``PestBeacon``s.
        GRADIENT: A robot's hop counts to the pests anchored in the swarm, and the killers'
            claims on them: a tuple of ``HopGradient``s and ``HopClaim``s.
    """

    HELLO = auto()
    KNOWLEDGE = auto()
    STATUS = auto()
    BEACON = auto()
    GRADIENT = auto()


@dataclass(frozen=True)
class PestBeacon:
    """News of one pest, passed from robot to robot with a hop count.

    The robot that senses a pest starts its beacon at hop 0; every robot that hears it passes
    it on one hop further, up to a hop limit and for a limited time. The robot that finds the
    pest gone passes on a cleared beacon, which wins over the live one wherever it arrives.

    Attributes:
        pest: The pest, by the id its finder gave it.
        cell: Where the pest is believed to be.
        hops: How many robots the news has passed through since its finder: 0 at the finder.
        found_at: When the finder sensed the pest; the beacon expires a fixed time later.
        cleared: Whether this beacon says the pest has been treated or was not found.
    """

    pest: PestId
    cell: CellId
    hops: int
    found_at: Time
    cleared: bool = False


@dataclass(frozen=True)
class HopGradient:
    """One robot's hop count to a pest, for hop-gradient navigation; it carries no position.

    The anchor, a robot standing on the pest, sends hop 0 with a new ``round`` at each of its
    decisions. A robot keeps the newest round it has heard and, within a round, the fewest
    hops, and passes on its own count, one more than its sender's. A killer follows the
    gradient down: it drives to the neighbour it heard its count from.

    Attributes:
        pest: The pest, by the id its anchor gave it.
        round: The anchor's running number for its refreshes; a newer round wins.
        hops: The sender's hop count to the anchor: 0 at the anchor.
        cleared: Whether this says the pest has been treated or is gone.
    """

    pest: PestId
    round: int
    hops: int
    cleared: bool = False


@dataclass(frozen=True)
class HopClaim:
    """A killer's report that it is following the gradient to a pest, passed on hop by hop.

    When two killers claim the same pest, the one with fewer hops to it keeps it, ties going
    by the order of ``swarm.priority``, and the other looks for another pest; both decide
    this alike from the claims, so they do not both give way.

    Attributes:
        pest: The pest the killer follows.
        killer: The killer.
        hops: The killer's hop count to the pest when it claimed it.
        sent_at: When the killer made the claim; a claim goes stale a fixed time later.
    """

    pest: PestId
    killer: RobotId
    hops: int
    sent_at: Time


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
