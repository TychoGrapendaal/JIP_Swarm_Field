"""Knowledge sharing: what of its memory a robot broadcasts, when, and how it merges what it hears.

Every robot broadcasts its status at every decision point. What it broadcasts of its memory is
the ``KnowledgeSharing``: everything, its infestations only, or nothing. When it broadcasts it is
the ``SharingStrategy``: at every decision, which is exact and the reference, or less often,
which changes what the robots know and when.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import cast

from swarm_field.contracts import Message, MessageKind, Observation, PeerObservation, RobotId
from swarm_field.swarm.memory import AgentMemory, KnowledgeSnapshot

logger = logging.getLogger(__name__)


class KnowledgeSharing(Enum):
    """Which memory facts a robot broadcasts to its peers.

    Attributes:
        EVERYTHING: Its whole memory: terrain, infestation, peer and crop facts.
        INFESTATIONS: Only its infestation facts, e.g. for relays that gossip pests.
        NOTHING: No memory at all, only its status, e.g. for robots that share beacons instead.
    """

    EVERYTHING = auto()
    INFESTATIONS = auto()
    NOTHING = auto()


class SharingStrategy(Enum):
    """When a robot broadcasts its memory.

    Attributes:
        FULL: Its whole memory at every decision. A peer in range then knows at least what the
            robot knew at its last broadcast.
        PERIODIC: Its whole memory every ``SharingConfig.full_every``-th decision, and nothing
            in between.
    """

    FULL = auto()
    PERIODIC = auto()


@dataclass(frozen=True)
class SharingConfig:
    """How a robot shares its memory.

    Attributes:
        strategy: When it broadcasts its memory.
        full_every: Every how many decisions a ``PERIODIC`` robot broadcasts its memory; 0
            means never.
        facts: Which of its memory facts it broadcasts.
    """

    strategy: SharingStrategy = SharingStrategy.FULL
    full_every: int = 20
    facts: KnowledgeSharing = KnowledgeSharing.EVERYTHING

    def describe(self) -> str:
        """The strategy and its parameter, e.g. ``periodic/20`` or ``full``, and any facts but all.

        E.g. ``full/infestations`` for a robot that shares only its infestations.
        """
        when = "full"
        if self.strategy is SharingStrategy.PERIODIC:
            when = f"periodic/{self.full_every}"
        if self.facts is KnowledgeSharing.EVERYTHING:
            return when
        return f"{when}/{self.facts.name.lower()}"


class KnowledgeExchange:
    """One robot's side of knowledge sharing: merges what it hears, and picks when it shares.

    A robot remembers the last snapshot it merged from each peer, so that it can skip the parts
    of the next one that did not change (``AgentMemory.merge_snapshot``). Skipping them changes
    nothing, so sharing stays exact.

    Attributes:
        config: How the robot shares.
        facts_sent: How many facts the robot has broadcast so far, a measure of message size.
    """

    def __init__(self, memory: AgentMemory, config: SharingConfig | None = None) -> None:
        self._memory: AgentMemory = memory
        self.config: SharingConfig = SharingConfig()
        if config is not None:
            self.config = config
        self.facts_sent: int = 0
        self._decisions: int = 0
        self._merged: dict[RobotId, KnowledgeSnapshot] = {}

    def receive(self, inbox: list[Message]) -> None:
        """Merge the status reports and memory snapshots in the inbox, in the order received.

        A knowledge message may also carry a tuple of facts, as a mock source sends; those are
        merged as they are.
        """
        for message in inbox:
            if message.kind is MessageKind.STATUS:
                if not isinstance(message.payload, PeerObservation):
                    logger.warning(
                        "Skipping a status message from %s with payload %r",
                        message.sender,
                        message.payload,
                    )
                    continue
                self._memory.merge((message.payload,))
            elif message.kind is MessageKind.KNOWLEDGE and isinstance(message.payload, tuple):
                facts: list[Observation] = []
                for fact in cast(tuple[object, ...], message.payload):
                    if isinstance(fact, Observation):
                        facts.append(fact)
                self._memory.merge(facts)
            elif message.kind is MessageKind.KNOWLEDGE:
                if not isinstance(message.payload, KnowledgeSnapshot):
                    logger.warning(
                        "Skipping a knowledge message from %s with payload %r",
                        message.sender,
                        message.payload,
                    )
                    continue
                self._memory.merge_snapshot(message.payload, self._merged.get(message.sender))
                self._merged[message.sender] = message.payload

    def knowledge_to_send(self) -> KnowledgeSnapshot | None:
        """The robot's memory to broadcast at this decision point, or ``None`` if not now.

        ``None`` too when the robot shares nothing; with ``INFESTATIONS`` only those facts.
        """
        self._decisions += 1
        config = self.config
        if config.facts is KnowledgeSharing.NOTHING:
            return None
        if config.strategy is SharingStrategy.PERIODIC and (
            config.full_every <= 0 or self._decisions % config.full_every != 0
        ):
            return None
        snapshot = self._memory.knowledge()
        if config.facts is KnowledgeSharing.INFESTATIONS:
            snapshot = snapshot.only_infestations()
        self.facts_sent += len(snapshot)
        return snapshot
