"""Knowledge sharing: when a robot broadcasts its memory, and how it merges what it hears.

Every robot broadcasts its status at every decision point. When it broadcasts its memory is the
``SharingStrategy``: at every decision, which is exact and the reference, or less often, which
changes what the robots know and when.
"""

import logging
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import cast

import numpy as np

from swarm_field.contracts import (
    InfestationObservation,
    Message,
    MessageKind,
    PeerObservation,
    RobotId,
)
from swarm_field.swarm.memory import AgentMemory, FactColumns, KnowledgeSnapshot

logger = logging.getLogger(__name__)


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
    """

    strategy: SharingStrategy = SharingStrategy.FULL
    full_every: int = 20

    def describe(self) -> str:
        """The strategy and its parameter, e.g. ``periodic/20``, or ``full``."""
        match self.strategy:
            case SharingStrategy.FULL:
                return "full"
            case SharingStrategy.PERIODIC:
                return f"periodic/{self.full_every}"


class KnowledgeExchange:
    """One robot's side of knowledge sharing: merges what it hears, and picks when it shares.

    A robot remembers the last snapshot it merged from each peer, so that it can skip the parts
    of the next one that did not change (``AgentMemory.merge_snapshot``). Skipping them changes
    nothing, so sharing stays exact.

    Attributes:
        config: How the robot shares.
        facts_sent: How many facts the robot has broadcast so far, a measure of message size.
        keep_pests: Whether the robot learns pests from its peers' snapshots. Hop-based
            navigation turns this off, so that pests reach a robot only as hop messages; clean
            cells are learned either way, since exploring robots use them.
    """

    def __init__(self, memory: AgentMemory, config: SharingConfig | None = None) -> None:
        self._memory: AgentMemory = memory
        self.config: SharingConfig = SharingConfig()
        if config is not None:
            self.config = config
        self.facts_sent: int = 0
        self._decisions: int = 0
        self._merged: dict[RobotId, KnowledgeSnapshot] = {}
        self.keep_pests: bool = True
        self._clean: dict[RobotId, tuple[FactColumns, FactColumns]] = {}

    def receive(self, inbox: list[Message]) -> None:
        """Merge the status reports and memory snapshots in the inbox, in the order received."""
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
            elif message.kind is MessageKind.KNOWLEDGE:
                if not isinstance(message.payload, KnowledgeSnapshot):
                    logger.warning(
                        "Skipping a knowledge message from %s with payload %r",
                        message.sender,
                        message.payload,
                    )
                    continue
                snapshot = message.payload
                if not self.keep_pests:
                    snapshot = self._without_pests(message.sender, snapshot)
                self._memory.merge_snapshot(snapshot, self._merged.get(message.sender))
                self._merged[message.sender] = snapshot

    def _without_pests(self, sender: RobotId, snapshot: KnowledgeSnapshot) -> KnowledgeSnapshot:
        """``snapshot`` without its infested cells.

        The filtered infestation is reused while the sender's infestation part is the same
        object, so that ``AgentMemory.merge_snapshot`` can still skip it.
        """
        known = self._clean.get(sender)
        if known is not None and known[0] is snapshot.infestation:
            return replace(snapshot, infestation=known[1])
        columns = snapshot.infestation
        facts = cast(list[InfestationObservation], columns.facts.tolist())
        clean = np.array([fact.severity <= 0 for fact in facts], dtype=np.bool_)
        filtered = FactColumns(
            cells=columns.cells[clean],
            observed_at=columns.observed_at[clean],
            facts=columns.facts[clean],
        )
        self._clean[sender] = (columns, filtered)
        return replace(snapshot, infestation=filtered)

    def knowledge_to_send(self) -> KnowledgeSnapshot | None:
        """The robot's memory to broadcast at this decision point, or ``None`` if not now."""
        self._decisions += 1
        config = self.config
        if config.strategy is SharingStrategy.PERIODIC and (
            config.full_every <= 0 or self._decisions % config.full_every != 0
        ):
            return None
        snapshot = self._memory.knowledge()
        self.facts_sent += len(snapshot)
        return snapshot
