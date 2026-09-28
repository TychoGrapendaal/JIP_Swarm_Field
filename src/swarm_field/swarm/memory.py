"""The robot's local memory: a collection of facts about the field and its peers.

A fact is a contract ``Observation``, stored as it arrived. Memory holds at most one fact per
identity (the cell for terrain and infestation, the peer's id for peers) and keeps the most
recently observed one. Confidence decays with age, but is computed when read, so nothing in
memory changes as time passes.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from math import exp
from typing import Protocol

from swarm_field.contracts import (
    CellId,
    FieldGeometry,
    InfestationObservation,
    Observation,
    ObservationBase,
    PeerObservation,
    Perception,
    RobotId,
    Terrain,
    TerrainObservation,
    Time,
    TimeUnit,
)


@dataclass(frozen=True)
class DecayConfig:
    """Decay time per fact kind: after one decay time, confidence has dropped to 1/e.

    Terrain is static and never decays.

    Attributes:
        infestation: Decay time of infestation facts.
        peer: Decay time of peer facts.
    """

    infestation: Time = 24 * TimeUnit.HOUR
    peer: Time = 10 * TimeUnit.MINUTE

    def decay_time_of(self, fact: Observation) -> Time | None:
        """The decay time of the fact's kind, or ``None`` if it does not decay."""
        match fact:
            case InfestationObservation():
                return self.infestation
            case PeerObservation():
                return self.peer
            case _:
                return None


class MemoryView(Protocol):
    """The read-only queries on memory that behaviours and objectives are typed against."""

    def terrain_at(self, cell: CellId) -> Terrain | None: ...

    def infestation_at(self, cell: CellId) -> InfestationObservation | None: ...

    def infestations(self) -> Iterator[InfestationObservation]: ...

    def peers(self) -> Iterator[PeerObservation]: ...

    def confidence(self, fact: Observation, now: Time) -> float: ...


class AgentMemory:
    """One robot's facts about the field and its peers, merged from its own sensing and peers.

    Terrain, infestation and peer facts are stored; crop facts, anonymous peer sightings and
    sightings of the robot itself are ignored.
    """

    def __init__(
        self, robot_id: RobotId, field: FieldGeometry, decay: DecayConfig | None = None
    ) -> None:
        self._robot_id: RobotId = robot_id
        self._field: FieldGeometry = field
        self._decay: DecayConfig = DecayConfig()
        if decay is not None:
            self._decay = decay
        self._terrain: dict[CellId, TerrainObservation] = {}
        self._infestation: dict[CellId, InfestationObservation] = {}
        self._peers: dict[RobotId, PeerObservation] = {}

    @property
    def width(self) -> int:
        return self._field.width

    @property
    def height(self) -> int:
        return self._field.height

    def ingest(self, perception: Perception) -> None:
        """Merge the robot's own observations from one decision point."""
        self.merge(perception.sensed)

    def merge(self, facts: Iterable[Observation]) -> None:
        """Merge facts, keeping per identity the one with the newest ``observed_at``.

        A fact replaces the stored one only if it is strictly newer, so on a tie the robot
        keeps what it already knows.
        """
        for fact in facts:
            match fact:
                case TerrainObservation():
                    _keep_newest(self._terrain, fact.cell, fact)
                case InfestationObservation():
                    _keep_newest(self._infestation, fact.cell, fact)
                case PeerObservation(peer=int() as peer) if peer != self._robot_id:
                    _keep_newest(self._peers, peer, fact)
                case _:
                    pass

    def records(self) -> Iterator[Observation]:
        """Every stored fact, e.g. to share with a peer."""
        return chain(self._terrain.values(), self._infestation.values(), self._peers.values())

    def snapshot(self) -> tuple[Observation, ...]:
        """All stored facts, for ``AgentInspection.belief``."""
        return tuple(self.records())

    def terrain_at(self, cell: CellId) -> Terrain | None:
        """The known terrain of a cell, or ``None`` if unknown.

        Cells beyond the field's bounding box are ``Terrain.OUTSIDE`` without being observed.
        """
        x, y = cell
        if not (0 <= x < self._field.width and 0 <= y < self._field.height):
            return Terrain.OUTSIDE
        fact = self._terrain.get(cell)
        if fact is None:
            return None
        return fact.terrain

    def infestation_at(self, cell: CellId) -> InfestationObservation | None:
        return self._infestation.get(cell)

    def infestations(self) -> Iterator[InfestationObservation]:
        """Every infestation fact, including cells observed to be clean (severity 0)."""
        return iter(self._infestation.values())

    def peers(self) -> Iterator[PeerObservation]:
        return iter(self._peers.values())

    def confidence(self, fact: Observation, now: Time) -> float:
        """How far a fact can be trusted at ``now``: its initial confidence, decayed by age."""
        decay_time = self._decay.decay_time_of(fact)
        if decay_time is None:
            return fact.confidence
        return fact.confidence * exp(-(now - fact.observed_at) / decay_time)


def _keep_newest[K, F: ObservationBase](store: dict[K, F], key: K, fact: F) -> None:
    """Store ``fact`` under ``key`` unless the stored fact is at least as new."""
    stored = store.get(key)
    if stored is None or fact.observed_at > stored.observed_at:
        store[key] = fact
