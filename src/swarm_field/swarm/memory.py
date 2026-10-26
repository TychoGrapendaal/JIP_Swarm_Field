"""The robot's local memory: a collection of facts about the field and its peers.

A fact is a contract ``Observation``, stored as it arrived. Memory holds at most one fact per
identity (the cell for terrain and infestation, the peer's id for peers) and keeps the most
recently observed one. Confidence decays with age, but is computed when read, so nothing in
memory changes as time passes.

Terrain and infestation facts are kept in arrays over the field (``CellFacts``): per cell the
fact itself, when it was observed, and when it was stored. Sharing the memory with a peer is
then a copy of a few arrays (``KnowledgeSnapshot``), and merging it a few vectorised operations
rather than one dict lookup per fact. The arrays also serve seeders that read every cell at
once: when each cell was last observed, and whether it is worth monitoring.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from itertools import chain
from math import exp
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import (
    CellId,
    CropObservation,
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

logger = logging.getLogger(__name__)

_TERRAINS: tuple[Terrain, ...] = tuple(Terrain)
"""Every terrain, indexed by the code ``AgentMemory`` stores it as."""

_UNKNOWN_TERRAIN = -1
"""The terrain code of a cell whose terrain is not known."""

_FREE = _TERRAINS.index(Terrain.FREE)


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
    """The read-only queries on memory that behaviours and field seeders are typed against."""

    def terrain_at(self, cell: CellId) -> Terrain | None: ...

    def infestation_at(self, cell: CellId) -> InfestationObservation | None: ...

    def infestations(self) -> Iterator[InfestationObservation]: ...

    def peers(self) -> Iterator[PeerObservation]: ...

    def confidence(self, fact: Observation, now: Time) -> float: ...

    def last_observed_times(self) -> NDArray[np.float64]: ...

    def monitorable_cells(self) -> NDArray[np.bool_]: ...


@dataclass(frozen=True, eq=False)
class FactColumns:
    """Facts of one kind, at most one per cell, as parallel arrays.

    Cells are flat indices into the field, ``x * height + y``.

    Attributes:
        cells: The cell of each fact, in the order the facts are to be merged.
        observed_at: When each fact was observed.
        facts: The facts themselves.
    """

    cells: NDArray[np.intp]
    observed_at: NDArray[np.float64]
    facts: NDArray[np.object_]

    def __len__(self) -> int:
        return len(self.cells)


@dataclass(frozen=True, eq=False)
class TerrainColumns(FactColumns):
    """Terrain facts as parallel arrays, with the terrain of each as its ``_TERRAINS`` code.

    Attributes:
        terrain: The terrain code of each fact.
    """

    terrain: NDArray[np.int8]


@dataclass(frozen=True, eq=False)
class KnowledgeSnapshot:
    """What a robot shares of its memory: the payload of a ``MessageKind.KNOWLEDGE`` message.

    A snapshot holds the sender's whole memory. It reuses a part of the sender's previous one
    while that kind of fact has not changed, so a receiver can tell a part it merged before by
    its identity.

    Attributes:
        terrain: The terrain facts.
        infestation: The infestation facts.
        peers: The peer facts.
        crops: The crop facts.
    """

    terrain: TerrainColumns
    infestation: FactColumns
    peers: tuple[PeerObservation, ...]
    crops: tuple[CropObservation, ...] = ()

    def __len__(self) -> int:
        """How many facts the snapshot carries."""
        return len(self.crops) + len(self.terrain) + len(self.infestation) + len(self.peers)

    def facts(self) -> Iterator[Observation]:
        """Every fact, in the order ``AgentMemory.merge_snapshot`` merges them."""
        terrain = cast(list[TerrainObservation], self.terrain.facts.tolist())
        infestation = cast(list[InfestationObservation], self.infestation.facts.tolist())
        return chain(self.crops, terrain, infestation, self.peers)

    def only_infestations(self, with_targets: bool = False) -> KnowledgeSnapshot:
        """This snapshot with its infestation facts alone, or also its peers that have a target.

        The empty parts are the same objects every time, so a receiver skips them as merged.
        """
        peers: tuple[PeerObservation, ...] = ()
        if with_targets:
            peers = tuple(peer for peer in self.peers if peer.target is not None)
        return KnowledgeSnapshot(terrain=_NO_TERRAIN, infestation=self.infestation, peers=peers)


_NO_TERRAIN = TerrainColumns(
    cells=np.empty(0, dtype=np.intp),
    observed_at=np.empty(0, dtype=np.float64),
    facts=np.empty(0, dtype=np.object_),
    terrain=np.empty(0, dtype=np.int8),
)
"""No terrain facts, for a snapshot that shares none."""


class CellFacts:
    """The newest fact per cell of one kind, as flat arrays over the field.

    Besides the fact and when it was observed, each cell remembers when it was first stored,
    so that facts are read back in the order they were learned, and when it was last stored,
    so that memory can tell which cells changed during a decision.

    Attributes:
        observed_at: When each cell's fact was observed, ``-inf`` where there is none.
        facts: Each cell's fact, ``None`` where there is none.
        added_at: The memory version at which each cell got its first fact, ``-1`` if never.
        stored_at: The memory version at which each cell's fact was stored, ``-1`` if never.
        last_stored: The memory version at which the last fact was stored, ``-1`` if never.
    """

    def __init__(self, cells: int) -> None:
        self.observed_at: NDArray[np.float64] = np.full(cells, -np.inf)
        self.facts: NDArray[np.object_] = np.full(cells, None, dtype=np.object_)
        self.added_at: NDArray[np.int64] = np.full(cells, -1, dtype=np.int64)
        self.stored_at: NDArray[np.int64] = np.full(cells, -1, dtype=np.int64)
        self.last_stored: int = -1
        self._learned_order: NDArray[np.intp] | None = None

    def store_one(self, cell: int, fact: ObservationBase, version: int) -> None:
        """Store ``fact`` at ``cell`` as stored at ``version``."""
        if self.added_at[cell] < 0:
            self.added_at[cell] = version
            self._learned_order = None
        self.observed_at[cell] = fact.observed_at
        self.facts[cell] = fact
        self.stored_at[cell] = version
        self.last_stored = version

    def store_many(
        self,
        cells: NDArray[np.intp],
        observed_at: NDArray[np.float64],
        facts: NDArray[np.object_],
        first_version: int,
    ) -> None:
        """Store the facts at ``cells``, in order, as stored from ``first_version`` on."""
        versions = first_version + np.arange(len(cells), dtype=np.int64)
        fresh = self.added_at[cells] < 0
        if fresh.any():
            self.added_at[cells[fresh]] = versions[fresh]
            self._learned_order = None
        self.observed_at[cells] = observed_at
        self.facts[cells] = facts
        self.stored_at[cells] = versions
        self.last_stored = int(versions[-1])

    def known_in_learned_order(self) -> NDArray[np.intp]:
        """Every cell with a fact, in the order the cells got their first fact."""
        if self._learned_order is None:
            known = np.flatnonzero(self.added_at >= 0)
            self._learned_order = known[np.argsort(self.added_at[known])]
        return self._learned_order

    def columns(self, cells: NDArray[np.intp]) -> FactColumns:
        return FactColumns(
            cells=cells, observed_at=self.observed_at[cells], facts=self.facts[cells]
        )


@dataclass(frozen=True)
class _SharedSnapshot:
    """The last snapshot memory handed out, kept to reuse its unchanged parts.

    Attributes:
        version: Memory's version when the snapshot was taken.
        snapshot: The snapshot.
    """

    version: int
    snapshot: KnowledgeSnapshot


class AgentMemory:
    """One robot's facts about the field and its peers, merged from its own sensing and peers.

    Crop, terrain, infestation and named peer facts are stored. Anonymous peer sightings and
    sightings of the robot itself are ignored. A memory that keeps no map stores only peer
    facts, for a robot that steers by beacons and its own sensing alone.

    Attributes:
        keeps_map: Whether crop, terrain and infestation facts are stored.
        version: How many facts memory has stored so far. It grows whenever a merge stores
            something, so two equal versions mean the same memory.
    """

    def __init__(
        self,
        robot_id: RobotId,
        field: FieldGeometry,
        decay: DecayConfig | None = None,
        keeps_map: bool = True,
    ) -> None:
        self.keeps_map: bool = keeps_map
        self._robot_id: RobotId = robot_id
        self._field: FieldGeometry = field
        self._decay: DecayConfig = DecayConfig()
        if decay is not None:
            self._decay = decay
        cells = field.width * field.height
        self._terrain: CellFacts = CellFacts(cells)
        self._terrain_codes: NDArray[np.int8] = np.full(cells, _UNKNOWN_TERRAIN, dtype=np.int8)
        self._infestation: CellFacts = CellFacts(cells)
        self._crop: dict[CellId, CropObservation] = {}
        self._crops_last_stored: int = -1
        self._peers: dict[RobotId, PeerObservation] = {}
        self._peers_last_stored: int = -1
        self._monitorable: NDArray[np.bool_] = np.ones(cells, dtype=np.bool_)
        self._shared: _SharedSnapshot | None = None
        self.version: int = 0

    @property
    def width(self) -> int:
        return self._field.width

    @property
    def height(self) -> int:
        return self._field.height

    def ingest(self, perception: Perception) -> None:
        """Merge the robot's own observations from one decision point; see ``merge``."""
        self.merge(perception.sensed)

    def merge(self, facts: Iterable[Observation]) -> None:
        """Merge facts, keeping per identity the one with the newest ``observed_at``.

        A fact replaces the stored one only if it is strictly newer, so on a tie the robot
        keeps what it already knows. Terrain is static, so a newer terrain fact that repeats
        the stored terrain is not stored either. Without ``keeps_map``, crop, terrain and
        infestation facts are dropped.
        """
        for fact in facts:
            match fact:
                case CropObservation() | TerrainObservation() | InfestationObservation() if (
                    not self.keeps_map
                ):
                    pass
                case CropObservation():
                    self._merge_crops((fact,))
                case TerrainObservation():
                    self._merge_terrain(fact)
                case InfestationObservation():
                    cell = self._flat_index(fact)
                    if cell is not None and fact.observed_at > self._infestation.observed_at[cell]:
                        self._infestation.store_one(cell, fact, self._next_version())
                case PeerObservation():
                    self._merge_peers((fact,))
                case _:
                    pass

    def merge_snapshot(
        self, snapshot: KnowledgeSnapshot, merged_before: KnowledgeSnapshot | None = None
    ) -> None:
        """Merge a peer's snapshot, exactly as ``merge`` would merge ``snapshot.facts()``.

        Args:
            snapshot: What the peer shared.
            merged_before: A snapshot from the same peer that was merged before. A part of
                ``snapshot`` that is the very part merged then is skipped: what memory learned
                since is only newer, so it would store nothing.
        """
        if not self.keeps_map:
            if merged_before is None or snapshot.peers is not merged_before.peers:
                self._merge_peers(snapshot.peers)
            return
        if merged_before is None or snapshot.terrain is not merged_before.terrain:
            self._merge_terrain_columns(snapshot.terrain)
        if merged_before is None or snapshot.infestation is not merged_before.infestation:
            self._merge_infestation_columns(snapshot.infestation)
        if merged_before is None or snapshot.peers is not merged_before.peers:
            self._merge_peers(snapshot.peers)
        if merged_before is None or snapshot.crops is not merged_before.crops:
            self._merge_crops(snapshot.crops)

    def knowledge(self) -> KnowledgeSnapshot:
        """Every fact, to share, in the order memory learned them.

        A peer that merges the snapshot learns new cells in that order. Each part of the
        previous snapshot is reused while nothing of its kind was stored since.
        """
        previous = self._shared
        previous_version = -1
        if previous is not None:
            previous_version = previous.version
        terrain: TerrainColumns
        infestation: FactColumns
        peers: tuple[PeerObservation, ...]
        crops: tuple[CropObservation, ...]
        if previous is not None and self._terrain.last_stored <= previous_version:
            terrain = previous.snapshot.terrain
        else:
            terrain = self._terrain_columns(self._terrain.known_in_learned_order())
        if previous is not None and self._infestation.last_stored <= previous_version:
            infestation = previous.snapshot.infestation
        else:
            infestation = self._infestation.columns(self._infestation.known_in_learned_order())
        if previous is not None and self._peers_last_stored <= previous_version:
            peers = previous.snapshot.peers
        else:
            peers = tuple(self._peers.values())
        if previous is not None and self._crops_last_stored <= previous_version:
            crops = previous.snapshot.crops
        else:
            crops = tuple(self._crop.values())
        snapshot = KnowledgeSnapshot(
            terrain=terrain, infestation=infestation, peers=peers, crops=crops
        )
        self._shared = _SharedSnapshot(version=self.version, snapshot=snapshot)
        return snapshot

    def terrain_stored_after(self, version: int) -> list[CellId]:
        """The cells whose terrain memory learned or changed after ``version``."""
        changed: list[CellId] = []
        for flat in np.flatnonzero(self._terrain.stored_at > version).tolist():
            changed.append(divmod(flat, self._field.height))
        return changed

    def records(self) -> Iterator[Observation]:
        """Every stored fact: crop, terrain, infestation and peers, each in the order learned."""
        return self.knowledge().facts()

    def snapshot(self) -> tuple[Observation, ...]:
        """All stored facts, for ``AgentInspection.belief``."""
        return tuple(self.records())

    def terrain_at(self, cell: CellId) -> Terrain | None:
        """The known terrain of a cell, or ``None`` if unknown.

        Cells beyond the field's bounding box are ``Terrain.OUTSIDE`` without being observed.
        """
        if not self._is_inside(cell):
            return Terrain.OUTSIDE
        code = int(self._terrain_codes[cell[0] * self._field.height + cell[1]])
        if code == _UNKNOWN_TERRAIN:
            return None
        return _TERRAINS[code]

    def infestation_at(self, cell: CellId) -> InfestationObservation | None:
        if not self._is_inside(cell):
            return None
        fact = self._infestation.facts[cell[0] * self._field.height + cell[1]]
        return cast(InfestationObservation | None, fact)

    def infestations(self) -> Iterator[InfestationObservation]:
        """Every infestation fact, including cells observed to be clean (severity 0)."""
        cells = self._infestation.known_in_learned_order()
        return iter(cast(list[InfestationObservation], self._infestation.facts[cells].tolist()))

    def peers(self) -> Iterator[PeerObservation]:
        return iter(self._peers.values())

    def confidence(self, fact: Observation, now: Time) -> float:
        """How far a fact can be trusted at ``now``: its initial confidence, decayed by age."""
        decay_time = self._decay.decay_time_of(fact)
        if decay_time is None:
            return fact.confidence
        return fact.confidence * exp(-(now - fact.observed_at) / decay_time)

    def last_observed_times(self) -> NDArray[np.float64]:
        """When each cell was last observed, by this robot or a peer, indexed ``[x, y]``.

        ``-inf`` means never. Read from the cells' infestation facts, which every observation
        of a cell carries (severity 0 when clean). The array is a read-only view.
        """
        return self._grid_view(self._infestation.observed_at)

    def monitorable_cells(self) -> NDArray[np.bool_]:
        """Whether each cell is worth monitoring, indexed ``[x, y]``: all but known obstacles.

        The array is a read-only view.
        """
        return self._grid_view(self._monitorable)

    def _terrain_columns(self, cells: NDArray[np.intp]) -> TerrainColumns:
        columns = self._terrain.columns(cells)
        return TerrainColumns(
            cells=columns.cells,
            observed_at=columns.observed_at,
            facts=columns.facts,
            terrain=self._terrain_codes[cells],
        )

    def _merge_terrain(self, fact: TerrainObservation) -> None:
        cell = self._flat_index(fact)
        if cell is None:
            return
        code = _TERRAINS.index(fact.terrain)
        if code == self._terrain_codes[cell] or fact.observed_at <= self._terrain.observed_at[cell]:
            return
        self._terrain.store_one(cell, fact, self._next_version())
        self._terrain_codes[cell] = code
        self._monitorable[cell] = code == _FREE

    def _merge_terrain_columns(self, columns: TerrainColumns) -> None:
        cells = columns.cells
        newer = columns.observed_at > self._terrain.observed_at[cells]
        newer &= columns.terrain != self._terrain_codes[cells]
        if not newer.any():
            return
        stored = cells[newer]
        codes = columns.terrain[newer]
        self._terrain.store_many(
            stored, columns.observed_at[newer], columns.facts[newer], self.version + 1
        )
        self.version += len(stored)
        self._terrain_codes[stored] = codes
        self._monitorable[stored] = codes == _FREE

    def _merge_infestation_columns(self, columns: FactColumns) -> None:
        newer = columns.observed_at > self._infestation.observed_at[columns.cells]
        if not newer.any():
            return
        stored = columns.cells[newer]
        self._infestation.store_many(
            stored, columns.observed_at[newer], columns.facts[newer], self.version + 1
        )
        self.version += len(stored)

    def _merge_peers(self, facts: Iterable[PeerObservation]) -> None:
        """Merge peer facts; anonymous sightings and sightings of the robot itself are ignored.

        A peer's own status reports its goal: the target it heads for and the objective. A
        sighting by another robot says where the peer is, and at most which cell it steps into
        next, so a newer sighting keeps the goal of the last status rather than erasing it.
        """
        for fact in facts:
            peer = fact.peer
            if peer is None or peer == self._robot_id:
                continue
            known = self._peers.get(peer)
            if known is None or fact.observed_at > known.observed_at:
                if fact.source != peer and known is not None and known.objective is not None:
                    fact = replace(fact, target=known.target, objective=known.objective)
                self._peers[peer] = fact
                self._peers_last_stored = self._next_version()

    def _merge_crops(self, facts: Iterable[CropObservation]) -> None:
        """Merge crop facts, keeping per cell the one with the newest ``observed_at``."""
        for fact in facts:
            known = self._crop.get(fact.cell)
            if known is None or fact.observed_at > known.observed_at:
                self._crop[fact.cell] = fact
                self._crops_last_stored = self._next_version()

    def _next_version(self) -> int:
        """Count one more stored fact, and return the version it was stored at."""
        self.version += 1
        return self.version

    def _flat_index(self, fact: ObservationBase) -> int | None:
        """The flat index of the fact's cell, or ``None`` (logged) if it lies outside the field."""
        if not self._is_inside(fact.cell):
            logger.warning("Ignoring a fact about %s, outside the field: %r", fact.cell, fact)
            return None
        return fact.cell[0] * self._field.height + fact.cell[1]

    def _grid_view[T: np.generic](self, flat: NDArray[T]) -> NDArray[T]:
        """A read-only view of a flat array over the field, indexed ``[x, y]``."""
        view = flat.reshape(self._field.width, self._field.height)
        view.flags.writeable = False
        return view

    def _is_inside(self, cell: CellId) -> bool:
        x, y = cell
        return 0 <= x < self._field.width and 0 <= y < self._field.height
