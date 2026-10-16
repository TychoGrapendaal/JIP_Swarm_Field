"""Field seeders: which cells a grid potential field draws a robot to, and how much.

A ``FieldSeeder`` reads the robot's memory and turns it into the ``Seed``s of one grid potential
field (TREAT, EXPLORE, ...). Seeders are registered by name in the robot's ``FieldProvider``
when the brain is assembled. Behaviours ask the provider for a seeder's field by that name, and
only the provider calls the seeder, whenever its cached field no longer answers.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial import KDTree

from swarm_field.contracts import CellId, Time, TimeUnit
from swarm_field.swarm.field import GridPotentialField, Seed
from swarm_field.swarm.memory import MemoryView


class FieldSeeder(Protocol):
    """Specifies one grid potential field: its targets, read from memory, and when they change.

    Attributes:
        name: Identifies the seeder in the field provider, and so the field's cache slot.
    """

    name: str

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        """A value that changes whenever the seeds would change.

        The field is rebuilt only when this key, the terrain or the robot's block changes, so
        the key covers what the seeds depend on and nothing else; terrain is accounted for by
        the field provider. Seeds that drift continuously, such as values weighted by a decaying
        confidence, need a coarse key, e.g. a memory version plus a time bucket.
        """
        ...

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        """The field's targets, each with its start value ``-beta * value`` (lower attracts more).

        The seeds may depend on the robot's position only through its block of
        ``Quadtree.max_leaf_side`` cells, since a field is reused for as long as the robot
        stays in that block.
        """
        ...


@dataclass(frozen=True)
class SeededField:
    """A seeder's grid potential field, with the seeds it was built from.

    Attributes:
        seeds: The targets the field was built from, never empty.
        field: The field itself.
    """

    seeds: tuple[Seed, ...]
    field: GridPotentialField


class TreatSeeder:
    """Seeds the TREAT field: draws the robot to remembered infestations.

    Each infested cell attracts by its severity times the memory's confidence in it, and
    clustered infestations attract more than isolated ones. A cell that a peer is already
    heading for attracts less. Rebuilt once a minute, and whenever the infested cells or the
    peers' targets change.

    Attributes:
        reduction_rate: What is left of a cell's attraction when a peer is heading for it, 0..1.
        value_amplifier: The attraction of a cell of severity 1 at full confidence, in cells.
        sigma: The width, in cells, of the Gaussian by which nearby infestations reinforce
            each other.
        cutoff_sigmas: Infestations further apart than this many widths do not reinforce each
            other. At the default of three the weight is ``exp(-9)``, about 0.0001, so what is
            left out is negligible.
    """

    name: str = "treat"

    def __init__(
        self,
        reduction_rate: float = 0.2,
        value_amplifier: float = 1.0,
        sigma: float = 1.0,
        cutoff_sigmas: float = 3.0,
    ) -> None:
        self._reduction_rate: float = reduction_rate
        self._value_amplifier: float = value_amplifier
        self._sigma: float = sigma
        self._cutoff_sigmas: float = cutoff_sigmas

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        infested: set[CellId] = set()
        for pest in memory.infestations():
            if pest.severity > 0:
                infested.add(pest.cell)
        # If this is too expensive, we can remove one of the sets
        return (now // TimeUnit.MINUTE, frozenset(infested), _cells_targeted_by_peers(memory))

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        seeds: list[Seed] = []
        targeted = _cells_targeted_by_peers(memory)
        for pest in memory.infestations():
            if pest.severity > 0:
                value = pest.severity * memory.confidence(pest, now)
                if pest.cell in targeted:
                    value *= self._reduction_rate
                seeds.append(Seed(pest.cell, -self._value_amplifier * value))

        if len(seeds) == 0:
            return ()
        updated_seeds = self.reinforce_clustered_seeds(seeds)
        return tuple(updated_seeds)

    def reinforce_clustered_seeds(self, seeds: list[Seed]) -> list[Seed]:
        """Update each value by the values of its neighbours, so clusters attract more.

        Every seed gains the values of the other seeds within ``cutoff_sigmas`` widths of it,
        weighted by a Gaussian of their Euclidean distance in cells. Only those nearby pairs
        are stored, in a sparse matrix, so the cost grows with the number of neighbours rather
        than with the number of seeds squared.
        """
        coords = np.array([s.cell for s in seeds])
        values = np.array([s.value for s in seeds])

        tree = KDTree(coords)
        cutoff = self._cutoff_sigmas * self._sigma
        weights = tree.sparse_distance_matrix(tree, cutoff, output_type="coo_matrix").tocsr()
        weights.data = np.exp(-((weights.data / self._sigma) ** 2))
        weights.setdiag(0)
        new_values = values + weights.dot(values)

        updated_seeds = []
        for s, new_value in zip(seeds, new_values, strict=True):
            updated_seeds.append(Seed(s.cell, new_value))
        return updated_seeds


class RegroupSeeder:
    """Seeds the REGROUP field: draws the robot to where its peers were last known to be.

    Each remembered peer attracts at its last known cell and at the cell it said it was heading
    for, by the memory's confidence in that fact, so the most recent contact attracts most. A
    cell the robot searched without hearing anyone no longer attracts. The rally point, if
    any, attracts least and is never dropped, so it is where a robot ends up, and waits, once
    every peer location turned out empty. Rebuilt once a minute, and whenever the peers' cells
    or targets, or the searched cells, change.

    Attributes:
        value_amplifier: The attraction of a peer location at full confidence, in cells.
        rally_point: The cell every robot falls back to, or ``None`` for no fallback.
    """

    name: str = "regroup"

    def __init__(self, value_amplifier: float = 10.0, rally_point: CellId | None = None) -> None:
        self._value_amplifier: float = value_amplifier
        self.rally_point: CellId | None = rally_point
        self._searched: set[CellId] = set()

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        locations: set[tuple[CellId, CellId | None]] = set()
        for peer in memory.peers():
            locations.add((peer.cell, peer.target))
        return (now // TimeUnit.MINUTE, frozenset(locations), frozenset(self._searched))

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        # One seed per cell: the most attractive of the peers pointing at it.
        values: dict[CellId, float] = {}
        for peer in memory.peers():
            value = -self._value_amplifier * memory.confidence(peer, now)
            cells = [peer.cell]
            if peer.target is not None:
                cells.append(peer.target)
            for cell in cells:
                if cell not in self._searched and cell != self.rally_point:
                    values[cell] = min(value, values.get(cell, 0.0))

        seeds: list[Seed] = []
        for cell, value in values.items():
            seeds.append(Seed(cell, value))
        if self.rally_point is not None:
            seeds.append(Seed(self.rally_point, 0.0))
        return tuple(seeds)

    def mark_searched(self, cell: CellId) -> None:
        """Stop being drawn to ``cell``: the robot went there and heard no one."""
        self._searched.add(cell)

    def forget_searched(self) -> None:
        """Let every peer location attract again, once the robot is back in contact."""
        self._searched.clear()


def _cells_targeted_by_peers(memory: MemoryView) -> frozenset[CellId]:
    """The cells the remembered peers say they are heading for."""
    targeted: set[CellId] = set()
    for peer in memory.peers():
        if peer.target is not None:
            targeted.add(peer.target)
    return frozenset(targeted)
