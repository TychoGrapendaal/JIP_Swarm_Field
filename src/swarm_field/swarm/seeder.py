"""Field seeders: which cells a grid potential field draws a robot to, and how much.

A ``FieldSeeder`` reads the robot's memory and turns it into the ``Seed``s of one grid potential
field (TREAT, EXPLORE, ...). Seeders are registered by name in the robot's ``FieldProvider``
when the brain is assembled. The brain asks the provider for the winning seeder's field by that
name, and only the provider calls the seeder, whenever its cached field no longer answers.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial import KDTree

from swarm_field.contracts import AgentObjective, BodyType, CellId, Time, TimeUnit
from swarm_field.swarm.explore import ExploreSeeder as ExploreSeeder
from swarm_field.swarm.field import GridPotentialField, Seed
from swarm_field.swarm.memory import MemoryView


class FieldSeeder(Protocol):
    """Specifies one grid potential field: its targets, read from memory, and when they change.

    Attributes:
        name: Identifies the seeder in the field provider, and so the field's cache slot.
        window_blocks: How many blocks of ``Quadtree.max_leaf_side`` cells around the robot's
            block the field is searched cell by cell; beyond them, leaf by leaf. A smaller
            window builds faster but measures far routes less exactly.
    """

    name: str
    window_blocks: int

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        """A value that changes whenever the seeds would change.

        The field is rebuilt only when this key, the terrain or the robot's block changes, so
        the key covers what the seeds depend on and nothing else; terrain is accounted for by
        the field provider. Seeds that drift continuously, such as values weighted by a decaying
        confidence, need a coarse key, e.g. a memory version plus a time bucket. Like the seeds,
        the key may depend on the robot's position only through its block.
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
    heading for, to treat it, attracts less. Rebuilt once a minute, and whenever the infested
    cells or the peers' targets change.

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
    window_blocks: int = 2

    def __init__(
        self,
        reduction_rate: float = 0.2,
        value_amplifier: float = 1.0,
        sigma: float = 1.0,
        cutoff_sigmas: float = 3.0,
    ) -> None:
        if not 0.0 <= reduction_rate <= 1.0:
            raise ValueError("reduction_rate must be between 0 and 1")
        if value_amplifier < 0.0:
            raise ValueError("value_amplifier must be non-negative")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if cutoff_sigmas <= 0.0:
            raise ValueError("cutoff_sigmas must be positive")
        self._reduction_rate: float = reduction_rate
        self._value_amplifier: float = value_amplifier
        self._sigma: float = sigma
        self._cutoff_sigmas: float = cutoff_sigmas

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        facts = tuple(
            (fact.cell, fact.observed_at, round(fact.severity, 3))
            for fact in memory.infestations()
            if fact.severity > 0.0
        )
        return now // TimeUnit.MINUTE, facts, _cells_targeted_by_peers(memory)

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        seeds: list[Seed] = []
        targeted = _cells_targeted_by_peers(memory)
        for pest in memory.infestations():
            if pest.severity <= 0.0:
                continue
            attractiveness = pest.severity * memory.confidence(pest, now)
            if pest.cell in targeted:
                attractiveness *= self._reduction_rate
            if attractiveness > 0.0:
                seeds.append(Seed(pest.cell, -self._value_amplifier * attractiveness))

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
    window_blocks: int = 2

    def __init__(self, value_amplifier: float = 10.0, rally_point: CellId | None = None) -> None:
        self._value_amplifier: float = value_amplifier
        self.rally_point: CellId | None = rally_point
        self._searched: set[CellId] = set()

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
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

    def attracts_to(self, memory: MemoryView, cell: CellId) -> bool:
        """Whether ``cell`` is a peer location the robot has not searched yet."""
        return any(peer.cell == cell or peer.target == cell for peer in memory.peers())

    def mark_searched(self, cell: CellId) -> None:
        """Stop being drawn to ``cell``: the robot went there and heard no one."""
        self._searched.add(cell)

    def forget_searched(self) -> None:
        """Let every peer location attract again, once the robot is back in contact."""
        self._searched.clear()


class BatteryRequestSeeder:
    """Seeds a field toward fresh reports of peers that need battery support.

    Only named SCOUT and KILLER peers whose reported battery is at or below the threshold are
    eligible. The reports expire after ``observation_ttl`` and need sufficient memory confidence.
    """

    name: str = "battery-support"
    window_blocks: int = 2
    low_battery_threshold: float = 0.25
    observation_ttl: Time = 60 * TimeUnit.SECOND

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        eligible = tuple(
            sorted(
                (
                    fact.peer,
                    fact.cell,
                    round(fact.battery, 2),
                    fact.observed_at,
                )
                for fact in memory.peers()
                if fact.peer is not None
                and fact.battery is not None
                and fact.body in (BodyType.SCOUT, BodyType.KILLER)
                and fact.battery <= self.low_battery_threshold
                and 0 <= now - fact.observed_at <= self.observation_ttl
                and memory.confidence(fact, now) >= 0.5
            )
        )
        return now // TimeUnit.MINUTE, eligible

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        return tuple(
            Seed(fact.cell, -1.0 + fact.battery)
            for fact in memory.peers()
            if fact.peer is not None
            and fact.battery is not None
            and fact.body in (BodyType.SCOUT, BodyType.KILLER)
            and fact.battery <= self.low_battery_threshold
            and 0 <= now - fact.observed_at <= self.observation_ttl
            and memory.confidence(fact, now) >= 0.5
        )


class DockSeeder:
    """Seed fixed charging docks so robots can return when their energy is low.

    Dock locations are scenario configuration rather than learned facts, so the field key is the
    tuple of configured dock cells and the seeds remain fixed throughout a run.
    """

    name: str = "dock"
    window_blocks: int = 2

    def __init__(self, docks: tuple[CellId, ...] = ()) -> None:
        self._docks: tuple[CellId, ...] = tuple(docks)

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        return self._docks

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        return tuple(Seed(dock, 0.0) for dock in self._docks)


def _cells_targeted_by_peers(memory: MemoryView) -> frozenset[CellId]:
    """The cells the remembered peers say they are heading for, to treat them."""
    targeted: set[CellId] = set()
    for peer in memory.peers():
        if peer.target is not None and peer.objective is AgentObjective.TREAT:
            targeted.add(peer.target)
    return frozenset(targeted)
