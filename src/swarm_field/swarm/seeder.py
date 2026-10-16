"""Field seeders: which cells a grid potential field draws a robot to, and how much.

A ``FieldSeeder`` reads the robot's memory and turns it into ``Seed`` objects. Seeders are
registered by name when a robot's ``FieldProvider`` is assembled. The provider caches each
seeder's field and rebuilds it when the key, terrain, or robot's search window changes.
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
    """Contract for producing a potential field from a robot's local memory."""

    name: str
    window_blocks: int

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        """Return a hashable key changed whenever the seeder's targets or values change."""
        ...

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        """Return target cells and their initial potential values (lower is more attractive)."""
        ...


@dataclass(frozen=True)
class SeededField:
    """A built potential field and the non-empty set of seeds that produced it."""

    seeds: tuple[Seed, ...]
    field: GridPotentialField


class TreatSeeder:
    """Attract killers to remembered infestations while reducing duplicate assignments.

    Nearby infestation seeds reinforce one another; cells a peer is already heading to for
    treatment have less attraction. The field key is refreshed at least once per simulated
    minute because observation confidence decays over time.
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
        targeted = _cells_targeted_by_peers(memory)
        seeds: list[Seed] = []
        for pest in memory.infestations():
            if pest.severity <= 0.0:
                continue
            attractiveness = pest.severity * memory.confidence(pest, now)
            if pest.cell in targeted:
                attractiveness *= self._reduction_rate
            if attractiveness > 0.0:
                seeds.append(Seed(pest.cell, -self._value_amplifier * attractiveness))

        if not seeds:
            return ()
        return tuple(self.reinforce_clustered_seeds(seeds))

    def reinforce_clustered_seeds(self, seeds: list[Seed]) -> list[Seed]:
        """Increase the attraction of clustered seeds using a sparse Gaussian weighting."""
        if len(seeds) < 2:
            return list(seeds)
        coords = np.asarray([seed.cell for seed in seeds], dtype=float)
        values = np.asarray([seed.value for seed in seeds], dtype=float)
        tree = KDTree(coords)
        cutoff = self._cutoff_sigmas * self._sigma
        weights = tree.sparse_distance_matrix(tree, cutoff, output_type="coo_matrix").tocsr()
        weights.data = np.exp(-((weights.data / self._sigma) ** 2))
        weights.setdiag(0)
        new_values = values + weights.dot(values)
        return [
            Seed(seed.cell, float(value)) for seed, value in zip(seeds, new_values, strict=True)
        ]


class BatteryRequestSeeder:
    """Attract battery-support robots to recent low-battery peer sightings."""

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
    """Seed fixed charging docks so robots can return when their energy is low."""

    name: str = "dock"
    window_blocks: int = 2

    def __init__(self, docks: tuple[CellId, ...] = ()) -> None:
        self._docks: tuple[CellId, ...] = tuple(docks)

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        return self._docks

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        return tuple(Seed(dock, 0.0) for dock in self._docks)


def _cells_targeted_by_peers(memory: MemoryView) -> frozenset[CellId]:
    """Return cells peers report targeting for the TREAT objective."""
    return frozenset(
        peer.target
        for peer in memory.peers()
        if peer.target is not None and peer.objective is AgentObjective.TREAT
    )
