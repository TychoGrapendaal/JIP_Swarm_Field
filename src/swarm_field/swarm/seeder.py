"""Field seeders: which cells a grid potential field draws a robot to, and how much.

A ``FieldSeeder`` reads the robot's memory and turns it into the ``Seed``s of one grid potential
field (TREAT, EXPLORE, ...). Seeders are registered by name in the robot's ``Navigator`` when
the brain is assembled. Behaviours ask the navigator for a seeder's field by that name, and only
the navigator calls the seeder, whenever its cached field no longer answers.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial.distance import cdist

from swarm_field.contracts import CellId, Time, TimeUnit
from swarm_field.swarm.field import GridPotentialField, Seed
from swarm_field.swarm.memory import MemoryView


class FieldSeeder(Protocol):
    """Specifies one grid potential field: its targets, read from memory, and when they change.

    Attributes:
        name: Identifies the seeder in the navigator, and so the field's cache slot.
    """

    name: str

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        """A value that changes whenever the seeds would change.

        The field is rebuilt only when this key, the terrain or the robot's block changes, so
        the key covers what the seeds depend on and nothing else; terrain is accounted for by
        the navigator. Seeds that drift continuously, such as values weighted by a decaying
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
    clustered infestations attract more than isolated ones. Rebuilt once a minute.
    """

    name: str = "treat"

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        return now // (TimeUnit.MINUTE)

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        seeds: list[Seed] = []
        for pest in memory.infestations():
            if pest.severity > 0:
                value = pest.severity * memory.confidence(pest, now)
                seeds.append(Seed(pest.cell, -10 * value))

        if len(seeds) == 0:
            return ()
        updated_seeds = self.reinforce_clustered_seeds(seeds)
        return tuple(updated_seeds)

    def reinforce_clustered_seeds(self, seeds: list[Seed], sigma: float = 1.0) -> list[Seed]:
        """Update each value by the values of its neighbours, so clusters attract more.

        Every seed gains the other seeds' values, weighted by a Gaussian of their Euclidean
        distance in cells; ``sigma`` is the width of that Gaussian.
        """
        coords = np.array([s.cell for s in seeds])
        values = np.array([s.value for s in seeds])

        dist_matrix = cdist(coords, coords, metric="euclidean")
        weights = np.exp(-((dist_matrix / sigma) ** 2))
        np.fill_diagonal(weights, 0)
        new_values = values + weights.dot(values)

        updated_seeds = []
        for s, new_value in zip(seeds, new_values, strict=True):
            updated_seeds.append(Seed(s.cell, new_value))
        return updated_seeds