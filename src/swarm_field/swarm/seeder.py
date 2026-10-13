"""Field seeders: which cells a grid potential field draws a robot to, and how much.

A ``FieldSeeder`` reads the robot's memory and turns it into the ``Seed``s of one grid potential
field (TREAT, EXPLORE, ...). Seeders are registered by name in the robot's ``FieldProvider``
when the brain is assembled. Behaviours ask the provider for a seeder's field by that name, and
only the provider calls the seeder, whenever its cached field no longer answers.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol

from swarm_field.contracts import BodyType, CellId, Time, TimeUnit
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
    """Attract the killer toward known infestations, weighted by severity and confidence."""

    name: str = "treat"

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        facts = tuple(
            (fact.cell, fact.observed_at, round(fact.severity, 3))
            for fact in memory.infestations()
            if fact.severity > 0.0
        )
        # Confidence changes with age, so refresh the cached field once per simulated minute.
        return now // TimeUnit.MINUTE, facts

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        result = []
        for fact in memory.infestations():
            confidence = memory.confidence(fact, now)
            attractiveness = fact.severity * confidence
            if attractiveness > 0.0:
                result.append(Seed(fact.cell, -attractiveness))
        return tuple(result)


class ExploreSeeder:
    """Seed every cell whose terrain the robot has not observed yet."""

    name: str = "explore"

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        return tuple(
            (x, y)
            for x in range(memory.width)
            for y in range(memory.height)
            if memory.terrain_at((x, y)) is None
        )

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        return tuple(
            Seed((x, y), 0.0)
            for x in range(memory.width)
            for y in range(memory.height)
            if memory.terrain_at((x, y)) is None
        )


class BatteryRequestSeeder:
    """Seed recent low-battery peer sightings for the mobile battery-support robot."""

    name: str = "battery-support"
    low_battery_threshold: float = 0.25
    observation_ttl: Time = 60 * TimeUnit.SECOND

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        return tuple(
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
    """Seed configured dock cells so low-energy robots can return for recharge."""

    name: str = "dock"

    def __init__(self, docks: tuple[CellId, ...] = ()) -> None:
        self._docks = tuple(docks)

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        return self._docks

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        return tuple(Seed(dock, 0.0) for dock in self._docks)
