"""Field seeders: which cells a grid potential field draws a robot to, and how much.

A ``FieldSeeder`` reads the robot's memory and turns it into the ``Seed``s of one grid potential
field (TREAT, EXPLORE, ...). Seeders are registered by name in the robot's ``FieldProvider``
when the brain is assembled. The brain asks the provider for the winning seeder's field by that
name, and only the provider calls the seeder, whenever its cached field no longer answers.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from math import hypot
from typing import Protocol

import numpy as np
from scipy.spatial import KDTree

from swarm_field.contracts import (
    AgentObjective,
    BodyType,
    CellId,
    InfestationObservation,
    PeerObservation,
    Time,
    TimeUnit,
)
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


CLAIM_MAX_AGE: Time = 2 * TimeUnit.MINUTE
"""How long a peer's report that it is heading for a pest keeps the pest claimed.

A killer reports its target with every status message, so a recent report means it is still on
its way; an older one may come from a killer that has since left radio range or failed.
"""


class TreatSeeder:
    """Seeds the TREAT field: draws the robot to remembered infestations.

    Each infested cell attracts by its severity times the memory's confidence in it, and
    clustered infestations attract more than isolated ones. With ``yield_to_closer``, only the
    pests left to this robot attract (``pests_left_to_robot``): a pest a closer killer freshly
    claimed is left out, so two killers do not drive to the same pest. Rebuilt once a minute,
    and whenever the infested cells or the pests left change.

    Attributes:
        value_amplifier: The attraction of a cell of severity 1 at full confidence, in cells.
        sigma: The width, in cells, of the Gaussian by which nearby infestations reinforce
            each other.
        cutoff_sigmas: Infestations further apart than this many widths do not reinforce each
            other. At the default of three the weight is ``exp(-9)``, about 0.0001, so what is
            left out is negligible.
        window_blocks: How many blocks around the robot's own the field is searched cell by
            cell, as in ``FieldSeeder``.
        yield_to_closer: Whether to leave out the pests a closer killer has claimed.
        claim_fresh_for: How old a peer's report of its target may be to count as a claim.
    """

    name: str = "treat"

    def __init__(
        self,
        value_amplifier: float = 1.0,
        sigma: float = 1.0,
        cutoff_sigmas: float = 3.0,
        window_blocks: int = 2,
        yield_to_closer: bool = False,
        claim_fresh_for: Time = CLAIM_MAX_AGE,
    ) -> None:
        if value_amplifier < 0.0:
            raise ValueError("value_amplifier must be non-negative")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if cutoff_sigmas <= 0.0:
            raise ValueError("cutoff_sigmas must be positive")
        self.window_blocks: int = window_blocks
        self.yield_to_closer: bool = yield_to_closer
        self.claim_fresh_for: Time = claim_fresh_for
        self._value_amplifier: float = value_amplifier
        self._sigma: float = sigma
        self._cutoff_sigmas: float = cutoff_sigmas

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        facts = tuple(
            (fact.cell, fact.observed_at, round(fact.severity, 3))
            for fact in memory.infestations()
            if fact.severity > 0.0
        )
        left: set[CellId] = set()
        for pest in self._pests(memory, robot, now):
            left.add(pest.cell)
        return now // TimeUnit.MINUTE, facts, frozenset(left)

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        seeds: list[Seed] = []
        for pest in self._pests(memory, robot, now):
            attractiveness = pest.severity * memory.confidence(pest, now)
            if attractiveness > 0.0:
                seeds.append(Seed(pest.cell, -self._value_amplifier * attractiveness))

        if len(seeds) == 0:
            return ()
        updated_seeds = self.reinforce_clustered_seeds(seeds)
        return tuple(updated_seeds)

    def _pests(self, memory: MemoryView, robot: CellId, now: Time) -> list[InfestationObservation]:
        """The pests this field draws to: those left to the robot, or every known one."""
        if self.yield_to_closer:
            return pests_left_to_robot(memory, robot, now, self.claim_fresh_for)
        return infested(memory)

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

    Each location of ``unsearched_peer_locations`` attracts by the memory's confidence in the
    peer fact behind it, so the most recent contact attracts most. The rally point, if any,
    attracts least and is never dropped, so it is where a robot ends up, and waits, once every
    peer location turned out empty. Rebuilt once a minute, and whenever those locations change.

    Attributes:
        value_amplifier: The attraction of a peer location at full confidence, in cells.
        rally_point: The cell every robot falls back to, or ``None`` for no fallback.
    """

    name: str = "regroup"
    window_blocks: int = 2

    def __init__(self, value_amplifier: float = 10.0, rally_point: CellId | None = None) -> None:
        self._value_amplifier: float = value_amplifier
        self.rally_point: CellId | None = rally_point

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        cells: set[CellId] = set()
        for cell, _peer in unsearched_peer_locations(memory):
            cells.add(cell)
        return (now // TimeUnit.MINUTE, frozenset(cells))

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        # One seed per cell: the most attractive of the peers pointing at it.
        values: dict[CellId, float] = {}
        for cell, peer in unsearched_peer_locations(memory):
            if cell == self.rally_point:
                continue
            value = -self._value_amplifier * memory.confidence(peer, now)
            values[cell] = min(value, values.get(cell, 0.0))

        seeds: list[Seed] = []
        for cell, value in values.items():
            seeds.append(Seed(cell, value))
        if self.rally_point is not None:
            seeds.append(Seed(self.rally_point, 0.0))
        return tuple(seeds)


def unsearched_peer_locations(memory: MemoryView) -> list[tuple[CellId, PeerObservation]]:
    """Every remembered peer's last known cell and announced target, unless searched since.

    A location counts as searched once its cell was observed after the peer was reported there.
    A peer still in that cell would have been sensed or heard at the same time, which refreshes
    its fact, so a newer observation of the cell without a newer report means the peer has left.
    """
    last_seen = memory.last_observed_times()
    locations: list[tuple[CellId, PeerObservation]] = []
    for peer in memory.peers():
        cells = [peer.cell]
        if peer.target is not None:
            cells.append(peer.target)
        for cell in cells:
            if last_seen[cell] > peer.observed_at:
                continue
            locations.append((cell, peer))
    return locations


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


def pests_left_to_robot(
    memory: MemoryView, robot: CellId, now: Time, claim_fresh_for: Time = CLAIM_MAX_AGE
) -> list[InfestationObservation]:
    """The known pests that no closer killer is heading for, to treat them.

    A peer that recently reported a pest as its TREAT target claims it when it is closer to the
    pest than ``robot``, or as close with a lower robot id, so of two killers heading for the
    same pest exactly one keeps it and the other looks elsewhere; killers that travel together
    are as close as each other, so without the tie-break none would give way. Both the TREAT
    field and ``TreatInfestation``'s bid read the pests through this, so the bid and the field
    agree on which pests are left.
    """
    claimed: set[CellId] = set()
    for peer in memory.peers():
        if peer.target is None or peer.objective is not AgentObjective.TREAT:
            continue
        if now - peer.observed_at > claim_fresh_for:
            continue
        peer_distance = hypot(peer.target[0] - peer.cell[0], peer.target[1] - peer.cell[1])
        own_distance = hypot(peer.target[0] - robot[0], peer.target[1] - robot[1])
        closer = peer_distance < own_distance
        as_close_with_lower_id = (
            peer_distance == own_distance and peer.peer is not None and peer.peer < memory.robot_id
        )
        if closer or as_close_with_lower_id:
            claimed.add(peer.target)

    left: list[InfestationObservation] = []
    for pest in infested(memory):
        if pest.cell not in claimed:
            left.append(pest)
    return left


def infested(memory: MemoryView) -> list[InfestationObservation]:
    """Every pest memory knows of: the infestation facts with a severity above 0."""
    pests: list[InfestationObservation] = []
    for pest in memory.infestations():
        if pest.severity > 0.0:
            pests.append(pest)
    return pests
