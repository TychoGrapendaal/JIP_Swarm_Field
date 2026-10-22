"""What a robot of the hop method remembers: beacons, and nothing about the map.

``BeaconMemory`` keeps the newest beacon heard from each sender per kind, the beacons the robot
placed itself to push it away from a cell, and which senders are killers. Every hop rule reads
it through ``BeaconView``, which offers no terrain, pest cells or quadtree, so a rule cannot
come to depend on a shared map.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from swarm_field.contracts import Beacon, BeaconKind, BodyType, CellId, RobotId, Time


@dataclass(frozen=True)
class HeardBeacon:
    """A beacon as the robot heard it.

    Attributes:
        sender: Who sent it: a robot, or a mock pest source.
        sent_at: When it was sent.
        beacon: The hop value and where it holds.
    """

    sender: RobotId
    sent_at: Time
    beacon: Beacon


@dataclass(frozen=True)
class PlacedBeacon:
    """A beacon the robot placed itself, which pushes it away from a cell. It is never sent.

    Attributes:
        cell: The cell it pushes away from.
        strength: Its push at its own cell when placed.
        placed_at: When it was placed.
        decay: The decay time of its push with its age.
    """

    cell: CellId
    strength: float
    placed_at: Time
    decay: Time


class BeaconView(Protocol):
    """The read-only queries on a ``BeaconMemory`` that the hop rules are typed against."""

    def beacons(self, kind: BeaconKind) -> Iterator[HeardBeacon]: ...

    def placed_beacons(self) -> Iterator[PlacedBeacon]: ...

    def placed_beacon_at(self, cell: CellId) -> PlacedBeacon | None: ...

    def killers(self) -> frozenset[RobotId]: ...


class BeaconMemory:
    """One robot's beacons: heard from its peers, and placed by itself."""

    def __init__(self) -> None:
        self._heard: dict[tuple[RobotId, BeaconKind], HeardBeacon] = {}
        self._placed: dict[CellId, PlacedBeacon] = {}
        self._killers: set[RobotId] = set()

    def hear(self, sender: RobotId, sent_at: Time, beacons: Iterable[Beacon]) -> None:
        """Keep the newest beacon per sender and kind."""
        for beacon in beacons:
            key = (sender, beacon.kind)
            stored = self._heard.get(key)
            if stored is None or sent_at > stored.sent_at:
                self._heard[key] = HeardBeacon(sender, sent_at, beacon)

    def place(self, cell: CellId, now: Time, strength: float, decay: Time) -> None:
        """Place a beacon on ``cell`` at ``now``: one per cell, the newest kept."""
        self._placed[cell] = PlacedBeacon(cell, strength, now, decay)

    def note_body(self, peer: RobotId, body: BodyType) -> None:
        """Remember a peer's body, from its status, to tell killers' beacons apart."""
        if body is BodyType.KILLER:
            self._killers.add(peer)

    def beacons(self, kind: BeaconKind) -> Iterator[HeardBeacon]:
        """The newest beacon heard from each sender of ``kind``, however old."""
        for heard in self._heard.values():
            if heard.beacon.kind is kind:
                yield heard

    def placed_beacons(self) -> Iterator[PlacedBeacon]:
        return iter(self._placed.values())

    def placed_beacon_at(self, cell: CellId) -> PlacedBeacon | None:
        return self._placed.get(cell)

    def killers(self) -> frozenset[RobotId]:
        """The peers the robot heard report themselves as killers."""
        return frozenset(self._killers)
