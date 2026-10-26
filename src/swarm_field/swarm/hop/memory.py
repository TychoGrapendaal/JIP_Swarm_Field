"""What a robot of the hop method remembers: beacons, and nothing about the map.

``BeaconMemory`` keeps the newest beacon heard from each sender per kind, the beacons the robot
placed itself to push it away from a cell, the pests the robot saw itself, and which senders are
killers. It forgets what no longer counts, and it keeps at most ``MEMORY_CAP`` entries of each
sort, so it stays small however large the swarm and the field. Every hop rule reads it through
``BeaconView``, which offers no terrain, other robots' pest cells or quadtree, so a rule cannot
come to depend on a shared map.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

from swarm_field.contracts import Beacon, BeaconKind, BodyType, CellId, RobotId, Time

_FADED_DECAYS = 5.0
"""After this many decay times a placed beacon pushes with under 1 % of its strength, and is
forgotten."""

MEMORY_CAP = 100
"""The most entries a ``BeaconMemory`` keeps of each sort: heard beacons of one kind, placed
beacons, pest sightings and killers. Lifetimes alone would let a sort grow with the swarm or the
field, so beyond the cap the oldest entry is forgotten first."""


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

    def latest_sighting(self, since: Time) -> CellId | None: ...


class BeaconMemory:
    """One robot's beacons, heard from peers and placed by itself, and its own pest sightings.

    Each sort holds at most ``cap`` entries, heard beacons ``cap`` per kind; storing one more
    forgets the oldest of its sort.
    """

    def __init__(self, cap: int = MEMORY_CAP) -> None:
        if cap < 1:
            raise ValueError("cap must be at least 1")
        self._cap: int = cap
        self._heard: dict[tuple[RobotId, BeaconKind], HeardBeacon] = {}
        self._placed: dict[CellId, PlacedBeacon] = {}
        self._killers: dict[RobotId, Time] = {}
        self._sightings: dict[CellId, Time] = {}

    def hear(self, sender: RobotId, sent_at: Time, beacons: Iterable[Beacon]) -> None:
        """Keep the newest beacon per sender and kind, and at most ``cap`` of each kind."""
        for beacon in beacons:
            key = (sender, beacon.kind)
            stored = self._heard.get(key)
            if stored is None or sent_at > stored.sent_at:
                self._heard[key] = HeardBeacon(sender, sent_at, beacon)
                if stored is None:
                    self._limit_heard(beacon.kind)

    def place(self, cell: CellId, now: Time, strength: float, decay: Time) -> None:
        """Place a beacon on ``cell`` at ``now``: one per cell, the newest kept."""
        self._placed[cell] = PlacedBeacon(cell, strength, now, decay)
        _forget_oldest(self._placed, _placed_at, self._cap)

    def note_body(self, peer: RobotId, body: BodyType, at: Time) -> None:
        """Remember that a peer reported itself a killer at ``at``, to tell its beacons apart."""
        if body is not BodyType.KILLER:
            return
        self._killers[peer] = max(at, self._killers.get(peer, at))
        _forget_oldest(self._killers, _itself, self._cap)

    def see(self, cell: CellId, at: Time, infested: bool) -> None:
        """Note what the robot itself sensed on ``cell``: a pest, or that the pest there is gone."""
        if infested:
            self._sightings[cell] = at
            _forget_oldest(self._sightings, _itself, self._cap)
        else:
            self._sightings.pop(cell, None)

    def forget(
        self, heard_before: Mapping[BeaconKind, Time | None], seen_before: Time, now: Time
    ) -> None:
        """Drop what no longer counts at ``now``.

        Args:
            heard_before: Per kind, heard beacons sent before this are dropped; with ``None``
                none of the kind is, and a kind not in it is dropped altogether. Killers that
                last reported before the TREAT cutoff are dropped too: a killer's status comes
                with its beacon.
            seen_before: Pest sightings older than this are dropped.
            now: Placed beacons that have faded by now are dropped.
        """
        kept: dict[tuple[RobotId, BeaconKind], HeardBeacon] = {}
        for key, heard in self._heard.items():
            kind = key[1]
            if kind not in heard_before:
                continue
            cutoff = heard_before[kind]
            if cutoff is None or heard.sent_at >= cutoff:
                kept[key] = heard
        self._heard = kept
        killers_before = heard_before.get(BeaconKind.TREAT, now)
        if killers_before is not None:
            self._killers = {peer: at for peer, at in self._killers.items() if at >= killers_before}
        self._sightings = {cell: at for cell, at in self._sightings.items() if at >= seen_before}
        self._placed = {
            cell: placed
            for cell, placed in self._placed.items()
            if now - placed.placed_at < _FADED_DECAYS * placed.decay
        }

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
        """The peers the robot heard report themselves as killers, and has not forgotten."""
        return frozenset(self._killers)

    def latest_sighting(self, since: Time) -> CellId | None:
        """The cell of the pest the robot saw itself last, at ``since`` or later, if any."""
        latest: CellId | None = None
        latest_at = since
        for cell, at in self._sightings.items():
            if at >= latest_at:
                latest = cell
                latest_at = at
        return latest

    def _limit_heard(self, kind: BeaconKind) -> None:
        """Forget the oldest heard beacon of ``kind`` while more than ``cap`` are kept."""
        while True:
            oldest: tuple[RobotId, BeaconKind] | None = None
            count = 0
            for key, heard in self._heard.items():
                if key[1] is not kind:
                    continue
                count += 1
                if oldest is None or heard.sent_at < self._heard[oldest].sent_at:
                    oldest = key
            if count <= self._cap or oldest is None:
                return
            del self._heard[oldest]


def _forget_oldest[K, V](entries: dict[K, V], age: Callable[[V], Time], cap: int) -> None:
    """Forget the entries with the oldest ``age`` until at most ``cap`` are left."""
    while len(entries) > cap:
        oldest = min(entries, key=lambda key: age(entries[key]))
        del entries[oldest]


def _itself(at: Time) -> Time:
    return at


def _placed_at(placed: PlacedBeacon) -> Time:
    return placed.placed_at
