"""Relaying hop beacons: the value a robot broadcasts so that its peers can find an objective.

A robot that saw a pest itself is the TREAT source: it beacons 0 on the pest's cell. Every
other robot relays its ``own_value`` on its own cell: the lowest fresh beacon it heard plus the
hop to it, which costs 1 or, with metric hops, the distance in radio ranges. The values form a
hop gradient that falls towards the pest. DOCK beacons work the same way, with every dock as
their source (``DockBeacons``). How the gradient pulls a robot is in
``swarm_field.swarm.hop.pull``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot

from swarm_field.contracts import Beacon, BeaconKind, CellId, Time, TimeUnit
from swarm_field.swarm.hop.escape import EscapeConfig
from swarm_field.swarm.hop.memory import BeaconView, HeardBeacon


@dataclass(frozen=True)
class DockBeacons:
    """How DOCK beacons differ from TREAT beacons: a dock never moves or vanishes.

    A DOCK beacon heard from a cell stays true as long as the dock stands, so it is kept and
    weighed for long and never capped, and killers do not repel each other on the way to a dock.

    Attributes:
        fresh_for: How long a heard DOCK beacon counts after it was sent.
        decay: The decay time of its pull with its age.
        gamma: The hop discount of DOCK beacons. Nearer 1 than TREAT's, since a dock may lie
            many hops away, where a discount of 0.5 per hop would leave no pull worth following.
        max_value: DOCK beacons above this are ignored and not relayed; ``None`` for no limit.
    """

    fresh_for: Time = 30 * TimeUnit.MINUTE
    decay: Time = 30 * TimeUnit.MINUTE
    gamma: float = 0.8
    max_value: float | None = None


@dataclass(frozen=True)
class BeaconConfig:
    """How beacons are kept, relayed and weighed.

    Attributes:
        fresh_for: How long a heard beacon counts after it was sent.
        source_for: How long a robot stays a pest's source after it saw the pest itself.
        gamma: The hop discount, between 0 and 1: a beacon attracts ``gamma`` to the power of
            its value.
        decay: The decay time of a beacon's attraction with its age.
        hop_cells: With a length, a hop costs the distance to the beacon's cell in units of
            this many cells (metric hops); without one, every hop costs 1.
        min_hop_cells: With metric hops, the distance a hop costs at least. Without it, two
            robots on one cell would hold each other's value at no cost, forever.
        max_value: Beacons above this value are ignored, and a robot above it beacons nothing,
            so that values left behind by a cleared pest rise past it and vanish; ``None``
            for no limit.
        killer_repulsion: How hard a killer's downhill beacon pushes, relative to how hard
            another robot's beacon of the same value pulls; 0 for no repulsion.
        distance_power: How a beacon's pull falls with its distance: it is divided by the
            distance in cells, at least one, to this power. 0 pulls equally from any distance,
            which holds a robot between several beacons at their geometric median; 1 makes such
            a point unstable, so the robot falls towards one of them.
        obstacle_range_cells: How far a beacon the robot placed to escape pushes; its push falls
            linearly to nothing at this distance.
        escape: When a robot counts as stuck and how it escapes, or ``None`` for no escape.
        dock: How DOCK beacons are kept and weighed, or ``None`` for a swarm that relays none.
            The other fields hold for TREAT beacons; ``for_kind`` gives the settings of a kind.
    """

    fresh_for: Time = TimeUnit.MINUTE
    source_for: Time = TimeUnit.MINUTE
    gamma: float = 0.5
    decay: Time = TimeUnit.MINUTE
    hop_cells: float | None = None
    min_hop_cells: float = 1.0
    max_value: float | None = None
    killer_repulsion: float = 0.0
    distance_power: float = 0.0
    obstacle_range_cells: float = 3.0
    escape: EscapeConfig | None = None
    dock: DockBeacons | None = None

    def for_kind(self, kind: BeaconKind) -> BeaconConfig:
        """These settings as they hold for beacons of ``kind``: themselves for TREAT, and for
        DOCK with the freshness, decay, discount and limit of ``dock`` and no killer repulsion.

        Raises:
            ValueError: For DOCK, if the configuration relays no DOCK beacons.
        """
        if kind is BeaconKind.TREAT:
            return self
        if self.dock is None:
            raise ValueError("this beacon configuration relays no DOCK beacons")
        return replace(
            self,
            fresh_for=self.dock.fresh_for,
            decay=self.dock.decay,
            gamma=self.dock.gamma,
            max_value=self.dock.max_value,
            killer_repulsion=0.0,
        )


def fresh_beacons(
    beacons: BeaconView, kind: BeaconKind, now: Time, config: BeaconConfig
) -> list[HeardBeacon]:
    """The beacons of ``kind`` heard within ``fresh_for``, up to ``max_value``."""
    fresh: list[HeardBeacon] = []
    for heard in beacons.beacons(kind):
        if now - heard.sent_at > config.fresh_for:
            continue
        if config.max_value is not None and heard.beacon.value > config.max_value:
            continue
        fresh.append(heard)
    return fresh


def own_value(fresh: list[HeardBeacon], position: CellId, config: BeaconConfig) -> float:
    """The value at ``position``: the lowest of the beacons' values plus the hop to each.

    ``fresh`` must not be empty.
    """
    values: list[float] = []
    for heard in fresh:
        hop = 1.0
        if config.hop_cells is not None:
            cell = heard.beacon.cell
            distance = hypot(cell[0] - position[0], cell[1] - position[1])
            hop = max(distance, config.min_hop_cells) / config.hop_cells
        values.append(heard.beacon.value + hop)
    return min(values)


def treat_beacon(
    beacons: BeaconView, position: CellId, now: Time, config: BeaconConfig
) -> Beacon | None:
    """The robot's TREAT beacon, or ``None`` if it neither saw a pest nor heard a fresh beacon.

    A robot that saw a pest itself within ``source_for`` is its source; one that saw several
    beacons the one it saw last. A relay whose value would be above ``max_value`` beacons
    nothing.
    """
    sighting = beacons.latest_sighting(now - config.source_for)
    if sighting is not None:
        return Beacon(BeaconKind.TREAT, 0.0, sighting)
    return relayed_beacon(beacons, BeaconKind.TREAT, position, now, config)


def dock_beacon(
    beacons: BeaconView, position: CellId, now: Time, config: BeaconConfig
) -> Beacon | None:
    """The robot's DOCK beacon, or ``None`` if the swarm relays none or it heard none fresh.

    A robot is never a dock's source; the docks broadcast their own beacons.
    """
    if config.dock is None:
        return None
    return relayed_beacon(beacons, BeaconKind.DOCK, position, now, config.for_kind(BeaconKind.DOCK))


def relayed_beacon(
    beacons: BeaconView, kind: BeaconKind, position: CellId, now: Time, config: BeaconConfig
) -> Beacon | None:
    """The robot's own value of ``kind`` on its own cell, as a relay.

    ``None`` without a fresh beacon of ``kind``, and for a value above ``max_value``. ``config``
    must be the settings of ``kind`` (``BeaconConfig.for_kind``).
    """
    fresh = fresh_beacons(beacons, kind, now, config)
    if not fresh:
        return None
    value = own_value(fresh, position, config)
    if config.max_value is not None and value > config.max_value:
        return None
    return Beacon(kind, value, position)
