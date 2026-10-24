"""Relaying hop beacons: the value a robot broadcasts so that its peers can find an objective.

A robot that saw a pest itself is the TREAT source: it beacons 0 on the pest's cell. Every
other robot relays its ``own_value`` on its own cell: the lowest fresh beacon it heard plus the
hop to it, which costs 1 or, with metric hops, the distance in radio ranges. The values form a
hop gradient that falls towards the pest. How the gradient pulls a robot is in
``swarm_field.swarm.hop.pull``.
"""

from dataclasses import dataclass
from math import hypot

from swarm_field.contracts import Beacon, BeaconKind, CellId, Time, TimeUnit
from swarm_field.swarm.hop.escape import EscapeConfig
from swarm_field.swarm.hop.memory import BeaconView, HeardBeacon


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

    fresh = fresh_beacons(beacons, BeaconKind.TREAT, now, config)
    if not fresh:
        return None
    value = own_value(fresh, position, config)
    if config.max_value is not None and value > config.max_value:
        return None
    return Beacon(BeaconKind.TREAT, value, position)
