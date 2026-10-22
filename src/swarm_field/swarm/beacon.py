"""Hop beacons: the value a robot broadcasts so that its peers can find an objective.

A robot that saw a pest itself is the TREAT source: it beacons 0 on the pest's cell. Every
other robot relays its ``own_value`` on its own cell: the lowest fresh beacon it heard plus the
hop to it, which costs 1 or, with metric hops, the distance in radio ranges. The values form a
hop gradient that falls towards the pest.
"""

from dataclasses import dataclass
from math import exp, hypot

from swarm_field.contracts import Beacon, BeaconKind, CellId, RobotId, Time, TimeUnit
from swarm_field.swarm.escape import EscapeConfig
from swarm_field.swarm.memory import HeardBeacon, MemoryView


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
        obstacle_repulsion: How hard an obstacle beacon pushes at its own cell; 0 for no
            obstacle beacons at all.
        obstacle_range_cells: How far an obstacle beacon pushes; its push falls linearly to
            nothing at this distance.
        obstacle_decay: The decay time of an obstacle beacon's push with its age.
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
    obstacle_repulsion: float = 0.0
    obstacle_range_cells: float = 3.0
    obstacle_decay: Time = 10 * TimeUnit.MINUTE
    escape: EscapeConfig | None = None


def fresh_beacons(
    memory: MemoryView, kind: BeaconKind, now: Time, config: BeaconConfig
) -> list[HeardBeacon]:
    """The beacons of ``kind`` heard within ``fresh_for``, up to ``max_value``."""
    fresh: list[HeardBeacon] = []
    for heard in memory.beacons(kind):
        if now - heard.sent_at > config.fresh_for:
            continue
        if config.max_value is not None and heard.beacon.value > config.max_value:
            continue
        fresh.append(heard)
    return fresh


def beacon_strength(heard: HeardBeacon, now: Time, config: BeaconConfig) -> float:
    """How strongly a beacon attracts: ``gamma`` per hop, decaying with the beacon's age."""
    return config.gamma**heard.beacon.value * exp(-(now - heard.sent_at) / config.decay)


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


def downhill_beacons(
    memory: MemoryView, position: CellId, now: Time, config: BeaconConfig
) -> list[HeardBeacon]:
    """The fresh TREAT beacons that pull the robot: below its own value and off its cell.

    A beacon on the robot's own cell pulls in no direction, so it is left out.
    """
    fresh = fresh_beacons(memory, BeaconKind.TREAT, now, config)
    if not fresh:
        return []
    here = own_value(fresh, position, config)
    downhill: list[HeardBeacon] = []
    for heard in fresh:
        if heard.beacon.value < here and heard.beacon.cell != position:
            downhill.append(heard)
    return downhill


def treat_beacon(
    memory: MemoryView, robot: RobotId, position: CellId, now: Time, config: BeaconConfig
) -> Beacon | None:
    """The robot's TREAT beacon, or ``None`` if it neither saw a pest nor heard a fresh beacon.

    A robot that saw several pests within ``source_for`` beacons the one it saw last. A relay
    whose value would be above ``max_value`` beacons nothing.
    """
    sighting_cell: CellId | None = None
    sighting_time: Time = now - config.source_for
    for pest in memory.infestations():
        if pest.source == robot and pest.severity > 0 and pest.observed_at >= sighting_time:
            sighting_cell = pest.cell
            sighting_time = pest.observed_at
    if sighting_cell is not None:
        return Beacon(BeaconKind.TREAT, 0.0, sighting_cell)

    fresh = fresh_beacons(memory, BeaconKind.TREAT, now, config)
    if not fresh:
        return None
    value = own_value(fresh, position, config)
    if config.max_value is not None and value > config.max_value:
        return None
    return Beacon(BeaconKind.TREAT, value, position)
