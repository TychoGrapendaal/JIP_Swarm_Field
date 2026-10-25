"""How hop beacons pull a robot: the sum of the attractions of the downhill beacons.

The robot's own value is the lowest fresh beacon it heard plus the hop to it (``own_value``).
Every fresh beacon below that value attracts the robot towards its cell, with its
``beacon_strength``. The sum of those attractions is the robot's pull, which ``HopField`` turns
into a heading and ``TreatByBeacon`` bids on. With ``killer_repulsion``, a killer's beacon
pushes instead, and the beacons the robot placed itself push it too.
"""

from dataclasses import dataclass
from math import exp, hypot

from swarm_field.contracts import BeaconKind, CellId, RobotId, Time, Vector2
from swarm_field.swarm.hop.memory import BeaconView, HeardBeacon, PlacedBeacon
from swarm_field.swarm.hop.relay import BeaconConfig, fresh_beacons, own_value


@dataclass(frozen=True)
class HopPull:
    """The summed pull of the downhill beacons on a robot.

    Attributes:
        vector: The sum of every beacon's pull, or push for a killer's beacon.
        strongest: The strongest attracting beacon.
        strength: That beacon's strength.
    """

    vector: Vector2
    strongest: HeardBeacon
    strength: float


def beacon_strength(heard: HeardBeacon, now: Time, config: BeaconConfig) -> float:
    """How strongly a beacon attracts: ``gamma`` per hop, decaying with the beacon's age."""
    return config.gamma**heard.beacon.value * exp(-(now - heard.sent_at) / config.decay)


def downhill_beacons(
    beacons: BeaconView,
    position: CellId,
    now: Time,
    config: BeaconConfig,
    kind: BeaconKind = BeaconKind.TREAT,
) -> list[HeardBeacon]:
    """The fresh beacons of ``kind`` that pull the robot: below its own value and off its cell.

    ``config`` must be the settings of ``kind`` (``BeaconConfig.for_kind``).

    A beacon on the robot's own cell pulls in no direction, so it is left out. A robot standing
    on a beacon's cell is as close to the pest as that beacon, so only beacons below it pull:
    its own value, which adds at least one hop, would let a neighbour just above that beacon
    pull it back, and the robot would shuttle between the two cells.
    """
    fresh = fresh_beacons(beacons, kind, now, config)
    if not fresh:
        return []
    here = own_value(fresh, position, config)
    for heard in fresh:
        if heard.beacon.cell == position:
            here = min(here, heard.beacon.value)
    downhill: list[HeardBeacon] = []
    for heard in fresh:
        if heard.beacon.value < here and heard.beacon.cell != position:
            downhill.append(heard)
    return downhill


def hop_pull(
    beacons: BeaconView,
    robot: CellId,
    now: Time,
    config: BeaconConfig,
    kind: BeaconKind = BeaconKind.TREAT,
) -> HopPull | None:
    """The pull of the ``downhill_beacons`` of ``kind``, or ``None`` if none attracts the robot.

    Beacons of ``kind`` are weighed with the settings of that kind (``BeaconConfig.for_kind``).

    A killer's beacon pushes with ``killer_repulsion`` times its strength instead, so that a
    killer closer to the pest counters the pull towards it, and the ``obstacle_push`` is added.
    With ``distance_power``, every strength is divided by the beacon's distance to that power,
    also when picking the strongest beacon. Pulls that cancel out exactly count as no pull.
    """
    config = config.for_kind(kind)
    killers: frozenset[RobotId] = frozenset()
    if config.killer_repulsion > 0:
        killers = beacons.killers()
    pull_x = 0.0
    pull_y = 0.0
    strongest: HeardBeacon | None = None
    strongest_strength = 0.0
    for heard in downhill_beacons(beacons, robot, now, config, kind):
        dx = heard.beacon.cell[0] - robot[0]
        dy = heard.beacon.cell[1] - robot[1]
        distance = hypot(dx, dy)
        strength = beacon_strength(heard, now, config) / max(distance, 1.0) ** config.distance_power
        if heard.sender in killers:
            strength *= -config.killer_repulsion
        pull_x += strength * dx / distance
        pull_y += strength * dy / distance
        if strength > strongest_strength:
            strongest = heard
            strongest_strength = strength
    if strongest is None:
        return None
    push_x, push_y = obstacle_push(beacons, robot, now, config)
    pull_x += push_x
    pull_y += push_y
    if hypot(pull_x, pull_y) == 0:
        return None
    return HopPull(vector=(pull_x, pull_y), strongest=strongest, strength=strongest_strength)


def obstacle_push(beacons: BeaconView, robot: CellId, now: Time, config: BeaconConfig) -> Vector2:
    """The summed push of the placed beacons within ``obstacle_range_cells`` of the robot.

    Each pushes away from its cell with its ``obstacle_strength``, falling linearly to nothing
    at the range. A beacon on the robot's own cell pushes in no direction.
    """
    push_x = 0.0
    push_y = 0.0
    for placed in beacons.placed_beacons():
        dx = placed.cell[0] - robot[0]
        dy = placed.cell[1] - robot[1]
        distance = hypot(dx, dy)
        if distance == 0 or distance >= config.obstacle_range_cells:
            continue
        falloff = 1 - distance / config.obstacle_range_cells
        push = obstacle_strength(placed, now) * falloff
        push_x -= push * dx / distance
        push_y -= push * dy / distance
    return (push_x, push_y)


def obstacle_strength(placed: PlacedBeacon, now: Time) -> float:
    """How hard a placed beacon pushes at its own cell: its strength, decaying with its age."""
    return placed.strength * exp(-(now - placed.placed_at) / placed.decay)
