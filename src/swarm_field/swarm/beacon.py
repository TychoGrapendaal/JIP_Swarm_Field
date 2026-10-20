"""Hop beacons: the value a robot broadcasts so that its peers can find an objective.

A robot that saw a pest itself is the TREAT source: it beacons 0 on the pest's cell. Every
other robot relays: it beacons one more than the lowest fresh beacon it heard, on its own cell.
The values form a hop gradient that falls towards the pest.
"""

from dataclasses import dataclass
from math import exp

from swarm_field.contracts import AgentObjective, Beacon, CellId, RobotId, Time, TimeUnit
from swarm_field.swarm.memory import HeardBeacon, MemoryView


@dataclass(frozen=True)
class BeaconConfig:
    """How long beacons and sightings count, and how strongly a beacon attracts.

    Attributes:
        fresh_for: How long a heard beacon counts after it was sent.
        source_for: How long a robot stays a pest's source after it saw the pest itself.
        gamma: The hop discount, between 0 and 1: each hop further away attracts this much less.
        decay: The decay time of a beacon's attraction with its age.
    """

    fresh_for: Time = TimeUnit.MINUTE
    source_for: Time = TimeUnit.MINUTE
    gamma: float = 0.5
    decay: Time = TimeUnit.MINUTE


def fresh_beacons(
    memory: MemoryView, objective: AgentObjective, now: Time, config: BeaconConfig
) -> list[HeardBeacon]:
    """The beacons for ``objective`` heard within ``fresh_for``."""
    fresh: list[HeardBeacon] = []
    for heard in memory.beacons(objective):
        if now - heard.sent_at <= config.fresh_for:
            fresh.append(heard)
    return fresh


def beacon_strength(heard: HeardBeacon, now: Time, config: BeaconConfig) -> float:
    """How strongly a beacon attracts: ``gamma`` per hop, decaying with the beacon's age."""
    return config.gamma**heard.beacon.value * exp(-(now - heard.sent_at) / config.decay)


def treat_beacon(
    memory: MemoryView, robot: RobotId, position: CellId, now: Time, config: BeaconConfig
) -> Beacon | None:
    """The robot's TREAT beacon, or ``None`` if it neither saw a pest nor heard a fresh beacon.

    A robot that saw several pests within ``source_for`` beacons the one it saw last.
    """
    sighting_cell: CellId | None = None
    sighting_time: Time = now - config.source_for
    for pest in memory.infestations():
        if pest.source == robot and pest.severity > 0 and pest.observed_at >= sighting_time:
            sighting_cell = pest.cell
            sighting_time = pest.observed_at
    if sighting_cell is not None:
        return Beacon(AgentObjective.TREAT, 0.0, sighting_cell)

    fresh = fresh_beacons(memory, AgentObjective.TREAT, now, config)
    if not fresh:
        return None
    lowest = min(heard.beacon.value for heard in fresh)
    return Beacon(AgentObjective.TREAT, lowest + 1, position)
