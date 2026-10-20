"""Hop beacons: the value a robot broadcasts so that its peers can find an objective.

A robot that saw a pest itself is the TREAT source: it beacons 0 on the pest's cell. Every
other robot relays: it beacons one more than the lowest fresh beacon it heard, on its own cell.
The values form a hop gradient that falls towards the pest.
"""

from dataclasses import dataclass

from swarm_field.contracts import AgentObjective, Beacon, CellId, RobotId, Time, TimeUnit
from swarm_field.swarm.memory import MemoryView


@dataclass(frozen=True)
class BeaconConfig:
    """When a heard beacon or a robot's own sighting stops counting.

    Attributes:
        fresh_for: How long a heard beacon counts after it was sent.
        source_for: How long a robot stays a pest's source after it saw the pest itself.
    """

    fresh_for: Time = TimeUnit.MINUTE
    source_for: Time = TimeUnit.MINUTE


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

    lowest: float | None = None
    for heard in memory.beacons(AgentObjective.TREAT):
        if now - heard.sent_at > config.fresh_for:
            continue
        if lowest is None or heard.beacon.value < lowest:
            lowest = heard.beacon.value
    if lowest is None:
        return None
    return Beacon(AgentObjective.TREAT, lowest + 1, position)
