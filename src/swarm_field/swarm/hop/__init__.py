"""The hop method: robots relay hop beacons, and killers walk down them to a pest.

A robot of the hop method remembers only beacons (``BeaconMemory``), never a map. Every hop rule
reads them through ``BeaconView``. A ``BeaconLink`` per robot connects the method to the brain:
it hears and sends beacons, and places the beacons that push the robot off a wall or out of a
place it circles in. ``HopField`` is the method's objective field and ``TreatByBeacon`` its
behaviour. The model is described in ``docs/research/hop-field.md``.
"""

from swarm_field.swarm.hop.behavior import TreatByBeacon
from swarm_field.swarm.hop.escape import EscapeBeacon, EscapeConfig, StuckEscape, cell_ahead
from swarm_field.swarm.hop.field import HopField
from swarm_field.swarm.hop.link import BeaconLink
from swarm_field.swarm.hop.memory import BeaconMemory, BeaconView, HeardBeacon, PlacedBeacon
from swarm_field.swarm.hop.pull import (
    HopPull,
    beacon_strength,
    downhill_beacons,
    hop_pull,
    obstacle_push,
    obstacle_strength,
)
from swarm_field.swarm.hop.relay import BeaconConfig, fresh_beacons, own_value, treat_beacon

__all__ = [
    "BeaconConfig",
    "BeaconLink",
    "BeaconMemory",
    "BeaconView",
    "EscapeBeacon",
    "EscapeConfig",
    "HeardBeacon",
    "HopField",
    "HopPull",
    "PlacedBeacon",
    "StuckEscape",
    "TreatByBeacon",
    "beacon_strength",
    "cell_ahead",
    "downhill_beacons",
    "fresh_beacons",
    "hop_pull",
    "obstacle_push",
    "obstacle_strength",
    "own_value",
    "treat_beacon",
]
