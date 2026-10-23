"""Baselines that the exploration tests and benchmark compare the EXPLORE seeder with.

Importing this module registers the random-waypoint baseline in the runner's registries, as the
behaviour and the seeder ``"random-waypoint"``; ``RANDOM_WAYPOINT_BRAIN`` names both.
"""

from collections.abc import Hashable
from random import Random
from typing import Any

import numpy as np

from swarm_field.contracts import AgentObjective, CellId, Time
from swarm_field.runner import BEHAVIORS, SEEDERS, BrainSpec
from swarm_field.swarm import ActionCandidate, DecisionContext, FollowField, MemoryView
from swarm_field.swarm.field import Seed


class FollowSeederField:
    """A behaviour that always bids 0.5 to follow one seeder's field, for the objective EXPLORE.

    As the only behaviour of a robot it always wins, so the robot goes wherever the seeder says.
    """

    name: str = "follow-field"

    def __init__(self, seeder: str) -> None:
        self._seeder: str = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        candidate = ActionCandidate(
            FollowField(self._seeder), 0.5, self.name, target=None, objective=AgentObjective.EXPLORE
        )
        return (candidate,)


class RandomWaypointSeeder:
    """A baseline: one target cell drawn at random, and a new one once the robot stands on it.

    The targets are drawn from the cells memory says are worth monitoring, with the seeder's own
    random source, so the robots' draws do not depend on each other.
    """

    name: str = "random-waypoint"
    window_blocks: int = 1

    def __init__(self, rng: Random) -> None:
        self._rng: Random = rng
        self._target: CellId | None = None

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        if self._target is None or robot == self._target:
            self._target = self._draw_target(memory)
        return self._target

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        if self._target is None:
            return ()
        return (Seed(self._target, 0.0),)

    def _draw_target(self, memory: MemoryView) -> CellId | None:
        cells: list[CellId] = []
        for x, y in np.argwhere(memory.monitorable_cells()):
            cells.append((int(x), int(y)))
        if not cells:
            return None
        return self._rng.choice(cells)


@BEHAVIORS.register(RandomWaypointSeeder.name)
def _build_random_waypoint(params: dict[str, Any], rng: Random) -> FollowSeederField:
    """The baseline's behaviour: follow the random-waypoint seeder's field."""
    return FollowSeederField(RandomWaypointSeeder.name)


@SEEDERS.register(RandomWaypointSeeder.name)
def _build_random_waypoint_seeder(params: dict[str, Any], rng: Random) -> RandomWaypointSeeder:
    """The baseline's seeder, with its own random source drawn now from ``rng``."""
    return RandomWaypointSeeder(Random(rng.getrandbits(64)))


RANDOM_WAYPOINT_BRAIN = BrainSpec(
    behaviors=(RandomWaypointSeeder.name,), seeders=(RandomWaypointSeeder.name,), params={}
)
"""The brain of a random-waypoint robot, to put in an ``ExperimentConfig``."""
