"""Baselines that the swarm is compared with: random waypoints and a grid search.

Importing this module registers both in the runner's registries. The random-waypoint baseline is
the behaviour and the seeder ``"random-waypoint"``, named together by ``RANDOM_WAYPOINT_BRAIN``.
The grid search is the seeder ``"grid-sweep"``: utility robots that each sweep a strip of the
field and treat what they drive over, without cooperating; ``grid_sweep_brain`` builds its brain.
"""

from collections.abc import Hashable
from random import Random
from typing import Any

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import CellId, Time
from swarm_field.runner import BEHAVIORS, SEEDERS, BrainSpec
from swarm_field.swarm import FollowSeederField, MemoryView
from swarm_field.swarm.field import Seed


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


class GridSweepSeeder:
    """A baseline: sweep a fixed strip of the field cell by cell, then start again.

    The field's columns are split into one strip per robot. A robot sweeps its strip in a lawnmower
    pattern, up one column and down the next, through every cell memory says is worth monitoring,
    and starts over at the end. Its strip is the one numbered by its starting column, so the robots
    must start side by side from column 0, as ``PestScenario.starts`` puts them.
    """

    name: str = "grid-sweep"
    window_blocks: int = 1

    def __init__(self, robots: int) -> None:
        self._robots: int = robots
        self._route: tuple[CellId, ...] = ()
        self._next: int = 0

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        if not self._route:
            self._route = sweep_route(memory.monitorable_cells(), robot[0], self._robots)
        if not self._route:
            return None
        if robot == self._route[self._next]:
            self._next = (self._next + 1) % len(self._route)
        return self._route[self._next]

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        if not self._route:
            return ()
        return (Seed(self._route[self._next], 0.0),)


def sweep_route(monitorable: NDArray[np.bool_], strip: int, strips: int) -> tuple[CellId, ...]:
    """The cells of one strip of columns, in lawnmower order: up the first column, down the next.

    The field's columns are split as evenly as possible into ``strips`` strips; ``strip`` counts
    from the left and is clamped to the last one.
    """
    width, height = monitorable.shape
    strip = min(strip, strips - 1)
    first = strip * width // strips
    last = (strip + 1) * width // strips
    route: list[CellId] = []
    for column, x in enumerate(range(first, last)):
        rows = range(height)
        if column % 2 == 1:
            rows = range(height - 1, -1, -1)
        for y in rows:
            if monitorable[x, y]:
                route.append((x, y))
    return tuple(route)


@BEHAVIORS.register(GridSweepSeeder.name)
def _build_grid_sweep(params: dict[str, Any], rng: Random) -> FollowSeederField:
    """The baseline's behaviour: follow the grid-sweep seeder's field."""
    return FollowSeederField(GridSweepSeeder.name)


@SEEDERS.register(GridSweepSeeder.name)
def _build_grid_sweep_seeder(params: dict[str, Any], rng: Random) -> GridSweepSeeder:
    """The baseline's seeder, for the number of robots under ``"grid-sweep-robots"``."""
    return GridSweepSeeder(int(params["grid-sweep-robots"]))


def grid_sweep_brain(robots: int) -> BrainSpec:
    """The brain of a grid-sweep utility robot among ``robots``: treat the pest underfoot, sweep.

    ``TreatInfestation`` treats a pest the robot stands on, with an override bid, and otherwise
    sends it to the pests it knows of; a robot that shares nothing only knows the pests it has
    driven over, which it has treated, so in effect it sweeps and treats what it finds.
    """
    return BrainSpec(
        behaviors=("treat-infestation", GridSweepSeeder.name),
        seeders=("treat", GridSweepSeeder.name),
        params={"grid-sweep-robots": robots},
    )
