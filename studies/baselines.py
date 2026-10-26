"""Baselines that the swarm is compared with: random waypoints and a grid search.

Importing this module registers both in the runner's registries. The random-waypoint baseline is
the behaviour and the seeder ``"random-waypoint"``, named together by ``RANDOM_WAYPOINT_BRAIN``.
The grid search is the seeder ``"grid-sweep"``: utility robots that each sweep a strip of the
field and treat what they drive over, without cooperating; ``grid_sweep_brain`` builds its brain,
and ``grid_explorer_brain`` the brain of an explorer that only sweeps its strip.
"""

from collections.abc import Hashable
from random import Random
from typing import Any

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import AgentObjective, CellId, Time, TimeUnit
from swarm_field.runner import BEHAVIORS, SEEDERS, BrainSpec
from swarm_field.swarm import (
    ActionCandidate,
    DecisionContext,
    FollowField,
    KnowledgeSharing,
    MemoryView,
    SharingConfig,
)
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


GIVE_UP_AFTER: Time = TimeUnit.MINUTE
"""How long a grid sweeper stands still before it gives up on its next cell as unreachable."""


class GridSweepSeeder:
    """A baseline: sweep a fixed strip of the field cell by cell, then start again.

    The field's columns are split into one strip per robot. A robot sweeps its strip in a lawnmower
    pattern, up one column and down the next, through every cell memory says is worth monitoring,
    and starts over at the end. Its strip is ``strip`` if given, else the one numbered by its
    starting column; the latter needs the robots side by side from column 0 with no obstacle in
    between, since a robot that starts past an obstacle would skip a strip.

    A robot that does not know the terrain plans its route over the obstacles too, and a ground
    robot can never stand on one. So it skips a cell once memory learns it is an obstacle, and
    gives up on a cell after standing still for ``GIVE_UP_AFTER``: the inside of a rock is never
    sensed, since obstacles are sensed only from a neighbouring cell.
    """

    name: str = "grid-sweep"
    window_blocks: int = 1

    def __init__(self, robots: int, strip: int | None = None) -> None:
        self._robots: int = robots
        self._strip: int | None = strip
        self._route: tuple[CellId, ...] = ()
        self._next: int = 0
        self._standing_at: CellId | None = None
        self._standing_since: Time = 0

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        if not self._route:
            strip = robot[0]
            if self._strip is not None:
                strip = self._strip
            self._route = sweep_route(memory.monitorable_cells(), strip, self._robots)
        if not self._route:
            return None
        if robot != self._standing_at:
            self._standing_at = robot
            self._standing_since = now
        if robot == self._route[self._next] or now - self._standing_since > GIVE_UP_AFTER:
            self._next = (self._next + 1) % len(self._route)
            self._standing_since = now
        self._skip_known_obstacles(memory)
        return self._route[self._next]

    def _skip_known_obstacles(self, memory: MemoryView) -> None:
        """Move past the route cells that memory has learned are obstacles."""
        monitorable = memory.monitorable_cells()
        for _ in range(len(self._route)):
            if monitorable[self._route[self._next]]:
                return
            self._next = (self._next + 1) % len(self._route)

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
    """The baseline's seeder, for the number of robots under ``"grid-sweep-robots"`` and the
    robot's strip under ``"grid-sweep-strip"``, if given."""
    strip: int | None = params.get("grid-sweep-strip")
    return GridSweepSeeder(int(params["grid-sweep-robots"]), strip)


def grid_sweep_brain(robots: int, strip: int | None = None) -> BrainSpec:
    """The brain of a grid-sweep utility robot among ``robots``: treat the pest underfoot, sweep.

    ``TreatInfestation`` treats a pest the robot stands on, with an override bid, and otherwise
    sends it to the pests it knows of; a robot that shares nothing only knows the pests it has
    driven over, which it has treated, so in effect it sweeps and treats what it finds. With
    ``strip`` the robot sweeps that strip, else the one numbered by its starting column. It
    broadcasts none of its memory, so even within radio range it never hears of a pest on
    another strip.
    """
    return BrainSpec(
        behaviors=("treat-infestation", GridSweepSeeder.name),
        seeders=("treat", GridSweepSeeder.name),
        params=_grid_sweep_params(robots, strip),
        sharing=_SHARES_NOTHING,
    )


def grid_explorer_brain(robots: int, strip: int | None = None) -> BrainSpec:
    """The brain of a grid-sweep explorer among ``robots``: sweep its strip, share nothing.

    The explorer's counterpart of ``grid_sweep_brain``, without treating: the baseline that the
    EXPLORE field's exploration is compared with. ``strip`` is as in ``grid_sweep_brain``.
    """
    return BrainSpec(
        behaviors=(GridSweepSeeder.name,),
        seeders=(GridSweepSeeder.name,),
        params=_grid_sweep_params(robots, strip),
        sharing=_SHARES_NOTHING,
    )


_SHARES_NOTHING = SharingConfig(facts=KnowledgeSharing.NOTHING)
"""A grid robot broadcasts only its status, none of its memory."""


def _grid_sweep_params(robots: int, strip: int | None) -> dict[str, int]:
    params: dict[str, int] = {"grid-sweep-robots": robots}
    if strip is not None:
        params["grid-sweep-strip"] = strip
    return params
