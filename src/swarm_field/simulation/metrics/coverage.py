"""Ground-truth coverage: when each cell was last visited by any robot, and what that is worth.

It measures how well a swarm explores and monitors the field, independently of what the robots
believe. A robot observes the cell it stands on, so a visit is an observation. ``CoverageSample``
records the robots' cells at a fixed period; a period shorter than any move misses no visit.

The measures, all over the cells that are not obstacles:

- coverage: the share of cells visited at least once, and when a share was first reached;
- idleness: the age of a cell, ``now - last visit``, averaged over cells and samples, and its
  maximum. For pests appearing at uniformly random times and cells, the mean idleness is the
  expected time until the next visit, i.e. the mean detection latency;
- redundancy: the share of visits to a cell some robot visited less than ``fresh_age`` before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import CellId, RobotId, Time
from swarm_field.simulation.world import WorldState

if TYPE_CHECKING:
    from swarm_field.simulation.context import SimulationContext


@dataclass(frozen=True)
class CoverageSummary:
    """The coverage measures of a run so far.

    Attributes:
        covered: The share of cells visited at least once.
        time_to_90: When 90 % of the cells had been visited, or ``None``.
        time_to_full: When every cell had been visited, or ``None``.
        mean_idleness: The age of a cell averaged over cells and the samples since
            ``measure_from``; a cell never visited is as old as the run.
        worst_idleness: The largest age of a cell in those samples.
        redundancy: The share of visits to a cell visited less than ``fresh_age`` before.
        visits: The number of visits.
    """

    covered: float
    time_to_90: Time | None
    time_to_full: Time | None
    mean_idleness: float
    worst_idleness: float
    redundancy: float
    visits: int


class CoverageTracker:
    """Records the robots' visits to cells and the measures derived from them.

    Attributes:
        start: When the run started; cells never visited count as visited then.
        fresh_age: A visit to a cell younger than this is redundant.
        measure_from: Idleness is averaged over the samples from this time on, e.g. after a
            warm-up in which the swarm explores the field for the first time.
    """

    def __init__(
        self, world: WorldState, fresh_age: Time, start: Time = 0, measure_from: Time = 0
    ) -> None:
        self.start: Time = start
        self.fresh_age: Time = fresh_age
        self.measure_from: Time = measure_from
        width, height = world.field.width, world.field.height
        self._monitored: NDArray[np.bool_] = np.ones((width, height), dtype=np.bool_)
        for (x, y), state in world.cells.items():
            self._monitored[x, y] = not state.obstacle
        self._last_visit: NDArray[np.float64] = np.full((width, height), -np.inf)
        self._positions: dict[RobotId, CellId] = {}
        self._covered_at: list[Time] = []
        self._visits: int = 0
        self._redundant: int = 0
        self._idleness_sum: float = 0.0
        self._idleness_samples: int = 0
        self._worst_idleness: float = 0.0

    def record(self, world: WorldState, now: Time) -> None:
        """Count the robots that arrived in a new cell since the last record, and sample ages."""
        for robot_id, robot in world.robots.items():
            if self._positions.get(robot_id) != robot.position:
                self._positions[robot_id] = robot.position
                self._visit(robot.position, now)
        if now >= self.measure_from:
            ages = now - np.maximum(self._last_visit, self.start)[self._monitored]
            self._idleness_sum += float(ages.mean())
            self._idleness_samples += 1
            self._worst_idleness = max(self._worst_idleness, float(ages.max()))

    def last_visits(self) -> NDArray[np.float64]:
        """When each cell was last visited, indexed ``[x, y]``; ``-inf`` for never. Read-only."""
        view = self._last_visit.view()
        view.flags.writeable = False
        return view

    def summary(self) -> CoverageSummary:
        """The measures over everything recorded so far."""
        cells = int(self._monitored.sum())
        mean_idleness = 0.0
        if self._idleness_samples > 0:
            mean_idleness = self._idleness_sum / self._idleness_samples
        redundancy = 0.0
        if self._visits > 0:
            redundancy = self._redundant / self._visits
        return CoverageSummary(
            covered=len(self._covered_at) / max(cells, 1),
            time_to_90=self._time_to_cover(0.9, cells),
            time_to_full=self._time_to_cover(1.0, cells),
            mean_idleness=mean_idleness,
            worst_idleness=self._worst_idleness,
            redundancy=redundancy,
            visits=self._visits,
        )

    def _visit(self, cell: CellId, now: Time) -> None:
        if not self._monitored[cell]:
            return
        self._visits += 1
        last = self._last_visit[cell]
        if last == -np.inf:
            self._covered_at.append(now)
        elif now - last < self.fresh_age:
            self._redundant += 1
        self._last_visit[cell] = now

    def _time_to_cover(self, share: float, cells: int) -> Time | None:
        """When ``share`` of the ``cells`` had been visited, or ``None`` if not yet."""
        needed = max(1, int(np.ceil(share * cells)))
        if len(self._covered_at) < needed:
            return None
        return self._covered_at[needed - 1]


@dataclass
class CoverageSample:
    """Record the robots' cells in a ``CoverageTracker``, then reschedule at the next period.

    Attributes:
        time: When this sample is taken.
        period: The time between samples; shorter than any move, so no visit is missed.
        tracker: Where the samples go.
    """

    time: Time
    period: Time
    tracker: CoverageTracker

    def handle(self, ctx: SimulationContext) -> None:
        ctx.world.advance_to(self.time)
        self.tracker.record(ctx.world, self.time)
        ctx.schedule(CoverageSample(self.time + self.period, self.period, self.tracker))
