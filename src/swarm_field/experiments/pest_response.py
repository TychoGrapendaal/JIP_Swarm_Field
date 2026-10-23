"""Measure how a swarm responds to pests: one run of a ``SwarmScenario`` into a ``PestResponse``.

``measure_pest_response`` is a run function for ``Experiment.run``: it runs a ``PestRunSpec``
with the run's seed, advancing one decision interval at a time, and a ``PestTracker`` samples
the ground truth and the killers' brains between steps. ``pest_table_headers`` and
``pest_table_cells`` turn the results of an arm into a row of the standard table, with the mean
over the seeds and its standard error (±) where it matters:

- ``infested``: infested cells, averaged over the run after the warm-up. The main outcome: it
  counts both how many pests stay and how long, and has no end-of-run ceiling;
- ``treated``: pests cleared during the run, which only treatment does;
- ``response``: minutes from a pest's first appearance to its clearing, over the cleared pests;
- ``p90``: the 90th percentile of those minutes;
- ``open``: pests still infested at the end;
- ``cells/treat``: cells moved by all killers per treated pest;
- ``msgs``: messages received per robot per hour;
- ``stuck``: the share of the time a killer committed to treating stands idle, e.g. at a
  dead end of the hop field;
- ``commit hop``: the median beacon value of a killer when it commits to treating, for arms
  with beacons;
- ``empty``: killers that used up their treatment consumable;
- ``coverage``: the share of the crop cells some robot sensed by the end;
- ``idle``: minutes since a robot last sensed a crop cell, averaged over the cells and the
  samples after the warm-up: the expected wait of a pest for its first sighting;
- ``redund``: the share of the robots' visits to a cell sensed less than ``fresh_age`` before;
- ``drift 1/2/3 h``: mean distance in cells of the killers from their start after each hour;
- ``wall``: seconds of computation per run.

The tracker reads the killers' goals and beacons, which the simulation's own metrics may not,
so it lives here, at the composition level, next to ``LiveRun``. Coverage, idleness and
redundancy come from the run's ``MetricsCollector``, sampled once a minute.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from statistics import mean, median, quantiles

from swarm_field.contracts import BodyType, CellId, Idle, RobotId, Time, TimeUnit
from swarm_field.experiments.batch import Run, mean_and_error
from swarm_field.experiments.run import LiveRun
from swarm_field.experiments.scenario import SwarmScenario
from swarm_field.runner import ExperimentConfig
from swarm_field.swarm import KnowledgeSharing, treat_beacon


@dataclass(frozen=True)
class PestRunSpec:
    """What to run and measure: a scenario for ``hours`` at ``pests_per_hour``.

    Attributes:
        scenario: The pest scenario; its seed, length and pest rate are set per run.
        hours: How long a run lasts.
        pests_per_hour: How many pests spawn per hour, on average.
        warmup_hours: How long at the start is left out of ``infested``.
    """

    scenario: SwarmScenario
    hours: int
    pests_per_hour: float = 15.0
    warmup_hours: int = 1

    def __post_init__(self) -> None:
        """Refuse a warm-up that leaves nothing to measure ``infested`` over.

        Raises:
            ValueError: If the warm-up lasts as long as the run, or longer.
        """
        if self.warmup_hours >= self.hours:
            raise ValueError(
                f"a warm-up of {self.warmup_hours} h leaves nothing of a {self.hours} h run"
            )

    def into_experiment(self, label: str, seed: int) -> ExperimentConfig:
        """The run of this spec with ``seed`` as an experiment for ``swarm_field.runner``.

        The scenario is sampled once a minute, its idleness measured after the warm-up, and it
        runs for ``hours`` at ``pests_per_hour``.
        """
        scenario = replace(
            self.scenario,
            until=self.hours * TimeUnit.HOUR,
            metrics=replace(
                self.scenario.metrics,
                period=TimeUnit.MINUTE,
                measure_from=self.warmup_hours * TimeUnit.HOUR,
            ),
            seed=seed,
            pests=replace(self.scenario.pests, pests_per_hour=self.pests_per_hour),
        )
        return scenario.into_experiment(label)

    def cost(self) -> float:
        """A rough relative cost: the field's cells, five times when relays gossip pests."""
        cost = float(self.scenario.map.size**2 * self.hours)
        if self.scenario.relaying().pest_facts is not KnowledgeSharing.NOTHING:
            cost *= 5
        return cost


@dataclass(frozen=True)
class PestResponse:
    """What one run measured; the attributes are the standard table's columns.

    Attributes:
        infested: Infested cells, averaged over the samples after the warm-up.
        treated: Pests cleared during the run.
        response_minutes: Mean minutes from a pest's first sample to its clearing, if any.
        response_p90: The 90th percentile of those minutes, if any pest was cleared.
        open: Pests still infested at the end.
        cells_per_treatment: Cells moved by all killers per cleared pest, if any.
        messages_per_robot_hour: Messages received per robot per hour.
        stuck: The share of committed killer samples in which the killer stood idle.
        commit_hops: The killer's beacon value each time it committed to treating.
        empty_killers: Killers that used up their treatment consumable.
        consumable_used: Treatment consumable used by all killers together, each starting with
            1; a treatment job uses ``1 / treatment_doses`` of the scenario.
        drift: The killers' mean distance in cells from their start, after each hour.
        coverage: The share of the crop cells some robot sensed by the end of the run.
        idleness_minutes: Minutes since a robot last sensed a crop cell, averaged over the cells
            and the samples after the warm-up.
        worst_idleness_minutes: The longest of those, over the same samples.
        redundancy: The share of the robots' visits to a crop cell sensed less than the
            scenario's ``fresh_age`` before.
        wall: Seconds the run took.
    """

    infested: float
    treated: int
    response_minutes: float | None
    response_p90: float | None
    open: int
    cells_per_treatment: float | None
    messages_per_robot_hour: float
    stuck: float
    commit_hops: tuple[float, ...]
    empty_killers: int
    consumable_used: float
    drift: tuple[float, ...]
    coverage: float
    idleness_minutes: float
    worst_idleness_minutes: float
    redundancy: float
    wall: float


class PestTracker:
    """Follows the pests and the killers of one run from one sample to the next.

    Attributes:
        appeared: When each pest still infested was first sampled.
        responses: How long each cleared pest stayed infested.
        infested_samples: The number of infested cells at every sample after the warm-up.
        cells_moved: Cells moved by all killers so far.
        committed: Killer samples in which the killer was committed to treating.
        stuck: Committed killer samples in which the killer stood idle, not in a move.
        commit_hops: The killer's beacon value each time it committed, if it had beacons.
        drift: The killers' mean distance from their start, at each hour.
    """

    def __init__(self, starts: dict[RobotId, CellId], warmup: Time) -> None:
        self._starts: dict[RobotId, CellId] = starts
        self._warmup: Time = warmup
        self._positions: dict[RobotId, CellId] = dict(starts)
        self.appeared: dict[CellId, Time] = {}
        self.responses: list[Time] = []
        self.infested_samples: list[int] = []
        self.cells_moved: int = 0
        self.committed: int = 0
        self.stuck: int = 0
        self.commit_hops: list[float] = []
        self.drift: list[float] = []
        self._treating: set[RobotId] = set()

    def sample(self, run: LiveRun, now: Time) -> None:
        """Record the pests and the killers' moves at ``now``."""
        world = run.context.world
        infested: set[CellId] = set()
        for cell, state in world.cells.items():
            if state.infestation > 0.0:
                infested.add(cell)
        for cell in infested:
            self.appeared.setdefault(cell, now)
        for cell in set(self.appeared) - infested:
            self.responses.append(now - self.appeared.pop(cell))
        if now > self._warmup:
            self.infested_samples.append(len(infested))

        for robot_id, before in self._positions.items():
            position = world.robots[robot_id].position
            self.cells_moved += max(abs(position[0] - before[0]), abs(position[1] - before[1]))
            self._positions[robot_id] = position
        self._sample_commitments(run, now)
        if now % TimeUnit.HOUR == 0:
            self.drift.append(self.mean_distance_from_start())

    def _sample_commitments(self, run: LiveRun, now: Time) -> None:
        """Count the killers committed to treating, those standing idle, and new commitments."""
        world = run.context.world
        agents = run.agents()
        for robot_id in self._positions:
            agent = agents[robot_id]
            goal = agent.inspect().current_goal
            if goal is None or not goal.startswith("treat"):
                self._treating.discard(robot_id)
                continue
            self.committed += 1
            robot = world.robots[robot_id]
            if isinstance(robot.current_action, Idle) and robot.move_target is None:
                self.stuck += 1
            if robot_id in self._treating:
                continue
            self._treating.add(robot_id)
            link = agent.beacon_link
            if link is not None:
                position = world.robots[robot_id].position
                beacon = treat_beacon(link.memory, position, now, link.config)
                if beacon is not None:
                    self.commit_hops.append(beacon.value)

    def mean_distance_from_start(self) -> float:
        """The killers' mean straight-line distance in cells from where they started; 0 without."""
        if not self._positions:
            return 0.0
        distances: list[float] = []
        for robot_id, position in self._positions.items():
            distances.append(math.dist(position, self._starts[robot_id]))
        return mean(distances)


class MeasuredPestRun:
    """One run of a ``PestRunSpec``, advanced one decision interval at a time and measured.

    ``measure_pest_response`` advances it to the end; ``record_video`` draws a frame after every
    step as well, so a video shows exactly the run that was measured.

    Attributes:
        run: The run: its arm, its spec and its seed.
        live: The run's simulation.
        tracker: Samples the pests and the killers after every step.
        killers: Each killer's start cell, by id.
    """

    def __init__(self, run: Run[PestRunSpec]) -> None:
        self._started: float = time.perf_counter()
        self.run: Run[PestRunSpec] = run
        self.live: LiveRun = LiveRun.from_config(run.config.into_experiment(run.arm, run.seed))
        self.killers: dict[RobotId, CellId] = {}
        for robot_id, robot in self.live.context.world.robots.items():
            if robot.body is BodyType.KILLER:
                self.killers[robot_id] = robot.position
        self.tracker: PestTracker = PestTracker(
            self.killers, run.config.warmup_hours * TimeUnit.HOUR
        )

    @property
    def step(self) -> Time:
        """How far one ``advance`` moves the run: its decision interval."""
        return self.live.config.decision_interval

    @property
    def steps(self) -> int:
        """How many steps carry the run from its start to its horizon."""
        return self.live.horizon // self.step

    @property
    def finished(self) -> bool:
        """Whether the run has reached its horizon."""
        return self.live.now + self.step > self.live.horizon

    def advance(self) -> None:
        """Advance the run by one decision interval, and sample it."""
        now = self.live.now + self.step
        self.live.advance_to(now)
        self.tracker.sample(self.live, now)

    def response(self) -> PestResponse:
        """What the run measured, once it is ``finished``."""
        tracker = self.tracker
        treated = len(tracker.responses)
        response: float | None = None
        response_p90: float | None = None
        cells_per_treatment: float | None = None
        if treated > 0:
            response = mean(tracker.responses) / TimeUnit.MINUTE
            response_p90 = _percentile_90(tracker.responses) / TimeUnit.MINUTE
            cells_per_treatment = tracker.cells_moved / treated
        world = self.live.context.world
        empty = 0
        used = 0.0
        for robot_id in self.killers:
            consumable = world.robots[robot_id].consumable
            used += 1.0 - consumable
            if consumable <= 0:
                empty += 1
        hours = self.run.config.hours
        metrics = self.live.metrics.latest()
        if metrics is None:
            raise ValueError("the run recorded no metrics sample")
        return PestResponse(
            infested=mean(tracker.infested_samples),
            treated=treated,
            response_minutes=response,
            response_p90=response_p90,
            open=len(tracker.appeared),
            cells_per_treatment=cells_per_treatment,
            messages_per_robot_hour=self.live.context.channel.received / len(world.robots) / hours,
            stuck=tracker.stuck / max(tracker.committed, 1),
            commit_hops=tuple(tracker.commit_hops),
            empty_killers=empty,
            consumable_used=used,
            drift=tuple(tracker.drift),
            coverage=metrics.coverage,
            idleness_minutes=metrics.mean_idleness_ms / TimeUnit.MINUTE,
            worst_idleness_minutes=metrics.worst_idleness_ms / TimeUnit.MINUTE,
            redundancy=metrics.redundancy,
            wall=time.perf_counter() - self._started,
        )


def measure_pest_response(run: Run[PestRunSpec]) -> PestResponse:
    """Make one run of the spec with the run's seed and measure it; a run function."""
    measured = MeasuredPestRun(run)
    while not measured.finished:
        measured.advance()
    return measured.response()


def pest_table_headers(hours: int, first: str = "arm") -> list[str]:
    """The standard table's column names, with one drift column per hour.

    Args:
        hours: How long the runs lasted, for the drift columns.
        first: The name of the label column.
    """
    headers = [first, "infested", "treated", "response", "p90", "open", "cells/treat", "msgs"]
    headers.extend(["stuck", "commit hop", "empty", "coverage", "idle", "redund"])
    for hour in range(1, hours + 1):
        headers.append(f"drift {hour} h")
    headers.append("wall (s)")
    return headers


def pest_table_cells(label: str, results: list[PestResponse], hours: int) -> list[str]:
    """One row of the standard table: the means over an arm's results, by seed."""
    cells = [
        label,
        mean_and_error([result.infested for result in results]),
        mean_and_error([float(result.treated) for result in results]),
        mean_and_error([result.response_minutes for result in results]),
        mean_and_error([result.response_p90 for result in results]),
        f"{mean(result.open for result in results):.1f}",
        mean_and_error([result.cells_per_treatment for result in results]),
        f"{mean(result.messages_per_robot_hour for result in results):.0f}",
        f"{mean(result.stuck for result in results):.2f}",
        _median_commit_hop(results),
        f"{mean(result.empty_killers for result in results):.1f}",
        f"{mean(result.coverage for result in results):.3f}",
        mean_and_error([result.idleness_minutes for result in results]),
        f"{mean(result.redundancy for result in results):.3f}",
    ]
    for hour in range(hours):
        cells.append(f"{mean(result.drift[hour] for result in results):.1f}")
    cells.append(f"{mean(result.wall for result in results):.0f}")
    return cells


def treated_share(results: list[PestResponse]) -> str:
    """The share of the pests that appeared and were treated, as mean ± standard error.

    Every pest is either treated or still open at the end, so the share is
    ``treated / (treated + open)``.
    """
    shares: list[float] = []
    for result in results:
        shares.append(result.treated / max(result.treated + result.open, 1))
    return mean_and_error(shares)


def _median_commit_hop(results: list[PestResponse]) -> str:
    """The median commit hop over every seed's commitments, or ``-`` for an arm without any."""
    hops: list[float] = []
    for result in results:
        hops.extend(result.commit_hops)
    if not hops:
        return "-"
    return f"{median(hops):g}"


def _percentile_90(values: list[Time]) -> float:
    """The 90th percentile of ``values``; the value itself for a single one."""
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=10)[-1]
