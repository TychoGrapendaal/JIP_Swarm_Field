"""Measure how a swarm responds to pests: one run of a ``SwarmScenario`` into a ``PestResponse``.

``measure_pest_response`` is a run function for ``Experiment.run``: it runs a ``SwarmScenario``,
made with ``SwarmScenario.with_hours``, with the run's seed, advancing one decision interval at a
time, and a ``PestTracker`` samples the ground truth and the killers' brains between steps.
``pest_table_headers`` and ``pest_table_cells`` turn the results of an arm into a row of the
standard table, with the mean over the seeds and its standard error (±) where it matters:

- ``infested``: infested cells, averaged over the run after the warm-up. The main outcome: it
  counts both how many pests stay and how long, and has no end-of-run ceiling;
- ``treated``: pests cleared during the run, which only treatment does;
- ``response``: minutes from a pest's first appearance to its clearing, over the cleared pests;
- ``p90``: the 90th percentile of those minutes;
- ``open``: pests still infested at the end;
- ``cells/treat``: cells moved by all killers per treated pest;
- ``msgs``: messages received per robot per hour; a robot sends its beacons in a message of
  their own next to its status, so a hop robot's broadcast counts as two;
- ``items``: the items those messages carried, per robot per hour: a status, every fact of a
  knowledge snapshot and every beacon is one;
- ``statuses``, ``facts``, ``beacons``: those items by kind, the statuses, the facts of the
  knowledge snapshots and the beacons;
- ``stuck``: the share of the time a killer committed to treating stands idle, e.g. at a
  dead end of the hop field;
- ``commit hop``: the median beacon value of a killer when it commits to treating, for arms
  with beacons;
- ``no doses``: the share of the live killer samples in which the killer had no doses left;
- ``shared``: the share of the samples with two or more killers committed to treating in which
  two of them report the same target, a duplicate trip; a hop killer's target is the cell of its
  strongest beacon, not the pest;
- ``co-treat``: the share of the samples with a treatment under way in which two killers treat
  the same cell;
- ``isolated``: the share of the killer samples in which no other robot is within radio range;
- ``no beacon``: the share of the samples of killers with beacons in which the killer heard no
  fresh TREAT beacon;
- ``coverage``: the share of the crop cells some robot sensed by the end;
- ``idle``: minutes since a robot last sensed a crop cell, averaged over the cells and the
  samples after the warm-up: the expected wait of a pest for its first sighting;
- ``redund``: the share of the robots' visits to a cell sensed less than ``fresh_age`` before;
- ``drift 1/2/3 h``: mean distance in cells of the killers from their start after each hour;
- ``wall``: seconds of computation per run.

``pest_isolation_headers`` and ``pest_isolation_cells`` give a second table, on whether killers
lose contact with the swarm and what that costs:

- ``idle isol``: the share of the killer samples in which the killer does not treat and no
  other robot is within radio range;
- ``isol p90``: the 90th percentile of the minutes a killer stayed isolated in one go;
- ``crowded``: the share of the samples of killers that do not treat in which another killer is
  within radio range;
- ``unknown``: pests per sample, after the warm-up, that no robot knows of;
- ``cut off``: pests per sample, after the warm-up, that some robot knows of but no robot in the
  radio network of any robot that knows is a killer; at most what perfect cohesion could save.

The tracker reads the killers' goals and beacons, which the simulation's own metrics may not,
so it lives here, at the composition level, next to ``LiveRun``. Coverage, idleness and
redundancy come from the run's ``MetricsCollector``, sampled once a minute.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import mean, median, quantiles

import numpy as np

from swarm_field.contracts import (
    BeaconKind,
    BodyType,
    CellId,
    Idle,
    MessageKind,
    RobotId,
    Time,
    TimeUnit,
)
from swarm_field.experiments.batch import Run, mean_and_error
from swarm_field.experiments.run import LiveRun
from swarm_field.experiments.scenario import SwarmScenario
from swarm_field.swarm import ComposedAgent, treat_beacon
from swarm_field.swarm.hop import fresh_beacons


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
        items_per_robot_hour: The items those messages carried, per robot per hour: a status,
            every fact of a knowledge snapshot and every beacon is one (``message_items``).
        status_items_per_robot_hour: Of those, the statuses.
        fact_items_per_robot_hour: Of those, the facts of the knowledge snapshots.
        beacon_items_per_robot_hour: Of those, the beacons.
        stuck: The share of committed killer samples in which the killer stood idle.
        commit_hops: The killer's beacon value each time it committed to treating.
        no_doses: The share of the live killer samples in which the killer had no doses left.
        shared_target: The share of the samples with two or more killers committed to treating
            in which two report the same target, or ``None`` if there were none.
        co_treating: The share of the samples with a treatment under way in which two killers
            treat the same cell, or ``None`` if nothing was treated.
        isolated: The share of the killer samples with no other live robot within radio range.
        no_fresh_beacon: The share of the samples of killers with beacons in which the killer
            heard no fresh TREAT beacon, or ``None`` without beacons.
        idle_isolated: The share of the killer samples in which the killer did not treat and no
            other live robot was within radio range.
        isolation_spells: The minutes each killer stayed isolated in one go.
        crowded: The share of the samples of killers that did not treat in which another killer
            was within radio range, or ``None`` if every killer always treated.
        unknown_infested: Pests per sample after the warm-up that no robot knew of.
        cut_off_infested: Pests per sample after the warm-up that some robot knew of, but no
            robot in the radio network of any that knew was a killer.
        doses_used: Doses used by all killers together, one per treatment job, also those
            used before a killer restocked.
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
    items_per_robot_hour: float
    status_items_per_robot_hour: float
    fact_items_per_robot_hour: float
    beacon_items_per_robot_hour: float
    stuck: float
    commit_hops: tuple[float, ...]
    no_doses: float
    shared_target: float | None
    co_treating: float | None
    isolated: float
    no_fresh_beacon: float | None
    idle_isolated: float
    isolation_spells: tuple[float, ...]
    crowded: float | None
    unknown_infested: float
    cut_off_infested: float
    doses_used: int
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
        several_committed: Samples with two or more killers committed to treating.
        shared_target: Of those, the samples in which two of them report the same target.
        treating: Samples with a treatment under way.
        co_treating: Of those, the samples in which two killers treat the same cell.
        killer_samples: Samples of a live killer, one per killer and sample.
        no_doses: Of those, the samples in which the killer had no doses left.
        isolated: Of those, the samples with no other live robot within radio range.
        beacon_samples: Samples of a live killer with beacons.
        no_fresh_beacon: Of those, the samples in which it heard no fresh TREAT beacon.
        idle_isolated: Killer samples in which the killer did not treat and was isolated.
        spells: The isolation spells that ended, in simulation time.
        idle_samples: Samples of a live killer that did not treat.
        crowded: Of those, the samples with another killer within radio range.
        reach_samples: Samples after the warm-up, for ``unknown`` and ``cut_off``.
        unknown: Pests no robot knew of, summed over those samples.
        cut_off: Pests known only in radio networks without a killer, summed likewise.
    """

    def __init__(self, starts: dict[RobotId, CellId], warmup: Time, radio_cells: float) -> None:
        """Follow the killers starting on ``starts``.

        Args:
            starts: Each killer's start cell, by id.
            warmup: How long at the start is left out of ``infested_samples``.
            radio_cells: The radio range in cells, for ``isolated``.
        """
        self._starts: dict[RobotId, CellId] = starts
        self._warmup: Time = warmup
        self._radio_cells: float = radio_cells
        self._positions: dict[RobotId, CellId] = dict(starts)
        self.appeared: dict[CellId, Time] = {}
        self.responses: list[Time] = []
        self.infested_samples: list[int] = []
        self.cells_moved: int = 0
        self.committed: int = 0
        self.stuck: int = 0
        self.commit_hops: list[float] = []
        self.drift: list[float] = []
        self.several_committed: int = 0
        self.shared_target: int = 0
        self.treating: int = 0
        self.co_treating: int = 0
        self.killer_samples: int = 0
        self.no_doses: int = 0
        self.isolated: int = 0
        self.beacon_samples: int = 0
        self.no_fresh_beacon: int = 0
        self.idle_isolated: int = 0
        self.spells: list[Time] = []
        self.idle_samples: int = 0
        self.crowded: int = 0
        self.reach_samples: int = 0
        self.unknown: int = 0
        self.cut_off: int = 0
        self._isolated_since: dict[RobotId, Time] = {}
        self._last_sample: Time = 0
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
        self._sample_treatments(run)
        self._sample_isolation(run, now)
        if now > self._warmup:
            self._sample_reach(run, now, infested)
        self._last_sample = now
        if now % TimeUnit.HOUR == 0:
            self.drift.append(self.mean_distance_from_start())

    def _sample_commitments(self, run: LiveRun, now: Time) -> None:
        """Count the killers committed to treating, those standing idle, and new commitments."""
        world = run.context.world
        agents = run.agents()
        targets: list[CellId | None] = []
        for robot_id in self._positions:
            agent = agents[robot_id]
            inspection = agent.inspect()
            goal = inspection.current_goal
            if goal is None or not goal.startswith("treat"):
                self._treating.discard(robot_id)
                continue
            targets.append(inspection.current_target)
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
        if len(targets) >= 2:
            self.several_committed += 1
            if _has_duplicate(targets):
                self.shared_target += 1

    def _sample_treatments(self, run: LiveRun) -> None:
        """Count the samples with a treatment under way, and those with two killers on a cell."""
        world = run.context.world
        cells: list[CellId | None] = []
        for robot_id in self._positions:
            cell = world.robots[robot_id].treating_cell
            if cell is not None:
                cells.append(cell)
        if not cells:
            return
        self.treating += 1
        if _has_duplicate(cells):
            self.co_treating += 1

    def _sample_isolation(self, run: LiveRun, now: Time) -> None:
        """Count the live killers with no other live robot in radio range, those without doses
        or beacons, and those idle and isolated or crowded; follow every killer's isolation
        spells."""
        world = run.context.world
        agents = run.agents()
        live = world.live_robots()
        for robot_id in self._positions:
            killer = world.robots[robot_id]
            if killer.failed:
                self._end_spell(robot_id, now)
                continue
            self.killer_samples += 1
            if killer.doses_held <= 0:
                self.no_doses += 1
            neighbours = 0
            killers_near = 0
            for other in live:
                if other == robot_id:
                    continue
                if math.dist(killer.position, world.robots[other].position) <= self._radio_cells:
                    neighbours += 1
                    if other in self._positions:
                        killers_near += 1
            goal = agents[robot_id].inspect().current_goal
            treating = goal is not None and goal.startswith("treat")
            if not treating:
                self.idle_samples += 1
                if killers_near > 0:
                    self.crowded += 1
            if neighbours == 0:
                self.isolated += 1
                self._isolated_since.setdefault(robot_id, now)
                if not treating:
                    self.idle_isolated += 1
            else:
                self._end_spell(robot_id, now)
            link = agents[robot_id].beacon_link
            if link is None:
                continue
            self.beacon_samples += 1
            if not fresh_beacons(link.memory, BeaconKind.TREAT, now, link.config):
                self.no_fresh_beacon += 1

    def _end_spell(self, robot_id: RobotId, now: Time) -> None:
        """End the robot's isolation spell at ``now``, if it is isolated."""
        since = self._isolated_since.pop(robot_id, None)
        if since is not None:
            self.spells.append(now - since)

    def isolation_spells(self) -> list[Time]:
        """Every isolation spell, the spells still under way ended at the last sample."""
        spells = list(self.spells)
        for since in self._isolated_since.values():
            spells.append(self._last_sample - since)
        return spells

    def _sample_reach(self, run: LiveRun, now: Time, infested: set[CellId]) -> None:
        """Count the pests no robot knows of, and those known only away from every killer."""
        if not infested:
            self.reach_samples += 1
            return
        world = run.context.world
        agents = run.agents()
        positions: dict[RobotId, CellId] = {}
        for robot_id in world.live_robots():
            positions[robot_id] = world.robots[robot_id].position
        networks = radio_networks(positions, self._radio_cells)
        with_killer: set[int] = set()
        for robot_id in self._positions:
            if robot_id in networks:
                with_killer.add(networks[robot_id])
        known_in: dict[CellId, set[int]] = {}
        for robot_id, position in positions.items():
            for cell in _pests_known_to(agents[robot_id], position, now, infested):
                known_in.setdefault(cell, set()).add(networks[robot_id])
        self.reach_samples += 1
        for cell in infested:
            networks_knowing = known_in.get(cell)
            if networks_knowing is None:
                self.unknown += 1
            elif not networks_knowing & with_killer:
                self.cut_off += 1

    def mean_distance_from_start(self) -> float:
        """The killers' mean straight-line distance in cells from where they started; 0 without."""
        if not self._positions:
            return 0.0
        distances: list[float] = []
        for robot_id, position in self._positions.items():
            distances.append(math.dist(position, self._starts[robot_id]))
        return mean(distances)


class MeasuredPestRun:
    """One run of a ``SwarmScenario``, advanced one decision interval at a time and measured.

    ``measure_pest_response`` advances it to the end; ``record_video`` draws a frame after every
    step as well, so a video shows exactly the run that was measured.

    Attributes:
        run: The run: its arm, its scenario and its seed.
        live: The run's simulation.
        tracker: Samples the pests and the killers after every step.
        killers: Each killer's start cell, by id.
    """

    def __init__(self, run: Run[SwarmScenario]) -> None:
        self._started: float = time.perf_counter()
        self.run: Run[SwarmScenario] = run
        self.live: LiveRun = LiveRun.from_config(
            run.config.with_seed(run.seed).into_experiment(run.arm)
        )
        self.killers: dict[RobotId, CellId] = {}
        for robot_id, robot in self.live.context.world.robots.items():
            if robot.body is BodyType.KILLER:
                self.killers[robot_id] = robot.position
        scenario = run.config
        self.tracker: PestTracker = PestTracker(
            self.killers,
            scenario.metrics.measure_from,
            scenario.robots.radio_range_m / scenario.map.cell_size,
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
        """Advance the run to the end of its current decision interval, and sample it.

        A video may have advanced ``live`` part of the way already, to draw a frame in between.
        """
        now = (self.live.now // self.step + 1) * self.step
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
        channel = self.live.context.channel
        doses_used = 0
        for robot_id in self.killers:
            doses_used += world.robots[robot_id].doses_used
        hours = self.run.config.until / TimeUnit.HOUR
        robot_hours = len(world.robots) * hours
        items = channel.items_received
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
            messages_per_robot_hour=channel.received / robot_hours,
            items_per_robot_hour=items.total() / robot_hours,
            status_items_per_robot_hour=items[MessageKind.STATUS] / robot_hours,
            fact_items_per_robot_hour=items[MessageKind.KNOWLEDGE] / robot_hours,
            beacon_items_per_robot_hour=items[MessageKind.BEACON] / robot_hours,
            stuck=tracker.stuck / max(tracker.committed, 1),
            commit_hops=tuple(tracker.commit_hops),
            no_doses=tracker.no_doses / max(tracker.killer_samples, 1),
            shared_target=_share(tracker.shared_target, tracker.several_committed),
            co_treating=_share(tracker.co_treating, tracker.treating),
            isolated=tracker.isolated / max(tracker.killer_samples, 1),
            no_fresh_beacon=_share(tracker.no_fresh_beacon, tracker.beacon_samples),
            idle_isolated=tracker.idle_isolated / max(tracker.killer_samples, 1),
            isolation_spells=_minutes(tracker.isolation_spells()),
            crowded=_share(tracker.crowded, tracker.idle_samples),
            unknown_infested=tracker.unknown / max(tracker.reach_samples, 1),
            cut_off_infested=tracker.cut_off / max(tracker.reach_samples, 1),
            doses_used=doses_used,
            drift=tuple(tracker.drift),
            coverage=metrics.coverage,
            idleness_minutes=metrics.mean_idleness_ms / TimeUnit.MINUTE,
            worst_idleness_minutes=metrics.worst_idleness_ms / TimeUnit.MINUTE,
            redundancy=metrics.redundancy,
            wall=time.perf_counter() - self._started,
        )


def measure_pest_response(run: Run[SwarmScenario]) -> PestResponse:
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
    headers.extend(["items", "statuses", "facts", "beacons"])
    headers.extend(
        ["stuck", "commit hop", "no doses", "shared", "co-treat", "isolated", "no beacon"]
    )
    headers.extend(["coverage", "idle", "redund"])
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
        f"{mean(result.items_per_robot_hour for result in results):.0f}",
        f"{mean(result.status_items_per_robot_hour for result in results):.0f}",
        f"{mean(result.fact_items_per_robot_hour for result in results):.0f}",
        f"{mean(result.beacon_items_per_robot_hour for result in results):.0f}",
        f"{mean(result.stuck for result in results):.2f}",
        _median_commit_hop(results),
        f"{mean(result.no_doses for result in results):.2f}",
        _mean_share([result.shared_target for result in results]),
        _mean_share([result.co_treating for result in results]),
        f"{mean(result.isolated for result in results):.2f}",
        _mean_share([result.no_fresh_beacon for result in results]),
        f"{mean(result.coverage for result in results):.3f}",
        mean_and_error([result.idleness_minutes for result in results]),
        f"{mean(result.redundancy for result in results):.3f}",
    ]
    for hour in range(hours):
        cells.append(f"{mean(result.drift[hour] for result in results):.1f}")
    cells.append(f"{mean(result.wall for result in results):.0f}")
    return cells


def pest_isolation_headers(first: str = "arm") -> list[str]:
    """The isolation table's column names."""
    return [first, "infested", "isolated", "idle isol", "isol p90", "crowded", "unknown", "cut off"]


def pest_isolation_cells(label: str, results: list[PestResponse]) -> list[str]:
    """One row of the isolation table: the means over an arm's results, by seed."""
    spells: list[float] = []
    for result in results:
        spells.extend(result.isolation_spells)
    p90 = "-"
    if spells:
        p90 = f"{_percentile_90(spells):.1f}"
    return [
        label,
        mean_and_error([result.infested for result in results]),
        f"{mean(result.isolated for result in results):.2f}",
        f"{mean(result.idle_isolated for result in results):.2f}",
        p90,
        _mean_share([result.crowded for result in results]),
        mean_and_error([result.unknown_infested for result in results]),
        mean_and_error([result.cut_off_infested for result in results]),
    ]


def radio_networks(positions: Mapping[RobotId, CellId], radio_cells: float) -> dict[RobotId, int]:
    """Which radio network each robot is in: robots within ``radio_cells`` of each other are
    joined, and so are robots joined through others. Networks are numbered from 0."""
    robots = list(positions)
    if not robots:
        return {}
    cells = np.array([positions[robot] for robot in robots], dtype=np.float64)
    gaps = cells[:, np.newaxis, :] - cells[np.newaxis, :, :]
    linked = np.hypot(gaps[..., 0], gaps[..., 1]) <= radio_cells
    network = np.full(len(robots), -1)
    label = 0
    for start in range(len(robots)):
        if network[start] >= 0:
            continue
        reached = np.zeros(len(robots), dtype=np.bool_)
        reached[start] = True
        frontier = reached.copy()
        while frontier.any():
            frontier = linked[frontier].any(axis=0) & ~reached
            reached |= frontier
        network[reached] = label
        label += 1
    found: dict[RobotId, int] = {}
    for index, robot in enumerate(robots):
        found[robot] = int(network[index])
    return found


def _pests_known_to(
    agent: ComposedAgent, position: CellId, now: Time, infested: set[CellId]
) -> list[CellId]:
    """The infested cells the robot knows of: the pest it is the hop source of, for a robot with
    beacons, which is all a hop killer can follow; the pests in its memory otherwise."""
    link = agent.beacon_link
    if link is not None:
        beacon = treat_beacon(link.memory, position, now, link.config)
        if beacon is not None and beacon.value == 0.0 and beacon.cell in infested:
            return [beacon.cell]
        return []
    known: list[CellId] = []
    for cell in infested:
        fact = agent.memory.infestation_at(cell)
        if fact is not None and fact.severity > 0:
            known.append(cell)
    return known


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


def _share(count: int, total: int) -> float | None:
    """``count`` out of ``total``, or ``None`` if there was nothing to count."""
    if total == 0:
        return None
    return count / total


def _mean_share(shares: list[float | None]) -> str:
    """The mean of the seeds' shares that exist, or ``-`` if none does."""
    present: list[float] = []
    for share in shares:
        if share is not None:
            present.append(share)
    if not present:
        return "-"
    return f"{mean(present):.2f}"


def _has_duplicate(cells: list[CellId | None]) -> bool:
    """Whether some cell occurs twice in ``cells``, ``None`` left out."""
    seen: set[CellId] = set()
    for cell in cells:
        if cell is None:
            continue
        if cell in seen:
            return True
        seen.add(cell)
    return False


def _minutes(times: list[Time]) -> tuple[float, ...]:
    """``times`` in minutes."""
    minutes: list[float] = []
    for value in times:
        minutes.append(value / TimeUnit.MINUTE)
    return tuple(minutes)


def _percentile_90(values: list[Time] | list[float]) -> float:
    """The 90th percentile of ``values``; the value itself for a single one."""
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=10)[-1]
