"""Benchmark pest response: the arms of the relay scenario, ``RELAYED_KILLERS``.

Every arm is a ``PestScenario`` run by ``swarm_field.runner``, repeated with several seeds. The
run is advanced one decision interval at a time and measured on the ground truth between
steps. A row reports the mean over the seeds, with its standard error (±) where it matters:

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
- ``drift 1/2/3 h``: mean distance in cells of the killers from their start after each hour;
- ``wall``: seconds of computation per run.

Runs are spread over processes with ``swarm_field.experiments.Experiment``, which writes every
run's result and the table to a new folder in ``output/``. Not part of ``pytest``; run it with:

    uv run python -m tests.benchmark_pests --hours 3 --seeds 10 --pests-per-hour 15
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, replace
from statistics import mean, median, quantiles

from swarm_field.contracts import BodyType, CellId, Idle, RobotId, Time, TimeUnit
from swarm_field.experiments import (
    RELAYED_KILLERS,
    Experiment,
    PestScenario,
    Run,
    by_arm,
    markdown_table,
    mean_and_error,
    runs_for,
)
from swarm_field.experiments.run import LiveRun
from swarm_field.swarm import BeaconConfig, EscapeConfig, KnowledgeSharing, treat_beacon

METRIC_BEACONS = BeaconConfig(
    hop_cells=RELAYED_KILLERS.radio_range_m / RELAYED_KILLERS.cell_size,
    max_value=math.log(0.05) / math.log(0.5),
)
"""Metric hops of one radio range, capped where a beacon's strength falls below 0.05."""


def arms() -> dict[str, PestScenario]:
    """The seeded arm, the oracle, which hears every pest at once, and the hop arms, by name.

    ``hop repel`` has metric hops, the cap, and killers repelling each other. ``hop tuned`` is
    the best hop configuration so far: ``hop repel`` with pulls weighted by the square root of
    the distance, and the escape.
    """
    oracle = replace(
        RELAYED_KILLERS, pest_source_range_m=math.inf, sharing=KnowledgeSharing.NOTHING
    )
    hop = replace(
        RELAYED_KILLERS,
        hop_treat=True,
        beacons=BeaconConfig(),
        sharing=KnowledgeSharing.NOTHING,
    )
    repel = replace(METRIC_BEACONS, killer_repulsion=2.0)
    tuned = replace(repel, distance_power=0.5, escape=EscapeConfig())
    return {
        "seeded": RELAYED_KILLERS,
        "oracle": oracle,
        "hop": hop,
        "hop metric": replace(hop, beacons=METRIC_BEACONS),
        "hop repel": replace(hop, beacons=repel),
        "hop tuned": replace(hop, beacons=tuned),
    }


@dataclass(frozen=True)
class Measurement:
    """How to run and measure one arm: its scenario for ``hours`` at ``pests_per_hour``.

    ``infested`` is averaged after ``warmup_hours``.
    """

    scenario: PestScenario
    hours: int
    pests_per_hour: float = 15.0
    warmup_hours: int = 1

    def cost(self) -> float:
        """A rough relative cost: the field's cells, five times for the seeded arm's gossip."""
        cost = float(self.scenario.size**2 * self.hours)
        if not self.scenario.hop_treat:
            cost *= 5
        return cost


@dataclass(frozen=True)
class Result:
    """What one run measured; the attributes are the table's columns."""

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
    drift: tuple[float, ...]
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
        """The killers' mean straight-line distance in cells from where they started."""
        distances: list[float] = []
        for robot_id, position in self._positions.items():
            distances.append(math.dist(position, self._starts[robot_id]))
        return mean(distances)


def measure(run: Run[Measurement]) -> Result:
    """Make one run and measure it."""
    job = run.config
    scenario = replace(
        job.scenario,
        until=job.hours * TimeUnit.HOUR,
        sample_period=TimeUnit.MINUTE,
        seed=run.seed,
        pests_per_hour=job.pests_per_hour,
    )
    started = time.perf_counter()
    live = LiveRun.from_config(scenario.into_experiment(run.arm))
    starts: dict[RobotId, CellId] = {}
    for robot_id, robot in live.context.world.robots.items():
        if robot.body is BodyType.KILLER:
            starts[robot_id] = robot.position
    tracker = PestTracker(starts, job.warmup_hours * TimeUnit.HOUR)
    step = scenario.decision_interval
    for now in range(step, scenario.until + 1, step):
        live.advance_to(now)
        tracker.sample(live, now)

    treated = len(tracker.responses)
    response: float | None = None
    response_p90: float | None = None
    cells_per_treatment: float | None = None
    if treated > 0:
        response = mean(tracker.responses) / TimeUnit.MINUTE
        response_p90 = _percentile_90(tracker.responses) / TimeUnit.MINUTE
        cells_per_treatment = tracker.cells_moved / treated
    world = live.context.world
    empty = 0
    for robot_id in starts:
        if world.robots[robot_id].consumable <= 0:
            empty += 1
    return Result(
        infested=mean(tracker.infested_samples),
        treated=treated,
        response_minutes=response,
        response_p90=response_p90,
        open=len(tracker.appeared),
        cells_per_treatment=cells_per_treatment,
        messages_per_robot_hour=live.context.channel.received / len(world.robots) / job.hours,
        stuck=tracker.stuck / max(tracker.committed, 1),
        commit_hops=tuple(tracker.commit_hops),
        empty_killers=empty,
        drift=tuple(tracker.drift),
        wall=time.perf_counter() - started,
    )


def _percentile_90(values: list[Time]) -> float:
    """The 90th percentile of ``values``; the value itself for a single one."""
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=10)[-1]


def table_headers(hours: int) -> list[str]:
    """The table's column names, with one drift column per hour."""
    headers = ["arm", "infested", "treated", "response", "p90", "open", "cells/treat", "msgs"]
    headers.extend(["stuck", "commit hop", "empty"])
    for hour in range(1, hours + 1):
        headers.append(f"drift {hour} h")
    headers.append("wall (s)")
    return headers


def table_cells(name: str, results: list[Result], hours: int) -> list[str]:
    """One table row: the means over the seeds' results, under ``table_headers``."""
    cells = [
        name,
        mean_and_error([result.infested for result in results]),
        mean_and_error([float(result.treated) for result in results]),
        _mean_or_dash([result.response_minutes for result in results]),
        _mean_or_dash([result.response_p90 for result in results]),
        f"{mean(result.open for result in results):.1f}",
        _mean_or_dash([result.cells_per_treatment for result in results]),
        f"{mean(result.messages_per_robot_hour for result in results):.0f}",
        f"{mean(result.stuck for result in results):.2f}",
        _median_or_dash(results),
        f"{mean(result.empty_killers for result in results):.1f}",
    ]
    for hour in range(hours):
        cells.append(f"{mean(result.drift[hour] for result in results):.1f}")
    cells.append(f"{mean(result.wall for result in results):.0f}")
    return cells


def _median_or_dash(results: list[Result]) -> str:
    """The median commit hop over every seed's commitments, or ``-`` for an arm without any."""
    hops: list[float] = []
    for result in results:
        hops.extend(result.commit_hops)
    if not hops:
        return "-"
    return f"{median(hops):g}"


def _mean_or_dash(values: list[float | None]) -> str:
    """The mean of ``values`` with its standard error, or ``-`` if any run has none."""
    present: list[float] = []
    for value in values:
        if value is None:
            return "-"
        present.append(value)
    return mean_and_error(present)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=int, default=3, help="simulated hours per run")
    parser.add_argument("--seeds", type=int, default=10, help="runs per arm")
    parser.add_argument("--jobs", type=int, default=8, help="processes")
    parser.add_argument("--pests-per-hour", type=float, default=15.0, help="pest spawn rate")
    parser.add_argument("--warmup-hours", type=int, default=1, help="left out of infested")
    args = parser.parse_args()

    measurements: dict[str, Measurement] = {}
    for name, scenario in arms().items():
        measurements[name] = Measurement(
            scenario, args.hours, args.pests_per_hour, args.warmup_hours
        )
    experiment = Experiment("pest_benchmark")
    runs = runs_for(measurements, range(args.seeds), Measurement.cost)
    results = experiment.run(runs, measure, args.jobs)

    scenario = RELAYED_KILLERS
    title = (
        f"{scenario.scouts} relays, {scenario.killers} killers, "
        f"{scenario.radio_range_m:g} m radio, {args.pests_per_hour:g} pests per hour, "
        f"{args.hours} h (warm-up {args.warmup_hours} h), mean over {args.seeds} seeds"
    )
    rows: list[list[str]] = []
    for name, arm_results in by_arm(results).items():
        rows.append(table_cells(name, arm_results, args.hours))
    text = f"{title}\n\n{markdown_table(table_headers(args.hours), rows)}"
    print(text)
    experiment.write_text("tables.md", text)


if __name__ == "__main__":
    main()
