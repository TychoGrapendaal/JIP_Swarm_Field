"""Benchmark pest response: the arms of the relay scenario, ``RELAYED_KILLERS``.

Every arm is a ``PestScenario`` run by ``swarm_field.runner``, repeated with several seeds. The
run is advanced one decision interval at a time and measured on the ground truth between
steps. A row reports the mean over the seeds:

- ``pests``: infested cells, averaged over the run;
- ``treated``: pests cleared during the run, which only treatment does;
- ``response``: minutes from a pest's first appearance to its clearing, over the cleared pests;
- ``open``: pests still infested at the end;
- ``cells/treat``: cells moved by all killers per treated pest;
- ``msgs``: messages received per robot per hour;
- ``stuck``: the share of the time a killer committed to treating stands idle, e.g. at a
  dead end of the hop field;
- ``commit hop``: the median beacon value of a killer when it commits to treating, for arms
  with beacons;
- ``drift 1/2/3 h``: mean distance in cells of the killers from their start after each hour;
- ``wall``: seconds of computation per run.

Runs are spread over processes. Not part of ``pytest``; run it with:

    uv run python -m tests.benchmark_pests --hours 3 --seeds 10
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from statistics import mean, median

from swarm_field.contracts import BodyType, CellId, Idle, RobotId, Time, TimeUnit
from swarm_field.experiments import RELAYED_KILLERS, PestScenario
from swarm_field.experiments.run import LiveRun
from swarm_field.swarm import BeaconConfig, KnowledgeSharing, treat_beacon


@dataclass(frozen=True)
class Arm:
    """One way of responding to pests: a row's label and its scenario."""

    name: str
    scenario: PestScenario


def arms() -> Iterator[Arm]:
    """The seeded arm, the oracle, which hears every pest at once, and the hop arm."""
    yield Arm("seeded", RELAYED_KILLERS)
    oracle = replace(
        RELAYED_KILLERS, pest_source_range_m=math.inf, sharing=KnowledgeSharing.NOTHING
    )
    yield Arm("oracle", oracle)
    hop = replace(
        RELAYED_KILLERS,
        hop_treat=True,
        beacons=BeaconConfig(),
        sharing=KnowledgeSharing.NOTHING,
    )
    yield Arm("hop", hop)


@dataclass(frozen=True)
class Job:
    """One run to make: an arm for ``hours``, with ``seed``."""

    arm: Arm
    hours: int
    seed: int


@dataclass(frozen=True)
class Result:
    """What one run measured; the attributes are the table's columns."""

    pests: float
    treated: int
    response_minutes: float | None
    open: int
    cells_per_treatment: float | None
    messages_per_robot_hour: float
    stuck: float
    commit_hops: tuple[float, ...]
    drift: tuple[float, ...]
    wall: float


class PestTracker:
    """Follows the pests and the killers of one run from one sample to the next.

    Attributes:
        appeared: When each pest still infested was first sampled.
        responses: How long each cleared pest stayed infested.
        infested_samples: The number of infested cells at every sample.
        cells_moved: Cells moved by all killers so far.
        committed: Killer samples in which the killer was committed to treating.
        stuck: Committed killer samples in which the killer stood idle, not in a move.
        commit_hops: The killer's beacon value each time it committed, if it had beacons.
        drift: The killers' mean distance from their start, at each hour.
    """

    def __init__(self, starts: dict[RobotId, CellId]) -> None:
        self._starts: dict[RobotId, CellId] = starts
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
            if agent.beacons is not None:
                position = world.robots[robot_id].position
                beacon = treat_beacon(agent.memory, robot_id, position, now, agent.beacons)
                if beacon is not None:
                    self.commit_hops.append(beacon.value)

    def mean_distance_from_start(self) -> float:
        """The killers' mean straight-line distance in cells from where they started."""
        distances: list[float] = []
        for robot_id, position in self._positions.items():
            distances.append(math.dist(position, self._starts[robot_id]))
        return mean(distances)


def run(job: Job) -> Result:
    """Make one run and measure it."""
    scenario = replace(
        job.arm.scenario,
        until=job.hours * TimeUnit.HOUR,
        sample_period=TimeUnit.MINUTE,
        seed=job.seed,
    )
    started = time.perf_counter()
    live = LiveRun.from_config(scenario.into_experiment(job.arm.name))
    starts: dict[RobotId, CellId] = {}
    for robot_id, robot in live.context.world.robots.items():
        if robot.body is BodyType.KILLER:
            starts[robot_id] = robot.position
    tracker = PestTracker(starts)
    step = scenario.decision_interval
    for now in range(step, scenario.until + 1, step):
        live.advance_to(now)
        tracker.sample(live, now)

    treated = len(tracker.responses)
    response: float | None = None
    cells_per_treatment: float | None = None
    if treated > 0:
        response = mean(tracker.responses) / TimeUnit.MINUTE
        cells_per_treatment = tracker.cells_moved / treated
    robots = len(live.context.world.robots)
    return Result(
        pests=mean(tracker.infested_samples),
        treated=treated,
        response_minutes=response,
        open=len(tracker.appeared),
        cells_per_treatment=cells_per_treatment,
        messages_per_robot_hour=live.context.channel.received / robots / job.hours,
        stuck=tracker.stuck / max(tracker.committed, 1),
        commit_hops=tuple(tracker.commit_hops),
        drift=tuple(tracker.drift),
        wall=time.perf_counter() - started,
    )


def table_row(name: str, results: list[Result], hours: int) -> str:
    """One Markdown table row: the means over the seeds' results."""
    cells = [
        name,
        f"{mean(result.pests for result in results):.2f}",
        f"{mean(result.treated for result in results):.1f}",
        _mean_or_dash([result.response_minutes for result in results]),
        f"{mean(result.open for result in results):.1f}",
        _mean_or_dash([result.cells_per_treatment for result in results]),
        f"{mean(result.messages_per_robot_hour for result in results):.0f}",
        f"{mean(result.stuck for result in results):.2f}",
        _median_or_dash(results),
    ]
    for hour in range(hours):
        cells.append(f"{mean(result.drift[hour] for result in results):.1f}")
    cells.append(f"{mean(result.wall for result in results):.0f}")
    return "| " + " | ".join(cells) + " |"


def _median_or_dash(results: list[Result]) -> str:
    """The median commit hop over every seed's commitments, or ``-`` for an arm without any."""
    hops: list[float] = []
    for result in results:
        hops.extend(result.commit_hops)
    if not hops:
        return "-"
    return f"{median(hops):g}"


def _mean_or_dash(values: list[float | None]) -> str:
    """The mean of ``values``, or ``-`` if any run has none."""
    present: list[float] = []
    for value in values:
        if value is None:
            return "-"
        present.append(value)
    return f"{mean(present):.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=int, default=3, help="simulated hours per run")
    parser.add_argument("--seeds", type=int, default=10, help="runs per arm")
    parser.add_argument("--jobs", type=int, default=8, help="processes")
    args = parser.parse_args()

    section = list(arms())
    jobs: list[Job] = []
    for arm in section:
        jobs.extend(Job(arm, args.hours, seed) for seed in range(args.seeds))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, jobs))

    scenario = RELAYED_KILLERS
    print(
        f"{scenario.scouts} relays, {scenario.killers} killers, "
        f"{scenario.radio_range_m:g} m radio, {args.hours} h, mean over {args.seeds} seeds\n"
    )
    headers = ["arm", "pests", "treated", "response", "open", "cells/treat", "msgs"]
    headers.extend(["stuck", "commit hop"])
    for hour in range(1, args.hours + 1):
        headers.append(f"drift {hour} h")
    headers.append("wall (s)")
    print("| " + " | ".join(headers) + " |")
    print("|" + " --- |" * len(headers))
    for index, arm in enumerate(section):
        seeds = results[index * args.seeds : (index + 1) * args.seeds]
        print(table_row(arm.name, seeds, args.hours))


if __name__ == "__main__":
    main()
