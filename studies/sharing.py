"""Study what sharing less of their memory costs the swarm, against what it saves in run time.

Question: in the pest experiment (``PestScenario``), how much faster is a run, and how much
fewer facts are sent, when robots broadcast their memory less often than at every decision
(``SharingStrategy``)? And how much does that change what the robots know, how fast news spreads,
and how well the swarm does? Every strategy is run on the same seeds as the exact one, ``full``;
a difference counts as acceptable while it stays within the spread of ``full`` over the seeds.

Besides the run's metrics, the study probes the robots every decision interval for how far
their knowledge drifts apart:

- known pests: per robot, the share of the pests in the field (ground truth) it believes in.
- last-seen gap: per robot and cell, how much older the robot's last sighting of the cell is
  than the newest sighting any robot has; a cell the robot never heard of counts from the start.
- spread time: per pest, from the first robot knowing of it until half (and 90 %) of the robots
  do. Pests treated before they got that far are left out.

Every run writes its metrics series and one row per run to ``--out`` (``runs/sharing`` by
default). The printed tables are means over the seeds. Runs go in parallel over ``--workers``
processes, so wall-clock times are comparable between strategies, not with a lone run. Run
from the repository root with:

    uv run python -m studies.sharing --swarms 10x32 20x32 40x64 --seeds 5 --hours 1
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import numpy as np

from swarm_field.contracts import CellId, Time, TimeUnit
from swarm_field.experiments import PestScenario
from swarm_field.runner import ExperimentConfig, RunSummary, build_run
from swarm_field.simulation import SimulationLoop, WorldState
from swarm_field.swarm import ComposedAgent, SharingConfig, SharingStrategy


@dataclass(frozen=True)
class Condition:
    """One sharing strategy the study runs, by name.

    Attributes:
        label: The name of the condition in the tables.
        sharing: How every robot shares.
    """

    label: str
    sharing: SharingConfig


CONDITIONS: tuple[Condition, ...] = (
    Condition("full", SharingConfig(SharingStrategy.FULL)),
    Condition("periodic-4", SharingConfig(SharingStrategy.PERIODIC, full_every=4)),
    Condition("periodic-20", SharingConfig(SharingStrategy.PERIODIC, full_every=20)),
)
"""``full`` is exact. ``periodic-k`` is A2 of #76: the whole memory every k-th decision (every
minute, and every five), nothing in between."""


@dataclass(frozen=True)
class Swarm:
    """The size of one pest run.

    Attributes:
        robots: How many robots, three killers per ten.
        size: The field's side in cells.
    """

    robots: int
    size: int

    @classmethod
    def from_text(cls, text: str) -> Swarm:
        """Read a swarm written as ``ROBOTSxSIZE``, e.g. ``20x32``."""
        robots, size = text.split("x")
        return cls(int(robots), int(size))

    def __str__(self) -> str:
        return f"{self.robots}x{self.size}"


@dataclass(frozen=True)
class RunSpec:
    """One run of the study.

    Attributes:
        condition: The sharing strategy.
        swarm: The swarm and field size.
        radio: The robots' radio range in metres.
        seed: The scenario seed.
        hours: The simulated time.
    """

    condition: Condition
    swarm: Swarm
    radio: float
    seed: int
    hours: float

    def into_experiment(self) -> ExperimentConfig:
        killers = self.swarm.robots * 3 // 10
        scenario = PestScenario(
            size=self.swarm.size,
            scouts=self.swarm.robots - killers,
            killers=killers,
            radio_range_m=self.radio,
            until=int(self.hours * TimeUnit.HOUR),
            seed=self.seed,
        )
        label = f"{self.condition.label}-{self.swarm}-{self.radio:g}m/{self.seed}"
        return scenario.into_experiment(label).with_sharing(self.condition.sharing)


@dataclass
class PestSpread:
    """When ever more robots first knew of one pest.

    Attributes:
        first: When the first robot knew of it.
        half: When half of the robots did, if they ever did.
        most: When 90 % of the robots did, if they ever did.
    """

    first: Time
    half: Time | None = None
    most: Time | None = None


@dataclass
class KnowledgeProbe:
    """Samples how far the robots' knowledge drifts apart, and how fast pests become known.

    Attributes:
        agents: The robots' brains.
        known_pest_shares: Per sample with pests in the field, the mean over the robots of the
            share of those pests a robot believes in.
        last_seen_gaps: Per sample, the mean last-seen gap over robots and cells.
        spreads: Per pest cell, when ever more robots first knew of it.
    """

    agents: Sequence[ComposedAgent]
    known_pest_shares: list[float] = field(default_factory=list)
    last_seen_gaps: list[float] = field(default_factory=list)
    spreads: dict[CellId, PestSpread] = field(default_factory=dict)

    def sample(self, world: WorldState, now: Time) -> None:
        """Probe every robot's memory against the field's pests at ``now``."""
        pests: list[CellId] = []
        for cell, state in world.cells.items():
            if state.infestation > 0:
                pests.append(cell)
        robots = len(self.agents)
        believed = 0
        for cell in pests:
            knowing = 0
            for agent in self.agents:
                fact = agent.memory.infestation_at(cell)
                if fact is not None and fact.severity > 0:
                    knowing += 1
            believed += knowing
            self._note_spread(cell, knowing, robots, now)
        if pests:
            self.known_pest_shares.append(believed / (len(pests) * robots))

        last_seen = np.stack([agent.memory.last_observed_times() for agent in self.agents])
        last_seen = np.maximum(last_seen, 0.0)
        self.last_seen_gaps.append(float(np.mean(last_seen.max(axis=0) - last_seen)))

    def _note_spread(self, cell: CellId, knowing: int, robots: int, now: Time) -> None:
        if knowing == 0:
            return
        spread = self.spreads.setdefault(cell, PestSpread(first=now))
        if spread.half is None and knowing >= 0.5 * robots:
            spread.half = now
        if spread.most is None and knowing >= 0.9 * robots:
            spread.most = now


@dataclass(frozen=True)
class SharingResult:
    """What one run of the study measured.

    Attributes:
        condition: The sharing strategy's label.
        swarm: The swarm and field size, e.g. ``20x32``.
        radio: The robots' radio range in metres.
        seed: The scenario seed.
        wall_s_per_hour: The run's wall time per simulated hour, without the probing.
        facts_per_robot_hour: How many facts a robot broadcast per simulated hour.
        mean_idleness_min: The run's mean idleness of a crop cell.
        redundancy: The run's share of redundant visits.
        detection_latency_min: The run's mean detection latency, ``None`` if none.
        mean_infested: The infested cells, averaged over the samples.
        known_pest_share: The share of the field's pests a robot believes in, averaged.
        last_seen_gap_min: The mean last-seen gap.
        spread_half_s: The mean spread time to half of the robots, ``None`` if no pest got there.
        spread_most_s: The mean spread time to 90 % of the robots, ``None`` if none got there.
    """

    condition: str
    swarm: str
    radio: float
    seed: int
    wall_s_per_hour: float
    facts_per_robot_hour: float
    mean_idleness_min: float
    redundancy: float
    detection_latency_min: float | None
    mean_infested: float
    known_pest_share: float | None
    last_seen_gap_min: float
    spread_half_s: float | None
    spread_most_s: float | None


def run_one(spec: RunSpec, out: Path) -> SharingResult:
    """Run ``spec``, probing the robots every decision interval, and export its series."""
    config = spec.into_experiment()
    ctx, collector = build_run(config)
    agents: list[ComposedAgent] = []
    for agent in ctx.agents.values():
        if isinstance(agent, ComposedAgent):
            agents.append(agent)
    probe = KnowledgeProbe(agents)
    loop = SimulationLoop()
    seconds = 0.0
    now: Time = 0
    while now < config.until:
        now = min(now + config.decision_interval, config.until)
        started = perf_counter()
        loop.run(ctx, now)
        seconds += perf_counter() - started
        probe.sample(ctx.world, now)

    summary = RunSummary.from_run(config, collector.series())
    summary.export(out)
    facts = sum(agent.exchange.facts_sent for agent in agents)
    latency: float | None = None
    if summary.mean_detection_latency_ms is not None:
        latency = summary.mean_detection_latency_ms / TimeUnit.MINUTE
    return SharingResult(
        condition=spec.condition.label,
        swarm=str(spec.swarm),
        radio=spec.radio,
        seed=spec.seed,
        wall_s_per_hour=seconds / spec.hours,
        facts_per_robot_hour=facts / len(agents) / spec.hours,
        mean_idleness_min=summary.series[-1].mean_idleness_ms / TimeUnit.MINUTE,
        redundancy=summary.series[-1].redundancy,
        detection_latency_min=latency,
        mean_infested=mean(sample.targets_remaining for sample in summary.series),
        known_pest_share=_mean_or_none(probe.known_pest_shares),
        last_seen_gap_min=mean(probe.last_seen_gaps) / TimeUnit.MINUTE,
        spread_half_s=_mean_spread(probe.spreads.values(), half=True),
        spread_most_s=_mean_spread(probe.spreads.values(), half=False),
    )


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def _mean_spread(spreads: Iterable[PestSpread], half: bool) -> float | None:
    """The mean time in seconds from the first robot knowing of a pest to half or 90 % knowing."""
    durations: list[float] = []
    for spread in spreads:
        reached = spread.most
        if half:
            reached = spread.half
        if reached is not None:
            durations.append((reached - spread.first) / TimeUnit.SECOND)
    return _mean_or_none(durations)


@dataclass(frozen=True)
class Column:
    """One column of the printed tables.

    Attributes:
        title: The column's heading.
        value: Reads the column's value off a run, ``None`` if the run has none.
        digits: The decimals to print.
        outcome: Whether it measures how the swarm did, so that a strategy is checked against
            the spread of ``full`` over the seeds.
    """

    title: str
    value: Callable[[SharingResult], float | None]
    digits: int
    outcome: bool = False


COLUMNS: tuple[Column, ...] = (
    Column("s / sim h", lambda run: run.wall_s_per_hour, 0),
    Column("facts / robot / h", lambda run: run.facts_per_robot_hour, 0),
    Column("idleness (min)", lambda run: run.mean_idleness_min, 1, outcome=True),
    Column("redundancy", lambda run: run.redundancy, 3, outcome=True),
    Column("latency (min)", lambda run: run.detection_latency_min, 1, outcome=True),
    Column("infested", lambda run: run.mean_infested, 1, outcome=True),
    Column("known pests", lambda run: run.known_pest_share, 2),
    Column("last-seen gap (min)", lambda run: run.last_seen_gap_min, 1),
    Column("spread 50 % (s)", lambda run: run.spread_half_s, 0),
    Column("spread 90 % (s)", lambda run: run.spread_most_s, 0),
)


def _values(runs: Sequence[SharingResult], column: Column) -> list[float]:
    values: list[float] = []
    for run in runs:
        value = column.value(run)
        if value is not None:
            values.append(value)
    return values


def format_table(results: Sequence[SharingResult], swarm: str, radio: float) -> str:
    """One Markdown table for a swarm and radio range: one row per condition, means over seeds.

    The ``full`` row gives each outcome as mean ± standard deviation over the seeds. Another
    row marks an outcome with ``*`` where it differs from ``full`` by more than that deviation.
    """
    groups: dict[str, list[SharingResult]] = {}
    for result in results:
        if result.swarm == swarm and result.radio == radio:
            groups.setdefault(result.condition, []).append(result)
    lines = [
        "| strategy | speed-up | " + " | ".join(column.title for column in COLUMNS) + " |",
        "|---|" + "---:|" * (len(COLUMNS) + 1),
    ]
    reference = groups.get(CONDITIONS[0].label, [])
    reference_speed = _values(reference, COLUMNS[0])
    for condition in CONDITIONS:
        runs = groups.get(condition.label, [])
        if not runs:
            continue
        cells: list[str] = []
        speed = _values(runs, COLUMNS[0])
        if reference_speed and speed:
            cells.append(f"{mean(reference_speed) / mean(speed):.2f}x")
        else:
            cells.append("-")
        for column in COLUMNS:
            is_reference = condition is CONDITIONS[0]
            values = _values(runs, column)
            cells.append(_format_cell(column, values, _values(reference, column), is_reference))
        lines.append(f"| {condition.label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_cell(
    column: Column, values: list[float], reference: list[float], is_reference: bool
) -> str:
    """The mean of ``values``; with its deviation for ``full``, or ``*`` beyond its spread."""
    if not values:
        return "-"
    text = f"{mean(values):.{column.digits}f}"
    if not column.outcome or len(reference) < 2:
        return text
    spread = stdev(reference)
    if is_reference:
        return f"{text} ± {spread:.{column.digits}f}"
    if abs(mean(values) - mean(reference)) > spread:
        return text + " *"
    return text


def write_results(results: Sequence[SharingResult], path: Path) -> Path:
    """Write one row per run to ``path`` as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--swarms", nargs="+", default=["10x32", "20x32", "40x64"])
    parser.add_argument("--radios", type=float, nargs="+", default=[300.0], help="in metres")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[condition.label for condition in CONDITIONS],
        choices=[condition.label for condition in CONDITIONS],
    )
    parser.add_argument("--seeds", type=int, default=5, help="runs per condition, seeds 0, 1, ...")
    parser.add_argument("--hours", type=float, default=1.0, help="simulated time per run")
    parser.add_argument("--workers", type=int, default=4, help="runs in parallel")
    parser.add_argument("--out", type=Path, default=Path("runs/sharing"), help="results go here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs: list[RunSpec] = []
    for text in args.swarms:
        for radio in args.radios:
            for condition in CONDITIONS:
                if condition.label not in args.conditions:
                    continue
                for seed in range(args.seeds):
                    specs.append(RunSpec(condition, Swarm.from_text(text), radio, seed, args.hours))
    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, spec, args.out) for spec in specs]
        results: list[SharingResult] = []
        for spec, future in zip(specs, futures, strict=True):
            results.append(future.result())
            print(f"done {spec.into_experiment().label}", flush=True)

    write_results(results, args.out / "sharing.csv")
    for text in args.swarms:
        for radio in args.radios:
            setup = f"radio {radio:g} m, {args.hours:g} h, {args.seeds} seeds"
            print(f"\n{text} (robots x field side), {setup}\n")
            print(format_table(results, text, radio))
    print(f"\nWrote {len(results)} runs to {args.out}")


if __name__ == "__main__":
    main()
