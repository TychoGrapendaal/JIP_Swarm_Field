"""Benchmark: how much of a pest run's wall time goes to knowledge sharing.

Run it with ``uv run python -m tests.benchmark_sharing``, optionally with ``--cases 10x32 20x32
40x64`` (robots x field side), ``--minutes 10``, ``--radio 300`` or ``--strategy periodic``. It is
not a pytest module: a run takes seconds to minutes.

Every case is a ``PestScenario`` with seed 0, three killers per ten robots, and every robot
sharing as ``--strategy`` says. Sharing is the time spent in ``KnowledgeExchange`` (merging what
a robot hears, and picking what it broadcasts) and in reading the terrain it learned back into
its quadtree. It is measured by wrapping those methods with a timer rather than with a profiler,
which would slow the many small calls of the rest of the run more than the vectorised merges.
"""

import argparse
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import Any

from swarm_field.contracts import TimeUnit
from swarm_field.experiments import PestScenario
from swarm_field.runner import ExperimentConfig, build_run
from swarm_field.simulation import SimulationLoop
from swarm_field.swarm import (
    AgentMemory,
    ComposedAgent,
    FieldProvider,
    KnowledgeExchange,
    SharingConfig,
    SharingStrategy,
)


@dataclass(frozen=True)
class Case:
    """One benchmarked run.

    Attributes:
        robots: How many robots, three killers per ten.
        size: The field's side in cells.
    """

    robots: int
    size: int

    @classmethod
    def from_text(cls, text: str) -> "Case":
        """Read a case written as ``ROBOTSxSIZE``, e.g. ``20x32``."""
        robots, size = text.split("x")
        return cls(int(robots), int(size))


@dataclass(frozen=True)
class SharingTiming:
    """Where one run's wall time went.

    Attributes:
        case: The run.
        seconds: The wall time of the run, without assembling it.
        sharing_seconds: The part of it spent sharing.
        decisions: How many decisions the robots took.
        facts_per_message: The mean number of facts per knowledge broadcast.
    """

    case: Case
    seconds: float
    sharing_seconds: float
    decisions: int
    facts_per_message: float


class SharingTimer:
    """Adds up the wall time spent in the sharing methods while it is installed."""

    def __init__(self) -> None:
        self.seconds: float = 0.0

    def timed[**P, R](self, method: Callable[P, R]) -> Callable[P, R]:
        @wraps(method)
        def timed_method(*args: P.args, **kwargs: P.kwargs) -> R:
            started = perf_counter()
            try:
                return method(*args, **kwargs)
            finally:
                self.seconds += perf_counter() - started

        return timed_method

    @contextmanager
    def installed(self) -> Generator["SharingTimer"]:
        """Wrap the sharing methods for as long as the context lasts."""
        methods: list[tuple[type, str]] = [
            (KnowledgeExchange, "receive"),
            (KnowledgeExchange, "knowledge_to_send"),
            (AgentMemory, "terrain_stored_after"),
            (FieldProvider, "update_terrain"),
        ]
        originals: list[tuple[type, str, Any]] = []
        for owner, name in methods:
            original = getattr(owner, name)
            originals.append((owner, name, original))
            setattr(owner, name, self.timed(original))
        try:
            yield self
        finally:
            for owner, name, original in originals:
                setattr(owner, name, original)


def pest_config(
    case: Case, minutes: float, radio: float, sharing: SharingConfig
) -> ExperimentConfig:
    """The pest scenario of ``case``, with every robot sharing as ``sharing`` says."""
    killers = case.robots * 3 // 10
    scenario = PestScenario(
        size=case.size,
        scouts=case.robots - killers,
        killers=killers,
        radio_range_m=radio,
        until=int(minutes * TimeUnit.MINUTE),
        seed=0,
    )
    return scenario.into_experiment(f"{case.robots}x{case.size}").with_sharing(sharing)


def measure_case(case: Case, minutes: float, radio: float, sharing: SharingConfig) -> SharingTiming:
    """Run one case and time it, and the sharing in it."""
    config = pest_config(case, minutes, radio, sharing)
    ctx, _ = build_run(config)
    timer = SharingTimer()
    with timer.installed():
        started = perf_counter()
        SimulationLoop().run(ctx, config.until)
        seconds = perf_counter() - started
    decisions = int(case.robots * config.until / config.decision_interval)
    facts = 0
    for agent in ctx.agents.values():
        if isinstance(agent, ComposedAgent):
            facts += agent.exchange.facts_sent
    return SharingTiming(
        case=case,
        seconds=seconds,
        sharing_seconds=timer.seconds,
        decisions=decisions,
        facts_per_message=facts / max(decisions, 1),
    )


def format_results_table(results: list[SharingTiming], minutes: float) -> str:
    """The results as a plain-text table, one row per case."""
    header = (
        f"{'robots':>6} {'field':>7} {'wall s':>8} {'s per sim h':>12} {'sharing s':>10} "
        f"{'sharing %':>10} {'ms/decision':>12} {'facts/msg':>10}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        size = result.case.size
        lines.append(
            f"{result.case.robots:>6} {f'{size}x{size}':>7} {result.seconds:>8.1f} "
            f"{result.seconds * 60 / minutes:>12.0f} {result.sharing_seconds:>10.2f} "
            f"{result.sharing_seconds / result.seconds:>10.1%} "
            f"{result.sharing_seconds * 1000 / result.decisions:>12.2f} "
            f"{result.facts_per_message:>10.0f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", nargs="+", default=["10x32", "20x32", "40x64"])
    parser.add_argument("--minutes", type=float, default=10.0, help="simulated time per run")
    parser.add_argument("--radio", type=float, default=300.0, help="radio range in metres")
    parser.add_argument(
        "--strategy",
        choices=[strategy.name.lower() for strategy in SharingStrategy],
        default="full",
        help="what every robot broadcasts of its memory, with the default parameters",
    )
    arguments = parser.parse_args()
    sharing = SharingConfig(SharingStrategy[arguments.strategy.upper()])

    results: list[SharingTiming] = []
    for text in arguments.cases:
        case = Case.from_text(text)
        print(f"running {case.robots} robots on {case.size} x {case.size} ...", flush=True)
        results.append(measure_case(case, arguments.minutes, arguments.radio, sharing))
    print()
    print(format_results_table(results, arguments.minutes))
    print()
    print(f"{arguments.minutes:g} simulated minutes per run, radio {arguments.radio:g} m, ")
    print(f"sharing {sharing.describe()}. sharing: merging what is heard, picking what to")
    print("broadcast, and reading learned terrain into the quadtree. facts/msg: mean facts per")
    print("knowledge broadcast, over all decisions.")


if __name__ == "__main__":
    main()
