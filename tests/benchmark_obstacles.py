"""Benchmark pest response on fields with obstacles: four configurations, two scenarios (#70).

Every configuration of ``pest_response_configurations`` runs on ``RIVER_SMALL`` and on
``RIVERS_LARGE``, with the same seeds. A run is measured as in ``tests/benchmark_pests.py``, whose
columns the table repeats, and adds:

- ``treated %``: the share of the pests that appeared and were treated, ``treated / (treated +
  open)``;
- ``Δ infested``: the paired difference in infested cells to ``seeded`` over the same seeds, with
  its standard error.

Every run's result is also written as JSON to ``runs/70-obstacle-comparison/results/``.

``--calibrate`` runs only ``hop-tuned`` at the pest rates given, to pick the rate at which it
treats 85-90 % of the pests. Not part of ``pytest``; run it with:

    uv run python -m tests.benchmark_obstacles --seeds 8
    uv run python -m tests.benchmark_obstacles --scenario river-small --calibrate 4 6 8 --seeds 2
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev

from swarm_field.experiments import (
    RIVER_SMALL,
    RIVERS_LARGE,
    PestScenario,
    pest_response_configurations,
)
from tests.benchmark_pests import Arm, Job, Result, mean_and_error, run, table_row

SCENARIOS: dict[str, PestScenario] = {"river-small": RIVER_SMALL, "rivers-large": RIVERS_LARGE}
"""The scenarios by the name used in file names and tables."""
RESULTS = Path("runs/70-obstacle-comparison/results")
"""Where every run's result is written."""
HOURS = 3
WARMUP_HOURS = 1


def jobs_for(name: str, scenario: PestScenario, configs: list[str], seeds: int) -> list[Job]:
    """One job per configuration and seed, at the scenario's own pest rate."""
    responses = pest_response_configurations(scenario)
    jobs: list[Job] = []
    for config in configs:
        arm = Arm(f"{name}_{config}", responses[config])
        for seed in range(seeds):
            jobs.append(Job(arm, HOURS, seed, scenario.pests_per_hour, WARMUP_HOURS))
    return jobs


def treated_fraction(results: list[Result]) -> str:
    """The share of the appeared pests that were treated, as mean ± standard error."""
    fractions: list[float] = []
    for result in results:
        fractions.append(result.treated / max(result.treated + result.open, 1))
    return mean_and_error(fractions)


def paired_difference(results: list[Result], seeded: list[Result]) -> str:
    """Infested cells minus those of ``seeded`` on the same seed, as mean ± standard error."""
    differences: list[float] = []
    for result, baseline in zip(results, seeded, strict=True):
        differences.append(result.infested - baseline.infested)
    error = 0.0
    if len(differences) > 1:
        error = stdev(differences) / math.sqrt(len(differences))
    return f"{mean(differences):+.2f} ± {error:.2f}"


def print_table(title: str, rows: dict[str, list[Result]]) -> None:
    """One Markdown table: the ``benchmark_pests`` columns, the treated share, and Δ infested."""
    print(f"\n{title}\n")
    headers = ["config", "infested", "treated", "response", "p90", "open", "cells/treat", "msgs"]
    headers.extend(["stuck", "commit hop", "empty"])
    for hour in range(1, HOURS + 1):
        headers.append(f"drift {hour} h")
    headers.extend(["wall (s)", "treated %", "Δ infested"])
    print("| " + " | ".join(headers) + " |")
    print("|" + " --- |" * len(headers))
    seeded = rows.get("seeded")
    for name, results in rows.items():
        difference = "-"
        if seeded is not None and name != "seeded":
            difference = paired_difference(results, seeded)
        row = table_row(name, results, HOURS)
        print(f"{row} {treated_fraction(results)} | {difference} |")


def save_results(jobs: list[Job], results: list[Result]) -> None:
    """Write every run's result as JSON, named after its arm, pest rate and seed."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    for job, result in zip(jobs, results, strict=True):
        path = RESULTS / f"{job.arm.name}_rate{job.pests_per_hour:g}_seed{job.seed}.json"
        record: dict[str, object] = {"arm": job.arm.name, "seed": job.seed}
        record["pests_per_hour"] = job.pests_per_hour
        record["killers"] = job.arm.scenario.killers
        record.update(asdict(result))
        path.write_text(json.dumps(record, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=[*SCENARIOS, "both"], default="both")
    parser.add_argument("--seeds", type=int, default=8, help="runs per configuration")
    parser.add_argument("--jobs", type=int, default=22, help="processes")
    parser.add_argument("--killers", type=int, help="override the scenario's killers")
    parser.add_argument(
        "--calibrate", type=float, nargs="+", help="run only hop-tuned at these pest rates"
    )
    args = parser.parse_args()

    names = list(SCENARIOS)
    if args.scenario != "both":
        names = [args.scenario]
    jobs: list[Job] = []
    for name in names:
        scenario = SCENARIOS[name]
        if args.killers is not None:
            scenario = replace(scenario, killers=args.killers)
        if args.calibrate is None:
            jobs.extend(
                jobs_for(name, scenario, list(pest_response_configurations(scenario)), args.seeds)
            )
            continue
        for rate in args.calibrate:
            jobs.extend(
                jobs_for(name, replace(scenario, pests_per_hour=rate), ["hop-tuned"], args.seeds)
            )
    # The slowest runs first, so that no worker is left with one of them at the end.
    order = sorted(range(len(jobs)), key=lambda index: _expected_cost(jobs[index]), reverse=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        finished = list(pool.map(run, [jobs[index] for index in order]))
    results: list[Result] = [finished[0]] * len(jobs)
    for position, index in enumerate(order):
        results[index] = finished[position]
    save_results(jobs, results)

    groups: dict[tuple[str, float, int], dict[str, list[Result]]] = {}
    for job, result in zip(jobs, results, strict=True):
        name, config = job.arm.name.split("_")
        key = (name, job.pests_per_hour, job.arm.scenario.killers)
        groups.setdefault(key, {}).setdefault(config, []).append(result)
    for (name, rate, killers), rows in groups.items():
        scenario = SCENARIOS[name]
        title = (
            f"{name}: {scenario.scouts} relays, {killers} killers, "
            f"{scenario.radio_range_m:g} m radio, {rate:g} pests per hour, "
            f"{HOURS} h (warm-up {WARMUP_HOURS} h), mean over {args.seeds} seeds"
        )
        print_table(title, rows)


def _expected_cost(job: Job) -> float:
    """A rough cost of a job, to start the slowest first: seeded and large runs cost most."""
    cost = float(job.arm.scenario.size**2)
    if job.arm.name.endswith("_seeded"):
        cost *= 5
    return cost


if __name__ == "__main__":
    main()
