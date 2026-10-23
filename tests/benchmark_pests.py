"""Benchmark pest response: the arms of the relay scenario, ``RELAYED_KILLERS``.

Every arm is a ``PestScenario`` measured with ``measure_pest_response``, repeated with several
seeds; the columns of the table are described in ``swarm_field.experiments.pest_response``.
The runs, their results and the table go to a new folder in ``output/``. Not part of
``pytest``; run it with:

    uv run python -m tests.benchmark_pests --hours 3 --seeds 10 --pests-per-hour 15
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace

from swarm_field.experiments import (
    RELAYED_KILLERS,
    Experiment,
    PestRunSpec,
    PestScenario,
    by_arm,
    markdown_table,
    measure_pest_response,
    pest_response_configurations,
    pest_table_cells,
    pest_table_headers,
    runs_for,
)
from swarm_field.swarm import BeaconConfig, KnowledgeSharing


def arms(seeded_no_yield: bool = False) -> dict[str, PestScenario]:
    """The seeded arm, the oracle, which hears every pest at once, and the hop arms, by name.

    ``hop`` counts integer hops without a cap. The others are ``pest_response_configurations``
    of ``RELAYED_KILLERS``: ``seeded`` with the killers' claims, ``hop metric`` with metric hops
    of one radio range and the cap, ``hop repel`` with killers repelling each other, and
    ``hop tuned``, the best hop configuration so far, with pulls weighted by the square root of
    the distance and the escape. With ``seeded_no_yield``, ``seeded no yield`` is the seeded arm
    without the claims.
    """
    configurations = pest_response_configurations(RELAYED_KILLERS, seeded_no_yield)
    oracle = replace(
        RELAYED_KILLERS, pest_source_range_m=math.inf, sharing=KnowledgeSharing.NOTHING
    )
    found = {"seeded": configurations["seeded"]}
    if seeded_no_yield:
        found["seeded no yield"] = configurations["seeded-no-yield"]
    found["oracle"] = oracle
    found["hop"] = replace(configurations["hop"], beacons=BeaconConfig())
    found["hop metric"] = configurations["hop"]
    found["hop repel"] = configurations["hop-repel"]
    found["hop tuned"] = configurations["hop-tuned"]
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=int, default=3, help="simulated hours per run")
    parser.add_argument("--seeds", type=int, default=10, help="runs per arm")
    parser.add_argument("--jobs", type=int, default=8, help="processes")
    parser.add_argument("--pests-per-hour", type=float, default=15.0, help="pest spawn rate")
    parser.add_argument("--warmup-hours", type=int, default=1, help="left out of infested")
    parser.add_argument(
        "--seeded-no-yield", action="store_true", help="also run seeded without the claims"
    )
    args = parser.parse_args()

    specs: dict[str, PestRunSpec] = {}
    for name, scenario in arms(args.seeded_no_yield).items():
        specs[name] = PestRunSpec(scenario, args.hours, args.pests_per_hour, args.warmup_hours)
    experiment = Experiment("pest_benchmark")
    runs = runs_for(specs, range(args.seeds), PestRunSpec.cost)
    results = experiment.run(runs, measure_pest_response, args.jobs)

    scenario = RELAYED_KILLERS
    title = (
        f"{scenario.scouts} relays, {scenario.killers} killers, "
        f"{scenario.radio_range_m:g} m radio, {args.pests_per_hour:g} pests per hour, "
        f"{args.hours} h (warm-up {args.warmup_hours} h), mean over {args.seeds} seeds"
    )
    rows: list[list[str]] = []
    for name, arm_results in by_arm(results).items():
        rows.append(pest_table_cells(name, arm_results, args.hours))
    text = f"{title}\n\n{markdown_table(pest_table_headers(args.hours), rows)}"
    print(text)
    experiment.write_text("tables.md", text)


if __name__ == "__main__":
    main()
