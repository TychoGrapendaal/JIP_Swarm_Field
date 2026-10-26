"""Study whether killers that lose radio contact with the swarm lose pests (#70, step 6).

Question: on the small river field with moving explorers only (the scenario of
``studies/explorer_hop_killers.py``), how often is a killer cut off from every other robot, and
do pests go untreated because the robots that know of them have no killer in their radio
network? A seeded killer keeps the pests it heard of in memory, while a hop killer forgets a
beacon after a minute, so cohesion may matter for hop killers only. The hop arm uses
``hop-tuned`` with a 3-minute ``source_for``, its best setting so far.

``cut off`` counts the pests that some robot knows of but that no killer can hear of, an upper
bound on what cohesion could save; ``unknown`` the pests no robot knows of, which only more
detection can save. Cohesion is worth building if, for the hop arm at 50 m, ``cut off`` is at
least a fifth of ``infested`` and at least half a cell, on at least three quarters of the seeds,
and isolation spells last long (a 90th percentile of 2 minutes or more). Not part of ``pytest``;
run it with:

    uv run python -m studies.killer_isolation --seeds 8
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from swarm_field.contracts import TimeUnit
from swarm_field.experiments import (
    PESTS_PER_KILLER_HOUR,
    RIVER_SMALL,
    SEEDED_KILLER,
    Experiment,
    HopKillerPreset,
    PestResponse,
    PestRunSpec,
    RobotPreset,
    arm_name,
    by_arm,
    markdown_table,
    measure_pest_response,
    moving_explorer_scenario,
    pest_isolation_cells,
    pest_isolation_headers,
    pest_table_cells,
    pest_table_headers,
    runs_for,
)
from swarm_field.experiments.presets import hop_beacon_variants

RADII_M = (30.0, 50.0)
"""The radio ranges compared, in metres."""
SOURCE_FOR = 3 * TimeUnit.MINUTE
"""How long a hop explorer keeps beaconing a pest it saw: the best setting of the hover study."""
CUT_OFF_SHARE = 0.2
"""``cut off`` must be at least this share of ``infested`` for cohesion to pay."""
CUT_OFF_CELLS = 0.5
"""And at least this many infested cells."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--explorers", type=int, default=20)
    parser.add_argument("--killers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=8, help="runs per arm, seeds 0, 1, ...")
    parser.add_argument("--hours", type=int, default=3, help="simulated hours per run")
    parser.add_argument("--warmup-hours", type=int, default=1, help="left out of infested")
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    return parser.parse_args()


def killers(radio_m: float) -> dict[str, RobotPreset]:
    """The seeded killer and the hop killer of a ``radio_m`` radio, by name."""
    hop_cells = radio_m / RIVER_SMALL.map.cell_size
    tuned = hop_beacon_variants(hop_cells)["hop-tuned"]
    hop = HopKillerPreset(beacons=replace(tuned, source_for=SOURCE_FOR))
    return {"seeded": SEEDED_KILLER, "hop": hop}


def verdict(results: list[PestResponse]) -> str:
    """How many seeds meet the cut-off part of the cohesion criterion."""
    meeting = 0
    for result in results:
        enough = result.cut_off_infested >= CUT_OFF_SHARE * result.infested
        if enough and result.cut_off_infested >= CUT_OFF_CELLS:
            meeting += 1
    return f"{meeting}/{len(results)} seeds"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    rate = PESTS_PER_KILLER_HOUR * args.killers
    specs: dict[str, PestRunSpec] = {}
    for radio in RADII_M:
        for name, killer in killers(radio).items():
            scenario = moving_explorer_scenario(killer, args.explorers, args.killers, radio)
            specs[arm_name(name, radio=f"{radio:g}m")] = PestRunSpec(
                scenario, args.hours, rate, args.warmup_hours
            )

    experiment = Experiment("killer_isolation")
    runs = runs_for(specs, range(args.seeds), PestRunSpec.cost)
    results = by_arm(experiment.run(runs, measure_pest_response, args.jobs))

    standard: list[list[str]] = []
    isolation: list[list[str]] = []
    for arm, arm_results in results.items():
        standard.append(pest_table_cells(arm, arm_results, args.hours))
        cells = pest_isolation_cells(arm, arm_results)
        cells.append(verdict(arm_results))
        isolation.append(cells)
    headers = pest_isolation_headers()
    headers.append("cut off criterion")
    title = (
        f"river-small: {args.explorers} moving explorers, {args.killers} killers, "
        f"{rate:g} pests per hour, {args.hours} h (warm-up {args.warmup_hours} h), mean over "
        f"{args.seeds} seeds; hop with source_for {SOURCE_FOR // TimeUnit.MINUTE} min"
    )
    text = (
        f"{title}\n\n{markdown_table(pest_table_headers(args.hours), standard)}\n\n"
        f"{markdown_table(headers, isolation)}"
    )
    print(text)
    experiment.write_text("tables.md", text)


if __name__ == "__main__":
    main()
