"""Study hop killers that hear of pests only from the explorers moving over the field (#70).

Question: on the small river field (``RIVER_SMALL``'s map), do hop killers find and treat the
pests when the only robots that relay their beacons are explorers that keep moving? There are no
standing relays and no mock sources on the pests: an explorer that senses a pest is its source,
and every explorer relays the hop values it hears. Every battery holds ``BATTERY_CAPACITY``
charges, since no robot charges yet and a scout's charge lasts only 30 minutes, and every
killer carries ``TREATMENT_DOSES`` doses, since none restocks and 10 run out within a run. The
explorers set their revisit age from the swarm's sweep time, as every ``ExplorerPreset`` does.
Seeded killers with the same explorers run as a reference, on the same seeds, so every row
differs from ``seeded`` only in the pest response.

The runs are measured with ``measure_pest_response`` and run in parallel by ``Experiment``,
which writes them and the table to a new folder in ``output/``. The runs of the arms in
``--film`` are then run again and filmed into its ``video/`` folder. Not part of ``pytest``; run
it with:

    uv run python -m studies.explorer_hop_killers --seeds 4
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from swarm_field.contracts import TimeUnit
from swarm_field.experiments import (
    EXPLORER,
    HOP_KILLER,
    REVISIT_SWEEPS,
    RIVER_SMALL,
    SEEDED_KILLER,
    Experiment,
    PestRunSpec,
    RobotPreset,
    SwarmScenario,
    arm_name,
    by_arm,
    markdown_table,
    measure_pest_response,
    paired_difference,
    pest_table_cells,
    pest_table_headers,
    record_videos,
    runs_for,
    treated_share,
)

KILLERS: dict[str, RobotPreset] = {"seeded": SEEDED_KILLER, "hop": HOP_KILLER}
"""The pest responses compared, by the name used in arm names and tables."""
BASELINE = "seeded"
"""The response every other is compared with, seed by seed."""
BATTERY_CAPACITY = 1000.0
"""Charges in every robot's battery: enough that no robot runs empty during a run."""
TREATMENT_DOSES = 1000
"""Doses every killer carries: enough that no killer runs out during a run."""
PESTS_PER_KILLER_HOUR = RIVER_SMALL.pests.pests_per_hour / RIVER_SMALL.killers
"""The load of ``RIVER_SMALL`` on each killer, which sets the default pest rate."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--explorers", type=int, default=20)
    parser.add_argument("--killers", type=int, default=4)
    parser.add_argument(
        "--pests-per-hour", type=float, help="pest rate; RIVER_SMALL's load per killer if not given"
    )
    parser.add_argument(
        "--radio", type=float, default=RIVER_SMALL.robots.radio_range_m, help="radio range in m"
    )
    parser.add_argument("--seeds", type=int, default=4, help="runs per arm, seeds 0, 1, ...")
    parser.add_argument("--hours", type=int, default=3, help="simulated hours per run")
    parser.add_argument("--warmup-hours", type=int, default=1, help="left out of infested")
    parser.add_argument(
        "--film",
        nargs="*",
        choices=list(KILLERS),
        default=list(KILLERS),
        help="arms to film, every seed",
    )
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    return parser.parse_args()


def explorer_scenario(
    killer: RobotPreset, explorers: int, killers: int, radio_range_m: float
) -> SwarmScenario:
    """``RIVER_SMALL``'s field with explorers and killers of one preset, and no mock sources."""
    robots = replace(
        RIVER_SMALL.robots, radio_range_m=radio_range_m, battery_capacity=BATTERY_CAPACITY
    )
    return (
        RIVER_SMALL.without_robots()
        .with_pests(
            replace(RIVER_SMALL.pests, source_range_m=None, treatment_doses=TREATMENT_DOSES)
        )
        .with_robot_config(robots)
        .add_robots(EXPLORER, explorers)
        .add_robots(killer, killers)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    rate = args.pests_per_hour
    if rate is None:
        rate = PESTS_PER_KILLER_HOUR * args.killers

    specs: dict[str, PestRunSpec] = {}
    arms: dict[str, str] = {}
    for name, killer in KILLERS.items():
        scenario = explorer_scenario(killer, args.explorers, args.killers, args.radio)
        arm = arm_name(name, explorers=args.explorers, killers=args.killers, rate=rate)
        specs[arm] = PestRunSpec(scenario, args.hours, rate, args.warmup_hours)
        arms[name] = arm

    experiment = Experiment("explorer_hop_killers")
    runs = runs_for(specs, range(args.seeds), PestRunSpec.cost)
    results = by_arm(experiment.run(runs, measure_pest_response, args.jobs))

    headers = pest_table_headers(args.hours, first="killers")
    headers.extend(["treated %", "Δ infested"])
    baseline = [result.infested for result in results[arms[BASELINE]]]
    rows: list[list[str]] = []
    for name, arm in arms.items():
        arm_results = results[arm]
        difference = "-"
        if name != BASELINE:
            difference = paired_difference([result.infested for result in arm_results], baseline)
        cells = pest_table_cells(name, arm_results, args.hours)
        cells.extend([treated_share(arm_results), difference])
        rows.append(cells)
    sweep_time = explorer_scenario(
        HOP_KILLER, args.explorers, args.killers, args.radio
    ).sweep_time()
    revisit_age = REVISIT_SWEEPS * sweep_time
    title = (
        f"river-small: {args.explorers} explorers, {args.killers} killers, no relays or mock "
        f"sources, {args.radio:g} m radio, {rate:g} pests per hour, explorers revisit after "
        f"{revisit_age / TimeUnit.MINUTE:.1f} min, {args.hours} h "
        f"(warm-up {args.warmup_hours} h), mean over {args.seeds} seeds"
    )
    text = f"{title}\n\n{markdown_table(headers, rows)}"
    print(text)
    experiment.write_text("tables.md", text)

    filmed: dict[str, PestRunSpec] = {}
    for name in args.film:
        filmed[arms[name]] = specs[arms[name]]
    if filmed:
        record_videos(experiment.directory, runs_for(filmed, range(args.seeds)), args.jobs)


if __name__ == "__main__":
    main()
