"""Study hop killers that hear of pests only from the explorers moving over the field (#70).

Question: on the small river field (``SMALL_RIVER_MAP``), do hop killers find and treat the
pests when the only robots that relay their beacons are explorers that keep moving? There are no
standing relays and no mock sources on the pests: an explorer that senses a pest is its source,
and every explorer relays the hop values it hears. This is ``moving_explorer_scenario``: battery
and doses are unlimited, so that no robot needs a dock and the comparison is about the pest
response alone; ``studies/hop_dock_return.py`` adds the dock
trips. The explorers set their revisit age from the swarm's sweep time, as every
``ExplorerPreset`` does. Seeded killers with the same explorers run as a reference, on the same
seeds, so every row differs from ``seeded`` only in the pest response.

The runs are measured with ``measure_pest_response`` and run in parallel by ``Experiment``,
which writes them and the table to a new folder in ``output/``. The runs of the arms in
``--film`` are then run again and filmed into its ``video/`` folder. Not part of ``pytest``; run
it with:

    uv run python -m studies.explorer_hop_killers --seeds 4
"""

from __future__ import annotations

import argparse
import logging

from swarm_field.contracts import TimeUnit
from swarm_field.experiments import (
    HOP_KILLER,
    LATTICE_ROBOTS,
    PESTS_PER_KILLER_HOUR,
    REVISIT_SWEEPS,
    SEEDED_KILLER,
    Experiment,
    RobotPreset,
    SwarmScenario,
    arm_name,
    by_arm,
    markdown_table,
    measure_pest_response,
    moving_explorer_scenario,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--explorers", type=int, default=20)
    parser.add_argument("--killers", type=int, default=4)
    parser.add_argument(
        "--pests-per-hour",
        type=float,
        help="pest rate; PESTS_PER_KILLER_HOUR per killer if not given",
    )
    parser.add_argument(
        "--radio", type=float, default=LATTICE_ROBOTS.radio_range_m, help="radio range in m"
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    rate = args.pests_per_hour
    if rate is None:
        rate = PESTS_PER_KILLER_HOUR * args.killers

    scenarios: dict[str, SwarmScenario] = {}
    arms: dict[str, str] = {}
    for name, killer in KILLERS.items():
        scenario = moving_explorer_scenario(killer, args.explorers, args.killers, args.radio)
        scenario = scenario.with_pests_per_hour(rate)
        arm = arm_name(name, explorers=args.explorers, killers=args.killers, rate=rate)
        scenarios[arm] = scenario.with_hours(args.hours, args.warmup_hours)
        arms[name] = arm

    experiment = Experiment("explorer_hop_killers")
    runs = runs_for(scenarios, range(args.seeds), SwarmScenario.run_cost)
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
    sweep_time = moving_explorer_scenario(
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

    filmed: dict[str, SwarmScenario] = {}
    for name in args.film:
        filmed[arms[name]] = scenarios[arms[name]]
    if filmed:
        record_videos(experiment.directory, runs_for(filmed, range(args.seeds)), args.jobs)


if __name__ == "__main__":
    main()
