"""Tune each pest response one parameter at a time, to freeze its best configuration (step 7).

Question: on the scenario of the core comparison (the small river field with moving explorers
only, as in ``studies/explorer_hop_killers.py``), which value of each parameter leaves the
fewest infested cells, for each arm on its own? Every variant changes one parameter of its arm's
base and is compared seed by seed with that base: ``hop-tuned`` with a 3-minute ``source_for``
for the hop arm, ``SEEDED_KILLER`` for the seeded arm. The explorers' revisit age is shared by
both arms and is varied in both. A change is worth adopting when it beats the base by at least
two standard errors of the paired difference.

``--stage combine`` then tests the promising changes of the sweep on top of the adopted one
(``fresh_for`` of 2 minutes), alone and together, against the hop base with that change; and the
longer claim ages of the seeded arm again. A value is frozen only if it holds on both fields:
``--field large`` runs on the large field with two rivers (``LARGE_RIVERS_MAP``), with four
times the area and, by default, half the robots per cell of the small one, at the same load per
killer. Not part of ``pytest``; run it with:

    uv run python -m studies.tune_arms --seeds 8
    uv run python -m studies.tune_arms --stage combine --seeds 16
    uv run python -m studies.tune_arms --stage combine --field large --seeds 8

``--stage refine`` checks one step beyond the adopted values, on either field, and
``--stage verify`` runs the frozen ``SEEDED_BEST`` and ``HOP_BEST`` against the defaults on seeds
not used for tuning, e.g. ``--first-seed 100``.

``--stage defaults`` tunes them again on the four default scenarios (``RIVER_SMALL``,
``RIVERS_LARGE``, ``OPEN_SMALL``, ``OPEN_LARGE``) as they are, with their own robots, loads and
run lengths: the ages of the hop beacons, the cap on a beacon's hops and the seeded claim age, one
at a time around ``HOP_BEST`` and ``SEEDED_BEST``. It ignores ``--field`` and the robot and hour
options, and takes about three hours on 24 processes:

    uv run python -m studies.tune_arms --stage defaults --seeds 8

``--stage source-sweeps`` then checks on fresh seeds what it pointed to, a ``source_for`` of
``HOP_BEST`` that grows with the swarm's sweep time, which ``HOP_BEST`` now has:

    uv run python -m studies.tune_arms --stage source-sweeps --seeds 8 --first-seed 100
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from statistics import mean, stdev

from swarm_field.contracts import BodyType, TimeUnit
from swarm_field.experiments import (
    DEFAULT_EXPLORERS,
    DEFAULT_KILLERS,
    EXPLORER,
    HOP_BEST,
    HOP_KILLER,
    LARGE_RIVERS_MAP,
    LATTICE_ROBOTS,
    OPEN_LARGE,
    OPEN_SMALL,
    PESTS_PER_KILLER_HOUR,
    RIVER_SMALL,
    RIVERS_LARGE,
    SEEDED_BEST,
    SEEDED_KILLER,
    SMALL_RIVER_MAP,
    Experiment,
    ExplorerPreset,
    HopKillerPreset,
    MapConfig,
    PestResponse,
    RobotPreset,
    SeededKillerPreset,
    SwarmScenario,
    by_arm,
    markdown_table,
    measure_pest_response,
    moving_explorer_scenario,
    paired_difference,
    pest_table_cells,
    pest_table_headers,
    runs_for,
)
from swarm_field.experiments.presets import hop_beacon_variants
from swarm_field.swarm import BeaconConfig

MINUTE = TimeUnit.MINUTE


@dataclass(frozen=True)
class Variant:
    """One arm of the sweep: a killer and an explorer, and the base it is compared with.

    Attributes:
        name: The arm's name, ``<base>/<what changed>``.
        base: The name of the arm it is compared with, or ``None`` for a base itself.
        killer: The killer preset.
        explorer: The explorer preset.
    """

    name: str
    base: str | None
    killer: RobotPreset
    explorer: RobotPreset = EXPLORER


def hop_base() -> BeaconConfig:
    """The hop arm's base: ``hop-tuned`` with a 3-minute ``source_for``."""
    hop_cells = LATTICE_ROBOTS.radio_range_m / SMALL_RIVER_MAP.cell_size
    return replace(hop_beacon_variants(hop_cells)["hop-tuned"], source_for=3 * MINUTE)


def variants() -> list[Variant]:
    """The bases and every one-parameter change of them."""
    base = hop_base()
    hop_changes: dict[str, BeaconConfig] = {
        "source_for=2m": replace(base, source_for=2 * MINUTE),
        "source_for=5m": replace(base, source_for=5 * MINUTE),
        "fresh_for=2m": replace(base, fresh_for=2 * MINUTE),
        "decay=2m": replace(base, decay=2 * MINUTE),
        "max_value=3": replace(base, max_value=3.0),
        "max_value=6": replace(base, max_value=6.0),
        "max_value=none": replace(base, max_value=None),
        "killer_repulsion=0": replace(base, killer_repulsion=0.0),
        "killer_repulsion=1": replace(base, killer_repulsion=1.0),
        "killer_repulsion=4": replace(base, killer_repulsion=4.0),
        "distance_power=0": replace(base, distance_power=0.0),
        "distance_power=1": replace(base, distance_power=1.0),
        "gamma=0.7": replace(base, gamma=0.7),
        "escape=off": replace(base, escape=None),
    }
    seeded_changes: dict[str, SeededKillerPreset] = {
        "claim_fresh_for=1m": replace(SEEDED_KILLER, claim_fresh_for=MINUTE),
        "claim_fresh_for=5m": replace(SEEDED_KILLER, claim_fresh_for=5 * MINUTE),
        "claim_fresh_for=10m": replace(SEEDED_KILLER, claim_fresh_for=10 * MINUTE),
        "window_blocks=1": replace(SEEDED_KILLER, window_blocks=1),
        "window_blocks=4": replace(SEEDED_KILLER, window_blocks=4),
        "claims=off": replace(SEEDED_KILLER, claims=False),
    }
    hop = HopKillerPreset(beacons=base)
    found = [Variant("hop", None, hop), Variant("seeded", None, SEEDED_KILLER)]
    for label, beacons in hop_changes.items():
        found.append(Variant(f"hop/{label}", "hop", HopKillerPreset(beacons=beacons)))
    for label, killer in seeded_changes.items():
        found.append(Variant(f"seeded/{label}", "seeded", killer))
    for sweeps in (0.25, 1.0):
        explorer = ExplorerPreset(revisit_sweeps=sweeps)
        found.append(Variant(f"hop/revisit_sweeps={sweeps:g}", "hop", hop, explorer))
        found.append(
            Variant(f"seeded/revisit_sweeps={sweeps:g}", "seeded", SEEDED_KILLER, explorer)
        )
    return found


def combine_variants() -> list[Variant]:
    """The promising changes of the sweep, on top of the adopted ``fresh_for`` of 2 minutes."""
    first = hop_base()
    base = replace(first, fresh_for=2 * MINUTE)
    together = replace(base, max_value=3.0, gamma=0.7, escape=None, killer_repulsion=4.0)
    changes: dict[str, BeaconConfig] = {
        "max_value=3": replace(base, max_value=3.0),
        "gamma=0.7": replace(base, gamma=0.7),
        "escape=off": replace(base, escape=None),
        "killer_repulsion=4": replace(base, killer_repulsion=4.0),
        "max_value=3+gamma=0.7": replace(base, max_value=3.0, gamma=0.7),
        "all four": together,
    }
    found = [
        Variant("hop/fresh_for=2m", None, HopKillerPreset(beacons=base)),
        Variant("hop/fresh_for=1m", "hop/fresh_for=2m", HopKillerPreset(beacons=first)),
        Variant("seeded", None, SEEDED_KILLER),
    ]
    for label, beacons in changes.items():
        found.append(
            Variant(
                f"hop/fresh_for=2m/{label}", "hop/fresh_for=2m", HopKillerPreset(beacons=beacons)
            )
        )
    for minutes in (5, 10):
        killer = replace(SEEDED_KILLER, claim_fresh_for=minutes * MINUTE)
        found.append(Variant(f"seeded/claim_fresh_for={minutes}m", "seeded", killer))
    return found


def refine_variants() -> list[Variant]:
    """One step beyond the values the combine stage adopted on both fields: a longer
    ``fresh_for`` for the hop arm, and a longer claim age for the seeded arm."""
    base = replace(hop_base(), fresh_for=2 * MINUTE)
    found = [Variant("hop/fresh_for=2m", None, HopKillerPreset(beacons=base))]
    for minutes in (3, 4):
        beacons = replace(base, fresh_for=minutes * MINUTE)
        found.append(
            Variant(
                f"hop/fresh_for={minutes}m", "hop/fresh_for=2m", HopKillerPreset(beacons=beacons)
            )
        )
    seeded = replace(SEEDED_KILLER, claim_fresh_for=10 * MINUTE)
    found.append(Variant("seeded/claim_fresh_for=10m", None, seeded))
    longer = replace(SEEDED_KILLER, claim_fresh_for=20 * MINUTE)
    found.append(Variant("seeded/claim_fresh_for=20m", "seeded/claim_fresh_for=10m", longer))
    return found


def hop_best_beacons(default: SwarmScenario) -> BeaconConfig:
    """The beacons of ``HOP_BEST`` in the swarm of ``default``."""
    hop_cells = default.robots.radio_range_m / default.map.cell_size
    return HOP_BEST.beacon_config(hop_cells, default.sweep_time())


def default_field_variants(default: SwarmScenario) -> list[Variant]:
    """``HOP_BEST`` and ``SEEDED_BEST``, and every one-parameter change of them that the
    defaults stage tries on ``default``.

    The ages double from 1 to 16 minutes. The large fields take four times as long to sweep as
    the small ones, so a best age that grows with the field shows up a step or two further
    along the doubling."""
    best = hop_best_beacons(default)
    hop_changes: dict[str, BeaconConfig] = {}
    for minutes in (1, 2, 4, 8, 16):
        hop_changes[f"fresh_for={minutes}m"] = replace(best, fresh_for=minutes * MINUTE)
    for minutes in (1, 2, 4, 8, 16):
        hop_changes[f"source_for={minutes}m"] = replace(best, source_for=minutes * MINUTE)
    for value in (2.0, 3.0):
        hop_changes[f"max_value={value:g}"] = replace(best, max_value=value)
    found = [Variant("hop-best", None, HOP_BEST), Variant("seeded-best", None, SEEDED_BEST)]
    for label, beacons in hop_changes.items():
        killer = replace(HOP_BEST, beacons=beacons, source_sweeps=None)
        found.append(Variant(f"hop-best/{label}", "hop-best", killer))
    for minutes in (1, 2, 4, 8, 16):
        killer = replace(SEEDED_BEST, claim_fresh_for=minutes * MINUTE)
        found.append(Variant(f"seeded-best/claim_fresh_for={minutes}m", "seeded-best", killer))
    return found


def source_sweep_variants(default: SwarmScenario) -> list[Variant]:
    """``HOP_BEST``, with a source of 1.5 sweep times, against its earlier fixed source of 3
    minutes and a source of 1 sweep time, and ``SEEDED_BEST`` next to them: the source-sweeps
    stage, which checked on fresh seeds the formula the defaults stage pointed to."""
    fixed = replace(HOP_BEST, source_sweeps=None)
    return [
        Variant("hop-best/source_for=3m", None, fixed),
        Variant(
            "hop-best/source_for=1sweep",
            "hop-best/source_for=3m",
            replace(fixed, source_sweeps=1.0),
        ),
        Variant("hop-best", "hop-best/source_for=3m", HOP_BEST),
        Variant("seeded-best", None, SEEDED_BEST),
    ]


DEFAULT_FIELDS: dict[str, SwarmScenario] = {
    "river-small": RIVER_SMALL,
    "rivers-large": RIVERS_LARGE,
    "open-small": OPEN_SMALL,
    "open-large": OPEN_LARGE,
}
"""The default scenarios the defaults stage tunes on, by name."""


def tune_on_default_fields(
    args: argparse.Namespace, variants_for: Callable[[SwarmScenario], list[Variant]]
) -> None:
    """A stage on the default scenarios: every variant of ``variants_for`` on every default
    scenario, with its killers swapped for the variant's, and one table per field."""
    scenarios: dict[str, SwarmScenario] = {}
    variants_by_field: dict[str, list[Variant]] = {}
    for field, default in DEFAULT_FIELDS.items():
        variants_by_field[field] = variants_for(default)
        explorers_only = default.without_robots(BodyType.KILLER)
        for variant in variants_by_field[field]:
            scenario = explorers_only.add_robots(variant.killer, DEFAULT_KILLERS)
            scenarios[f"{field}/{variant.name}"] = scenario

    experiment = Experiment(f"tune_arms_{args.stage}")
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    runs = runs_for(scenarios, seeds, SwarmScenario.run_cost)
    results = by_arm(experiment.run(runs, measure_pest_response, args.jobs))

    sections: list[str] = []
    for field, default in DEFAULT_FIELDS.items():
        hours = round(default.until / TimeUnit.HOUR)
        headers = pest_table_headers(hours)
        headers.extend(["Δ infested", "adopt"])
        rows: list[list[str]] = []
        for variant in variants_by_field[field]:
            arm_results = results[f"{field}/{variant.name}"]
            cells = pest_table_cells(variant.name, arm_results, hours)
            difference = "-"
            adopted = "-"
            if variant.base is not None:
                base = results[f"{field}/{variant.base}"]
                difference = paired_difference(
                    [result.infested for result in arm_results],
                    [result.infested for result in base],
                )
                adopted = adopt(arm_results, base)
            cells.extend([difference, adopted])
            rows.append(cells)
        title = (
            f"{field}: {DEFAULT_EXPLORERS} explorers, {DEFAULT_KILLERS} killers, "
            f"{default.pests.initial_pests} pests at the start and "
            f"{default.pests.pests_per_hour:g} per hour, {hours} h, seeds {seeds.start} to "
            f"{seeds.stop - 1}; Δ infested against the arm's base"
        )
        sections.append(f"{title}\n\n{markdown_table(headers, rows)}")
    text = "\n".join(sections)
    print(text)
    experiment.write_text("tables.md", text)


def verify_variants() -> list[Variant]:
    """The frozen presets against the defaults they were tuned from, and against each other."""
    return [
        Variant("seeded", None, SEEDED_KILLER),
        Variant("seeded-best", "seeded", SEEDED_BEST),
        Variant("hop", None, HOP_KILLER),
        Variant("hop-best", "hop", HOP_BEST),
        Variant("hop-best vs seeded-best", "seeded-best", HOP_BEST),
    ]


FIELD_ROBOTS: dict[str, tuple[int, int]] = {"small": (20, 4), "large": (40, 8)}
"""The explorers and killers of each field by default."""
FIELD_MAPS: dict[str, MapConfig] = {"small": SMALL_RIVER_MAP, "large": LARGE_RIVERS_MAP}
"""Each field: the small one with one river, and the large one with two rivers and two docks."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        choices=("sweep", "combine", "refine", "verify", "defaults", "source-sweeps"),
        default="sweep",
    )
    parser.add_argument("--field", choices=tuple(FIELD_ROBOTS), default="small")
    parser.add_argument("--explorers", type=int, help="the field's default if not given")
    parser.add_argument("--killers", type=int, help="the field's default if not given")
    parser.add_argument("--seeds", type=int, default=8, help="runs per arm")
    parser.add_argument("--first-seed", type=int, default=0, help="seeds start here")
    parser.add_argument("--hours", type=int, default=3, help="simulated hours per run")
    parser.add_argument("--warmup-hours", type=int, default=1, help="left out of infested")
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    return parser.parse_args()


def adopt(results: list[PestResponse], base: list[PestResponse]) -> str:
    """Whether the change beats its base by two standard errors of the paired difference."""
    differences: list[float] = []
    for result, reference in zip(results, base, strict=True):
        differences.append(result.infested - reference.infested)
    if len(differences) < 2:
        return "-"
    error = stdev(differences) / len(differences) ** 0.5
    if mean(differences) < -2 * error:
        return "yes"
    return "no"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    if args.stage == "defaults":
        tune_on_default_fields(args, default_field_variants)
        return
    if args.stage == "source-sweeps":
        tune_on_default_fields(args, source_sweep_variants)
        return
    explorers, killers = FIELD_ROBOTS[args.field]
    if args.explorers is not None:
        explorers = args.explorers
    if args.killers is not None:
        killers = args.killers
    rate = PESTS_PER_KILLER_HOUR * killers
    radio = LATTICE_ROBOTS.radio_range_m
    found = variants()
    if args.stage == "combine":
        found = combine_variants()
    elif args.stage == "refine":
        found = refine_variants()
    elif args.stage == "verify":
        found = verify_variants()
    scenarios: dict[str, SwarmScenario] = {}
    for variant in found:
        scenario = moving_explorer_scenario(
            variant.killer, explorers, killers, radio, variant.explorer, FIELD_MAPS[args.field]
        )
        scenarios[variant.name] = scenario.with_hours(args.hours, args.warmup_hours)

    experiment = Experiment(f"tune_arms_{args.stage}_{args.field}")
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    runs = runs_for(scenarios, seeds, SwarmScenario.run_cost)
    results = by_arm(experiment.run(runs, measure_pest_response, args.jobs))

    headers = pest_table_headers(args.hours)
    headers.extend(["Δ infested", "adopt"])
    rows: list[list[str]] = []
    for variant in found:
        arm_results = results[variant.name]
        cells = pest_table_cells(variant.name, arm_results, args.hours)
        difference = "-"
        adopted = "-"
        if variant.base is not None:
            base = results[variant.base]
            difference = paired_difference(
                [result.infested for result in arm_results], [result.infested for result in base]
            )
            adopted = adopt(arm_results, base)
        cells.extend([difference, adopted])
        rows.append(cells)
    title = (
        f"{args.field} field: {explorers} moving explorers, {killers} killers, {rate:g} pests "
        f"per hour, {radio:g} m radio, {args.hours} h (warm-up {args.warmup_hours} h), seeds "
        f"{seeds.start} to {seeds.stop - 1}; Δ infested against the arm's base"
    )
    text = f"{title}\n\n{markdown_table(headers, rows)}"
    print(text)
    experiment.write_text("tables.md", text)


if __name__ == "__main__":
    main()
