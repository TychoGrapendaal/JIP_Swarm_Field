"""Benchmark pest response on fields with obstacles: four configurations, two scenarios (#70).

Every configuration of ``pest_response_configurations`` runs on ``RIVER_SMALL`` and on
``RIVERS_LARGE``, with the same seeds. A run is measured with ``measure_pest_response``, whose
standard columns the table repeats, and adds:

- ``treated %``: the share of the pests that appeared and were treated, ``treated / (treated +
  open)``;
- ``Δ infested``: the paired difference in infested cells to ``seeded`` over the same seeds, with
  its standard error.

The runs, their results and the tables go to a new folder in ``output/``.

``--calibrate`` runs only ``hop-tuned`` at the pest rates given, to pick the rate at which it
treats 85-90 % of the pests. Not part of ``pytest``; run it with:

    uv run python -m tests.benchmark_obstacles --seeds 8
    uv run python -m tests.benchmark_obstacles --scenario river-small --calibrate 4 6 8 --seeds 2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

from swarm_field.experiments import (
    RIVER_SMALL,
    RIVERS_LARGE,
    Experiment,
    PestResponse,
    PestRunSpec,
    PestScenario,
    arm_name,
    by_arm,
    markdown_table,
    measure_pest_response,
    paired_difference,
    pest_response_configurations,
    pest_table_cells,
    pest_table_headers,
    runs_for,
    treated_share,
)

SCENARIOS: dict[str, PestScenario] = {"river-small": RIVER_SMALL, "rivers-large": RIVERS_LARGE}
"""The scenarios by the name used in arm names and tables."""
BASELINE = "seeded"
"""The configuration every other is compared with, seed by seed."""


@dataclass(frozen=True)
class Section:
    """One table: a title, and its arms by the name of their row.

    Attributes:
        title: What the table's runs share, written above it.
        rows: The arm name of each row, by the row's label.
    """

    title: str
    rows: dict[str, str]


def section_table(section: Section, results: dict[str, list[PestResponse]], hours: int) -> str:
    """One section as Markdown: the standard columns, the treated share, and Δ infested."""
    headers = pest_table_headers(hours, first="config")
    headers.extend(["treated %", "Δ infested"])
    baseline: list[float] | None = None
    if BASELINE in section.rows:
        baseline = [result.infested for result in results[section.rows[BASELINE]]]
    rows: list[list[str]] = []
    for label, arm in section.rows.items():
        arm_results = results[arm]
        difference = "-"
        if baseline is not None and label != BASELINE:
            infested = [result.infested for result in arm_results]
            difference = paired_difference(infested, baseline)
        cells = pest_table_cells(label, arm_results, hours)
        cells.extend([treated_share(arm_results), difference])
        rows.append(cells)
    return f"{section.title}\n\n{markdown_table(headers, rows)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=[*SCENARIOS, "both"], default="both")
    parser.add_argument("--seeds", type=int, default=8, help="runs per configuration")
    parser.add_argument("--hours", type=int, default=3, help="simulated hours per run")
    parser.add_argument("--warmup-hours", type=int, default=1, help="left out of infested")
    parser.add_argument("--jobs", type=int, default=22, help="processes")
    parser.add_argument("--killers", type=int, help="override the scenario's killers")
    parser.add_argument(
        "--calibrate", type=float, nargs="+", help="run only hop-tuned at these pest rates"
    )
    args = parser.parse_args()

    names = list(SCENARIOS)
    if args.scenario != "both":
        names = [args.scenario]
    arms: dict[str, PestRunSpec] = {}
    sections: list[Section] = []
    for name in names:
        scenario = SCENARIOS[name]
        if args.killers is not None:
            scenario = replace(scenario, killers=args.killers)
        rates = args.calibrate or [scenario.pests_per_hour]
        for rate in rates:
            configs = pest_response_configurations(scenario)
            if args.calibrate is not None:
                configs = {"hop-tuned": configs["hop-tuned"]}
            rows: dict[str, str] = {}
            for config, configured in configs.items():
                arm = arm_name(name, config, killers=scenario.killers, rate=rate)
                arms[arm] = PestRunSpec(configured, args.hours, rate, args.warmup_hours)
                rows[config] = arm
            title = (
                f"{name}: {scenario.scouts} relays, {scenario.killers} killers, "
                f"{scenario.radio_range_m:g} m radio, {rate:g} pests per hour, "
                f"{args.hours} h (warm-up {args.warmup_hours} h), mean over {args.seeds} seeds"
            )
            sections.append(Section(title, rows))

    experiment = Experiment("obstacle_comparison")
    if args.calibrate is not None:
        experiment = Experiment("obstacle_calibration")
    runs = runs_for(arms, range(args.seeds), PestRunSpec.cost)
    results = by_arm(experiment.run(runs, measure_pest_response, args.jobs))
    tables: list[str] = []
    for section in sections:
        tables.append(section_table(section, results, args.hours))
    text = "\n".join(tables)
    print(text)
    experiment.write_text("tables.md", text)


if __name__ == "__main__":
    main()
