"""Study how far the robots' radio reaches against how fast the swarm finds and treats pests.

Question: in the pest experiment (``PESTS``: explorers and seeded killers), how do the time to
treat a pest, the infested cells, the coverage and the amount of communication change with the
radio range? Every range is run with the same seeds, so two rows differ only in the range.

The runs are measured with ``measure_pest_response`` and run in parallel by ``Experiment``, which
writes them and the table to a new folder in ``output/``. The table is the mean over the seeds,
with its standard error. Run from the repository root with:

    uv run python -m studies.radio_range --ranges 50 150 300 --seeds 3 --hours 2
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from swarm_field.experiments import (
    PESTS,
    Experiment,
    PestRunSpec,
    arm_name,
    by_arm,
    markdown_table,
    mean_and_error,
    measure_pest_response,
    runs_for,
)

DEFAULT_RANGES = (50.0, 150.0, 300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ranges", type=float, nargs="+", default=list(DEFAULT_RANGES), help="radio ranges in m"
    )
    parser.add_argument("--seeds", type=int, default=3, help="runs per range, seeds 0, 1, ...")
    parser.add_argument("--hours", type=int, default=2, help="simulated hours per run")
    parser.add_argument("--warmup-hours", type=int, default=1, help="left out of infested")
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    return parser.parse_args()


def radio_range_spec(radio_range_m: float, hours: int, warmup_hours: int) -> PestRunSpec:
    """The pest experiment with every robot's radio reaching ``radio_range_m``."""
    scenario = PESTS.with_robot_config(replace(PESTS.robots, radio_range_m=radio_range_m))
    return PestRunSpec(scenario, hours, PESTS.pests.pests_per_hour, warmup_hours)


def main() -> None:
    args = parse_args()
    specs: dict[str, PestRunSpec] = {}
    for radio_range in args.ranges:
        specs[arm_name("radio", m=radio_range)] = radio_range_spec(
            radio_range, args.hours, args.warmup_hours
        )
    experiment = Experiment("radio_range")
    runs = runs_for(specs, range(args.seeds), PestRunSpec.cost)
    results = by_arm(experiment.run(runs, measure_pest_response, args.jobs))

    headers = ["radio", "response (min)", "infested", "coverage", "idle (min)", "msgs"]
    rows: list[list[str]] = []
    for arm, arm_results in results.items():
        rows.append(
            [
                arm,
                mean_and_error([result.response_minutes for result in arm_results]),
                mean_and_error([result.infested for result in arm_results]),
                mean_and_error([result.coverage for result in arm_results]),
                mean_and_error([result.idleness_minutes for result in arm_results]),
                mean_and_error([result.messages_per_robot_hour for result in arm_results]),
            ]
        )
    table = markdown_table(headers, rows)
    print(table)
    experiment.write_text("tables.md", table)


if __name__ == "__main__":
    main()
