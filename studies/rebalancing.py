"""Role rebalancing (#64): fixed roles against robots that switch between explorer and killer.

Both arms of ``rebalancing_configurations`` run for each pest response, with the same seeds:
``seeded`` on ``REBALANCING`` (eight explorers and six seeded killers on the river field, a dock
in every corner, 20 pests an hour) and ``hop`` on ``REBALANCING_HOP``, the same with hop killers.
A run is measured with ``measure_pest_response``, whose columns the table repeats in part, and
adds:

- ``killers``: the robots that were killers, averaged over the run, and the fewest and most;
- ``switches``: role switches over the whole swarm;
- ``Δ infested``: the paired difference in infested cells to the fixed roles of the same pest
  response, over the same seeds.

The runs, their results and the table go to a new folder in ``output/``. Not part of
``pytest``; run it with:

    uv run python -m studies.rebalancing --seeds 8
    uv run python -m studies.rebalancing --seeds 8 --response hop
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean

from swarm_field.contracts import BodyType
from swarm_field.experiments import (
    REBALANCING,
    REBALANCING_HOP,
    Experiment,
    MeasuredPestRun,
    PestResponse,
    PestRunSpec,
    Run,
    SwarmScenario,
    by_arm,
    markdown_table,
    mean_and_error,
    paired_difference,
    rebalancing_configurations,
    runs_for,
)

SCENARIOS: dict[str, SwarmScenario] = {"seeded": REBALANCING, "hop": REBALANCING_HOP}
"""The fixed-role scenario of each pest response, by the response's name."""
BASELINE = "fixed"
"""The arm of each response that the others of that response are compared with, seed by seed."""


@dataclass(frozen=True)
class RebalancingResult:
    """What one run measured.

    Attributes:
        pests: The standard pest-response measures.
        killers: The robots that were killers, averaged over the run's steps.
        fewest_killers: The fewest killers at any step.
        most_killers: The most killers at any step.
        switches: Role switches over the whole swarm.
    """

    pests: PestResponse
    killers: float
    fewest_killers: int
    most_killers: int
    switches: int


def measure_rebalancing(run: Run[PestRunSpec]) -> RebalancingResult:
    """Make one run and measure its pest response and its roles; a run function."""
    measured = MeasuredPestRun(run)
    killers: list[int] = []
    while not measured.finished:
        measured.advance()
        robots = measured.live.context.world.robots.values()
        killers.append(sum(1 for robot in robots if robot.body is BodyType.KILLER))
    robots = measured.live.context.world.robots.values()
    return RebalancingResult(
        pests=measured.response(),
        killers=mean(killers),
        fewest_killers=min(killers),
        most_killers=max(killers),
        switches=sum(robot.role_switches for robot in robots),
    )


def table(results: dict[str, list[RebalancingResult]]) -> str:
    """The arms as Markdown, by pest response, the fixed roles first."""
    headers = ["arm", "infested", "Δ infested", "treated", "response", "open", "coverage", "idle"]
    headers.extend(["killers", "killers range", "switches"])
    rows: list[list[str]] = []
    for arm, arm_results in results.items():
        pests = [result.pests for result in arm_results]
        response, roles = arm.split("/")
        difference = "-"
        if roles != BASELINE:
            baseline = [result.pests.infested for result in results[f"{response}/{BASELINE}"]]
            infested = [result.infested for result in pests]
            difference = paired_difference(infested, baseline)
        fewest = min(result.fewest_killers for result in arm_results)
        most = max(result.most_killers for result in arm_results)
        rows.append(
            [
                arm,
                mean_and_error([result.infested for result in pests]),
                difference,
                mean_and_error([float(result.treated) for result in pests]),
                mean_and_error([result.response_minutes for result in pests]),
                mean_and_error([float(result.open) for result in pests]),
                mean_and_error([result.coverage for result in pests]),
                mean_and_error([result.idleness_minutes for result in pests]),
                mean_and_error([result.killers for result in arm_results]),
                f"{fewest}-{most}",
                mean_and_error([float(result.switches) for result in arm_results]),
            ]
        )
    return markdown_table(headers, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=4, help="seeds 0 to N-1 for every arm")
    parser.add_argument("--hours", type=int, default=3)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument(
        "--response",
        choices=(*SCENARIOS, "both"),
        default="both",
        help="the pest response of the killers",
    )
    args = parser.parse_args()

    responses = list(SCENARIOS)
    if args.response != "both":
        responses = [args.response]
    arms: dict[str, PestRunSpec] = {}
    for response in responses:
        for name, scenario in rebalancing_configurations(SCENARIOS[response]).items():
            spec = PestRunSpec(scenario, args.hours, scenario.pests.pests_per_hour)
            arms[f"{response}/{name}"] = spec
    experiment = Experiment("rebalancing")
    results = experiment.run(runs_for(arms, range(args.seeds)), measure_rebalancing, args.processes)
    text = table(by_arm(results))
    experiment.write_text("table.md", text + "\n")
    print(text)
    print(f"\nResults in {experiment.directory}")


if __name__ == "__main__":
    main()
