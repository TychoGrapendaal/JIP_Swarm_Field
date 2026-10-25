"""Study whether hop killers find a dock without a map, against seeded killers (#70, step 5).

Question: on the small river field with moving explorers only (the scenario of
``studies/explorer_hop_killers.py``), do hop killers reach the dock down the DOCK beacons the dock
broadcasts, to restock their doses and to charge, as reliably and as fast as seeded killers
that plan over their map? Two conditions make the killers go: ``doses``, where every killer
carries 10 doses and the battery is unlimited, so killers restock; and ``battery``, where every
battery holds half a standard charge and doses are unlimited, so robots charge. The dock is at
``(0, 0)``, west of the river, so a killer on the east bank must find the bridge.

A trip starts when ``return-to-dock`` or ``resupply`` wins a killer's decision and ends when the
killer stands on a dock. The table reports trips started and completed, their mean and longest
duration, the trips still under way at the end, the killers that died of an empty battery and
the share of killer time without doses, next to the pest response. Not part of ``pytest``; run
it with:

    uv run python -m studies.hop_dock_return --seeds 8
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field, replace
from statistics import mean

from studies.explorer_hop_killers import KILLERS, PESTS_PER_KILLER_HOUR, explorer_scenario
from swarm_field.contracts import FailureMode, RobotId, Time, TimeUnit
from swarm_field.experiments import (
    RIVER_SMALL,
    Experiment,
    PestResponse,
    PestRunSpec,
    Run,
    arm_name,
    by_arm,
    markdown_table,
    mean_and_error,
    pest_table_cells,
    pest_table_headers,
    runs_for,
)
from swarm_field.experiments.pest_response import MeasuredPestRun
from swarm_field.swarm import Resupply, ReturnToDock

DOCK_GOALS = frozenset({ReturnToDock.name, Resupply.name})
"""The behaviours whose win sends a killer to a dock."""
CONDITIONS: dict[str, tuple[float, int]] = {"doses": (1000.0, 10), "battery": (0.5, 1000)}
"""Each condition's battery capacity and doses per killer."""


@dataclass
class DockTrips:
    """The killers' trips to a dock in one run.

    Attributes:
        durations: The minutes each completed trip took.
        open_trips: Trips still under way at the end.
        empty_samples: Killer samples without doses left.
        killer_samples: Killer samples in all.
    """

    durations: list[float] = field(default_factory=list)
    open_trips: int = 0
    empty_samples: int = 0
    killer_samples: int = 0


@dataclass(frozen=True)
class DockResponse:
    """What one run measured: the pest response, and how the killers went to the dock.

    Attributes:
        pest: The standard pest measurement.
        trips: The completed trips' durations in minutes.
        open_trips: Trips still under way at the end.
        dead_killers: Killers that failed with an empty battery.
        empty_share: The share of killer time without doses left.
    """

    pest: PestResponse
    trips: tuple[float, ...]
    open_trips: int
    dead_killers: int
    empty_share: float


def measure_dock_response(run: Run[PestRunSpec]) -> DockResponse:
    """Make one run and measure it, also following every killer's trips to a dock."""
    measured = MeasuredPestRun(run)
    world = measured.live.context.world
    trips = DockTrips()
    started: dict[RobotId, Time] = {}
    while not measured.finished:
        measured.advance()
        now = measured.live.now
        agents = measured.live.agents()
        for robot_id in measured.killers:
            robot = world.robots[robot_id]
            if robot.failed:
                started.pop(robot_id, None)
                continue
            trips.killer_samples += 1
            if robot.consumable <= 0.0:
                trips.empty_samples += 1
            on_dock = robot.position in world.docks
            if robot_id in started and on_dock:
                trips.durations.append((now - started.pop(robot_id)) / TimeUnit.MINUTE)
            goal = agents[robot_id].inspect().current_goal
            if goal in DOCK_GOALS and not on_dock and robot_id not in started:
                started[robot_id] = now
    trips.open_trips = len(started)
    dead = 0
    for robot_id in measured.killers:
        if world.robots[robot_id].failure_mode is FailureMode.DEAD_BATTERY:
            dead += 1
    return DockResponse(
        pest=measured.response(),
        trips=tuple(trips.durations),
        open_trips=trips.open_trips,
        dead_killers=dead,
        empty_share=trips.empty_samples / max(trips.killer_samples, 1),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--explorers", type=int, default=20)
    parser.add_argument("--killers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=8, help="runs per arm, seeds 0, 1, ...")
    parser.add_argument("--hours", type=int, default=6, help="simulated hours per run")
    parser.add_argument("--warmup-hours", type=int, default=1, help="left out of infested")
    parser.add_argument("--jobs", type=int, help="processes; one per CPU if not given")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    rate = PESTS_PER_KILLER_HOUR * args.killers
    radio = RIVER_SMALL.robots.radio_range_m
    specs: dict[str, PestRunSpec] = {}
    for condition, (capacity, doses) in CONDITIONS.items():
        for name, killer in KILLERS.items():
            scenario = explorer_scenario(killer, args.explorers, args.killers, radio)
            scenario = scenario.with_robot_config(
                replace(scenario.robots, battery_capacity=capacity)
            ).with_pests(replace(scenario.pests, treatment_doses=doses))
            spec = PestRunSpec(scenario, args.hours, rate, args.warmup_hours)
            specs[arm_name(condition, name)] = spec

    experiment = Experiment("hop_dock_return")
    runs = runs_for(specs, range(args.seeds), PestRunSpec.cost)
    results = by_arm(experiment.run(runs, measure_dock_response, args.jobs))

    headers = pest_table_headers(args.hours)
    headers.extend(["trips", "done", "trip min", "longest", "open", "dead", "no doses"])
    rows: list[list[str]] = []
    for arm, arm_results in results.items():
        cells = pest_table_cells(arm, [result.pest for result in arm_results], args.hours)
        durations: list[float] = []
        for result in arm_results:
            durations.extend(result.trips)
        opened = mean(len(result.trips) + result.open_trips for result in arm_results)
        cells.append(f"{opened:.1f}")
        cells.append(f"{mean(len(result.trips) for result in arm_results):.1f}")
        trip_minutes = "-"
        longest = "-"
        if durations:
            trip_minutes = mean_and_error(durations)
            longest = f"{max(durations):.1f}"
        cells.extend([trip_minutes, longest])
        cells.append(f"{mean(result.open_trips for result in arm_results):.2f}")
        cells.append(f"{mean(result.dead_killers for result in arm_results):.2f}")
        cells.append(f"{mean(result.empty_share for result in arm_results):.3f}")
        rows.append(cells)
    title = (
        f"river-small, dock at (0, 0): {args.explorers} explorers, {args.killers} killers, "
        f"{radio:g} m radio, {rate:g} pests per hour, {args.hours} h (warm-up "
        f"{args.warmup_hours} h), mean over {args.seeds} seeds; doses: 10 doses and unlimited "
        "battery; battery: half a standard charge and unlimited doses"
    )
    text = f"{title}\n\n{markdown_table(headers, rows)}"
    print(text)
    experiment.write_text("tables.md", text)


if __name__ == "__main__":
    main()
