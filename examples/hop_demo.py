"""Killers find their way to pests by hop-based beacons, on the console.

The pest experiment of ``pest_demo.py``, but news of a pest travels only robot to robot as a
``PestBeacon`` with a hop count. A robot that senses a pest starts a beacon at hop 0, every
robot that hears one passes it on one hop further, and a robot that finds the pest gone passes
on a cleared beacon. Each killer follows the live beacon with the fewest hops that no other
killer is already heading for, along the grid potential field seeded at that pest's cell, and
explores otherwise. With ``--navigation gradient`` a scout that finds a pest stays on it
as an anchor and killers drive down the hop gradient from neighbour to neighbour, with no pest
position in any message (see ``hop_explainer.py``). With ``--navigation memory`` killers
instead learn pests from their peers' memory snapshots, as in ``pest_demo.py``.
``HopPestScenario`` describes the run.

Red cells are the pests in ground truth, trails show where the robots went, and dashed lines
the target each robot is heading for. The panel shows, per killer, its goal and the beacon it
follows.

Run from the repository root with:
    uv run python examples/hop_demo.py

The console's controls work as in ``scenario_viewer.py``: pause, step, change the speed, inject
a pest, inspect a robot. Record a video instead (mp4 through ffmpeg, or a gif):
    uv run python examples/hop_demo.py --save videos/hop.mp4

Compare hop-based navigation with memory sharing, without a window:
    uv run python examples/hop_demo.py --compare --hours 3 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import dataclass, replace

from swarm_field.console import (
    InteractiveConsole,
    add_session_arguments,
    format_time,
    save_session,
    show_window,
)
from swarm_field.contracts import BodyType, TimeUnit, Treat, WorldSnapshot
from swarm_field.experiments import NAVIGATIONS, HopPestScenario, LiveRun
from swarm_field.swarm import GradientAgent, HopAgent, chosen_beacon, chosen_gradient
from swarm_field.visualization import MatplotlibRenderer

DEFAULT_SPEED = 40.0
"""Twenty simulated seconds per frame."""
TRAIL_LENGTH = 30

GOAL_LABELS: dict[str | None, str] = {
    "explore": "explore",
    "follow-field": "explore",
    "follow-beacon": "to pest",
    "follow-gradient": "to pest",
    "anchor": "anchor",
    "treat-infestation": "to pest",
    None: "idle",
}


def parse_args() -> argparse.Namespace:
    """The command line; the scenario's settings default to those of ``HopPestScenario``."""
    defaults = HopPestScenario()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--navigation", choices=NAVIGATIONS, default=defaults.navigation)
    parser.add_argument("--scouts", type=int, default=defaults.scouts)
    parser.add_argument("--killers", type=int, default=defaults.killers)
    parser.add_argument(
        "--radio", type=float, default=defaults.radio_range_m, help="radio range in metres"
    )
    parser.add_argument(
        "--killer-speed", type=float, default=defaults.speed_mps_ground, help="in m/s"
    )
    parser.add_argument("--max-hops", type=int, default=defaults.hop.max_hops)
    parser.add_argument("--pests-per-hour", type=float, default=defaults.pests_per_hour)
    add_session_arguments(parser, default_speed=DEFAULT_SPEED, recording=False)
    parser.add_argument(
        "--compare", action="store_true", help="print hop against memory instead of a window"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="seeds for --compare")
    parser.add_argument(
        "--radios", type=float, nargs="+", default=[50.0, 300.0], help="radio ranges for --compare"
    )
    return parser.parse_args()


def scenario_from(args: argparse.Namespace) -> HopPestScenario:
    """The scenario the command line describes."""
    defaults = HopPestScenario()
    return HopPestScenario(
        navigation=args.navigation,
        scouts=args.scouts,
        killers=args.killers,
        radio_range_m=args.radio,
        speed_mps_ground=args.killer_speed,
        hop=replace(defaults.hop, max_hops=args.max_hops),
        pests_per_hour=args.pests_per_hour,
        until=int(_hours(args) * TimeUnit.HOUR),
        seed=_seed(args, defaults),
    )


def main() -> None:
    args = parse_args()
    scenario = scenario_from(args)
    if args.compare:
        logging.disable(logging.WARNING)
        compare(scenario, args.radios, args.seeds)
        return
    run = LiveRun.from_config(scenario.into_experiment())
    killer_goals: Counter[str] = Counter()

    def panel(snapshot: WorldSnapshot) -> list[str]:
        return panel_lines(run, scenario, snapshot, killer_goals)

    console = InteractiveConsole(
        run,
        MatplotlibRenderer((12.0, 8.0), show_targets=False),
        speed=args.speed,
        panel=panel,
        trail_length=TRAIL_LENGTH,
        show_targets=True,
    )
    if args.save is None:
        show_window(console)
        return
    frames = save_session(console, args.save)
    print(f"Wrote {frames} frames up to {format_time(run.now)} to {args.save}")


def _hours(args: argparse.Namespace) -> float:
    """The simulated hours of the command line, 3 unless given."""
    if args.hours is None:
        return 3.0
    return float(args.hours)


def _seed(args: argparse.Namespace, defaults: HopPestScenario) -> int:
    """The seed of the command line, ``HopPestScenario``'s unless given."""
    if args.seed is None:
        return defaults.seed
    return int(args.seed)


@dataclass(frozen=True)
class Outcome:
    """What one run did about its pests.

    Attributes:
        spawned: Pests that appeared.
        cleared: Pests that went, nearly all of them treated.
        left: Pests still there at the end.
        mean_infested: Infested cells, averaged over the run's minutes.
        to_pest: The share of the killers' time spent heading for or treating a pest.
    """

    spawned: int
    cleared: int
    left: int
    mean_infested: float
    to_pest: float


def outcome_of(scenario: HopPestScenario) -> Outcome:
    """Run ``scenario`` and measure it once a simulated minute."""
    run = LiveRun.from_config(scenario.into_experiment())
    world = run.context.world
    seen: set[tuple[int, int]] = set()
    spawned = 0
    cleared = 0
    infested = 0
    samples = 0
    goals: Counter[str] = Counter()
    for minute in range(scenario.until // TimeUnit.MINUTE + 1):
        run.advance_to(minute * TimeUnit.MINUTE)
        now = {cell for cell, state in world.cells.items() if state.infestation > 0}
        spawned += len(now - seen)
        cleared += len(seen - now)
        seen = now
        infested += len(now)
        samples += 1
        for robot_id, agent in run.agents().items():
            if world.robots[robot_id].body is not BodyType.KILLER:
                continue
            goal = GOAL_LABELS.get(agent.inspect().current_goal, "other")
            if isinstance(world.robots[robot_id].current_action, Treat):
                goal = "to pest"
            goals[goal] += 1
    total = max(1, sum(goals.values()))
    return Outcome(
        spawned=spawned,
        cleared=cleared,
        left=len(seen),
        mean_infested=infested / samples,
        to_pest=goals["to pest"] / total,
    )


def compare(scenario: HopPestScenario, radios: list[float], seeds: list[int]) -> None:
    """Print hop-based navigation against memory sharing, per radio range and seed."""
    print("radio  navigation  seed  spawned  cleared  left  mean infested  killers to pest")
    for radio in radios:
        for navigation in NAVIGATIONS:
            for seed in seeds:
                point = replace(scenario, navigation=navigation, radio_range_m=radio, seed=seed)
                result = outcome_of(point)
                print(
                    f"{radio:5.0f}  {navigation:10s}  {seed:4d}  {result.spawned:7d}  "
                    f"{result.cleared:7d}  {result.left:4d}  {result.mean_infested:13.2f}  "
                    f"{result.to_pest:15.1%}"
                )


def panel_lines(
    run: LiveRun, scenario: HopPestScenario, snapshot: WorldSnapshot, killer_goals: Counter[str]
) -> list[str]:
    """The run's settings, its measures, and what each killer does and follows.

    ``killer_goals`` counts the killers' goals over the frames drawn so far; frames are evenly
    spaced in time, so it says how the killers spent their time.
    """
    lines = [
        f"time          {_clock(snapshot.time)}",
        f"navigation    {scenario.navigation}",
        f"robots        {scenario.scouts} scouts, {scenario.killers} killers",
        f"radio range   {scenario.radio_range_m:.0f} m",
        f"killer speed  {scenario.speed_mps_ground:.1f} m/s",
        f"pest rate     {scenario.pests_per_hour:.1f} per hour",
        "",
    ]
    latest = run.metrics.latest()
    if latest is not None:
        lines.extend(
            [
                f"infested      {latest.targets_remaining} cells",
                f"detection     {_minutes(latest.detection_latency_ms)} (mean)",
                f"covered       {latest.coverage:6.1%}",
                "",
            ]
        )
    agents = run.agents()
    rows: list[str] = []
    for robot in snapshot.robots:
        if robot.body is not BodyType.KILLER:
            continue
        agent = agents[robot.robot_id]
        goal = GOAL_LABELS.get(agent.inspect().current_goal, "other")
        killer_goals[goal] += 1
        rows.append(f"{robot.robot_id:6d}  {goal:8s} {_followed(agent, snapshot.time)}")
    total = sum(killer_goals.values())
    for goal, count in sorted(killer_goals.items()):
        lines.append(f"killer time   {count / total:6.1%} {goal}")
    lines.append("")
    lines.append("killer  goal     beacon followed")
    lines.extend(rows)
    return lines


def _followed(agent: object, now: int) -> str:
    """The pest a hop killer follows and its hops, or its gradient parent; ``-`` if none."""
    if isinstance(agent, GradientAgent):
        entry = chosen_gradient(agent.gradients, now)
        if entry is None:
            return "-"
        return f"{entry.hops} hops, to robot {entry.parent}"
    if not isinstance(agent, HopAgent):
        return "-"
    beacon = chosen_beacon(agent.beacons, agent.memory, agent.robot_id, now)
    if beacon is None:
        return "-"
    return f"{beacon.cell} {beacon.hops} hops"


def _clock(time: float) -> str:
    """A time as hours and minutes."""
    minutes = int(time // TimeUnit.MINUTE)
    return f"{minutes // 60}:{minutes % 60:02d}"


def _minutes(time: float | None) -> str:
    if time is None:
        return "    -"
    return f"{time / TimeUnit.MINUTE:5.1f} min"


if __name__ == "__main__":
    main()
