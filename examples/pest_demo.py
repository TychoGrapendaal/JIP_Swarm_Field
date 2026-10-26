"""Explorers find pests and killers treat them while pests keep appearing, on the console.

Five explorers and three killers start side by side in a corner of a field of 10 m cells with a
river and two rocks. Each is a real ``ComposedAgent`` with a ``UtilityArbiter`` on the real
simulation loop: explorers follow the EXPLORE field, and seeded killers treat the pests they
know of on the TREAT field. With ``--hop`` the killers are hop killers instead: they follow the
hop beacons that every robot relays, from the explorer that sees a pest. Pests spawn at random
free cells over time, and robots share what they know only within radio range. A
``SwarmScenario`` describes the run, and ``swarm_field.runner`` assembles and runs it.

Red cells are the pests in ground truth, trails show where the robots went, and dashed lines the
target each robot is heading for. The panel shows the measures of ``MetricsCollector``, how the
killers spent their time, and what each killer does and knows. The console's controls work as in
``scenario_viewer.py``: pause, step, change the speed, inject a pest, inspect a robot.

Run from the repository root with:
    uv run python examples/pest_demo.py

Record a video instead (mp4 through ffmpeg, or a gif):
    uv run python examples/pest_demo.py --save videos/pests.mp4
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace

from swarm_field.console import (
    InteractiveConsole,
    add_session_arguments,
    format_time,
    save_session,
    show_window,
)
from swarm_field.contracts import BodyType, Time, TimeUnit, WorldSnapshot
from swarm_field.experiments import (
    EXPLORER,
    FAST_PESTS,
    HOP_KILLER,
    PESTS,
    SEEDED_KILLER,
    LiveRun,
    MapConfig,
    RobotConfig,
    RobotPreset,
    SwarmScenario,
)
from swarm_field.visualization import DEFAULT_TRAIL_LENGTH, MatplotlibRenderer

DEFAULT_SPEED = 40.0
"""Twenty simulated seconds per frame."""
GOAL_LABELS: dict[str | None, str] = {
    "treat-infestation": "treat",
    "treat-hop": "treat",
    None: "idle",
}


def parse_args() -> argparse.Namespace:
    """The command line; the swarm defaults to that of ``PESTS``, the settings to its parts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--explorers", type=int, default=PESTS.count(EXPLORER.name))
    parser.add_argument("--killers", type=int, default=PESTS.killers)
    parser.add_argument("--hop", action="store_true", help="hop killers instead of seeded ones")
    parser.add_argument(
        "--radio", type=float, default=RobotConfig().radio_range_m, help="radio range in metres"
    )
    parser.add_argument("--pests-per-hour", type=float, default=FAST_PESTS.pests_per_hour)
    parser.add_argument(
        "--spread",
        type=float,
        default=FAST_PESTS.spread_probability,
        help="pest spread probability",
    )
    add_session_arguments(parser, default_speed=DEFAULT_SPEED, recording=False)
    return parser.parse_args()


def build_scenario(args: argparse.Namespace) -> SwarmScenario:
    """The scenario of the command line, on the river field with one dock (``MapConfig()``), with
    ``FAST_PESTS`` changed by it."""
    killer: RobotPreset = SEEDED_KILLER
    if args.hop:
        killer = HOP_KILLER
    pests = replace(FAST_PESTS, pests_per_hour=args.pests_per_hour, spread_probability=args.spread)
    scenario = (
        SwarmScenario()
        .with_map(MapConfig())
        .with_pests(pests)
        .with_robot_config(RobotConfig(radio_range_m=args.radio))
        .add_robots(EXPLORER, args.explorers)
        .add_robots(killer, args.killers)
    )
    if args.seed is not None:
        scenario = replace(scenario, seed=args.seed)
    if args.hours is not None:
        scenario = replace(scenario, until=int(args.hours * TimeUnit.HOUR))
    return scenario


class KillerTime:
    """How long the killers have spent on each goal, in simulated time.

    Weighted by the time between two observations, so a paused, stepped or sped-up console
    counts every goal for as long as the killers actually pursued it.
    """

    def __init__(self) -> None:
        self._time_by_goal: Counter[str] = Counter()
        self._goals: list[str] = []
        self._seen_at: Time = 0

    def observe(self, now: Time, goals: list[str]) -> None:
        """Credit the goals seen last with the time since then, and remember ``goals``."""
        elapsed = now - self._seen_at
        if elapsed > 0:
            for goal in self._goals:
                self._time_by_goal[goal] += elapsed
            self._seen_at = now
        self._goals = goals

    def shares(self) -> list[tuple[str, float]]:
        """Each goal and its share of the killers' time so far, by goal name."""
        total = sum(self._time_by_goal.values())
        if total == 0:
            return []
        return [(goal, time / total) for goal, time in sorted(self._time_by_goal.items())]


def main() -> None:
    args = parse_args()
    scenario = build_scenario(args)
    run = LiveRun.from_config(scenario.into_experiment())
    killer_time = KillerTime()

    def panel(snapshot: WorldSnapshot) -> list[str]:
        return panel_lines(run, scenario, snapshot, killer_time)

    console = InteractiveConsole(
        run,
        MatplotlibRenderer((12.0, 8.0), show_targets=False),
        speed=args.speed,
        panel=panel,
        trail_length=DEFAULT_TRAIL_LENGTH,
        show_targets=True,
    )
    if args.save is None:
        show_window(console)
        return
    frames = save_session(console, args.save)
    print(f"Wrote {frames} frames up to {format_time(run.now)} to {args.save}")


def panel_lines(
    run: LiveRun, scenario: SwarmScenario, snapshot: WorldSnapshot, killer_time: KillerTime
) -> list[str]:
    """The run's settings, its measures, and what the killers do and know."""
    killers = "seeded"
    if scenario.count(HOP_KILLER.name) > 0:
        killers = "hop"
    lines = [
        f"time          {_clock(snapshot.time)}",
        f"robots        {scenario.count(EXPLORER.name)} explorers, {scenario.killers} killers",
        f"killers       {killers}",
        f"radio range   {scenario.robots.radio_range_m:.0f} m",
        f"pest rate     {scenario.pests.pests_per_hour:.1f} per hour",
        "",
    ]
    latest = run.metrics.latest()
    if latest is not None:
        lines.extend(
            [
                f"infested      {latest.targets_remaining} cells",
                f"detection     {_minutes(latest.detection_latency_ms)} (mean)",
                f"covered       {latest.coverage:6.1%}",
                f"idleness      {latest.mean_idleness_ms / TimeUnit.MINUTE:5.1f} min (mean)",
                f"redundancy    {latest.redundancy:6.1%}",
                "",
            ]
        )
    agents = run.agents()
    goals: list[str] = []
    rows: list[str] = []
    for robot in snapshot.robots:
        if robot.body is not BodyType.KILLER:
            continue
        agent = agents[robot.robot_id]
        goal = GOAL_LABELS.get(agent.inspect().current_goal, "other")
        goals.append(goal)
        known = 0
        for pest in agent.memory.infestations():
            if pest.severity > 0:
                known += 1
        # A hop killer keeps no map, so it knows of no pest cell at all.
        shown = f"{known:5d}"
        if not agent.memory.keeps_map:
            shown = "    -"
        rows.append(f"{robot.robot_id:6d}  {goal:8s} {shown}")
    killer_time.observe(snapshot.time, goals)
    for goal, share in killer_time.shares():
        lines.append(f"killer time   {share:6.1%} {goal}")
    lines.append("")
    lines.append("killer  goal     known pests")
    lines.extend(rows)
    return lines


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
