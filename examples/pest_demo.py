"""Scouts explore and killers explore and treat while pests keep appearing, on the console.

Five scouts and three killers start side by side in a corner of a field of 10 m cells with a
river and two rocks. Each is a real ``ComposedAgent`` with a ``UtilityArbiter`` on the real
simulation loop: scouts follow the EXPLORE field, killers weigh it against ``TreatInfestation``
on the TREAT field. With ``--killers-treat-only`` killers do not explore and wait until they hear
of a pest. Pests spawn at random free cells over time, and robots share what they know only
within radio range. ``PestScenario`` describes the run, and ``swarm_field.runner`` assembles and
runs it.

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
from swarm_field.experiments import LiveRun, PestScenario
from swarm_field.visualization import MatplotlibRenderer

DEFAULT_SPEED = 40.0
"""Twenty simulated seconds per frame."""
TRAIL_LENGTH = 30
GOAL_LABELS: dict[str | None, str] = {
    "follow-field": "explore",
    "treat-infestation": "treat",
    None: "idle",
}


def parse_args() -> argparse.Namespace:
    """The command line; the scenario's settings default to those of ``PestScenario``."""
    defaults = PestScenario()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scouts", type=int, default=defaults.scouts)
    parser.add_argument("--killers", type=int, default=defaults.killers)
    parser.add_argument("--killers-treat-only", action="store_true", help="killers do not explore")
    parser.add_argument(
        "--radio", type=float, default=defaults.radio_range_m, help="radio range in metres"
    )
    parser.add_argument("--pests-per-hour", type=float, default=defaults.pests_per_hour)
    parser.add_argument(
        "--spread",
        type=float,
        default=defaults.spread_probability,
        help="pest spread probability",
    )
    add_session_arguments(parser, default_speed=DEFAULT_SPEED, recording=False)
    return parser.parse_args()


def build_scenario(args: argparse.Namespace) -> PestScenario:
    """The scenario of the command line, on ``PestScenario``'s seed and horizon unless given."""
    scenario = PestScenario(
        scouts=args.scouts,
        killers=args.killers,
        killers_explore=not args.killers_treat_only,
        radio_range_m=args.radio,
        pests_per_hour=args.pests_per_hour,
        spread_probability=args.spread,
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
        trail_length=TRAIL_LENGTH,
        show_targets=True,
    )
    if args.save is None:
        show_window(console)
        return
    frames = save_session(console, args.save)
    print(f"Wrote {frames} frames up to {format_time(run.now)} to {args.save}")


def panel_lines(
    run: LiveRun, scenario: PestScenario, snapshot: WorldSnapshot, killer_time: KillerTime
) -> list[str]:
    """The run's settings, its measures, and what the killers do and know."""
    killers = "explore + treat"
    if not scenario.killers_explore:
        killers = "treat only"
    lines = [
        f"time          {_clock(snapshot.time)}",
        f"robots        {scenario.scouts} scouts, {scenario.killers} killers",
        f"killers       {killers}",
        f"radio range   {scenario.radio_range_m:.0f} m",
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
        rows.append(f"{robot.robot_id:6d}  {goal:8s} {known:5d}")
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
