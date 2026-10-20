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

``--smoke`` starts from the base demo (``PestScenario.smoke``): five scouts that only explore and
five killers at 0.5 m/s that only treat. Any other option given changes that setup.

Run from the repository root with:
    uv run python examples/pest_demo.py
    uv run python examples/pest_demo.py --smoke

Record a video instead (mp4 through ffmpeg, or a gif):
    uv run python examples/pest_demo.py --save videos/pests.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from swarm_field.console import (
    InteractiveConsole,
    add_session_arguments,
    format_time,
    save_session,
    show_window,
)
from swarm_field.contracts import BodyType, TimeUnit, WorldSnapshot
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
    """The command line; an option left out keeps the value of the starting scenario."""
    defaults = PestScenario()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke", action="store_true", help="start from the base demo: 5 + 5 robots, one task each"
    )
    parser.add_argument("--scouts", type=int, default=None, help=f"default {defaults.scouts}")
    parser.add_argument("--killers", type=int, default=None, help=f"default {defaults.killers}")
    parser.add_argument("--killers-treat-only", action="store_true", help="killers do not explore")
    parser.add_argument(
        "--killer-speed",
        type=float,
        default=None,
        help=f"killers' speed in m/s (default {defaults.speed_mps_ground:g})",
    )
    parser.add_argument(
        "--radio",
        type=float,
        default=None,
        help=f"radio range in metres (default {defaults.radio_range_m:g})",
    )
    parser.add_argument(
        "--pests-per-hour",
        type=float,
        default=None,
        help=f"default {defaults.pests_per_hour:g}",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=None,
        help=f"pest spread probability (default {defaults.spread_probability:g})",
    )
    add_session_arguments(parser, default_speed=DEFAULT_SPEED, recording=False)
    return parser.parse_args()


def build_scenario(args: argparse.Namespace) -> PestScenario:
    """``PestScenario``, or its smoke preset with ``--smoke``, changed by the options given."""
    scenario = PestScenario()
    if args.smoke:
        scenario = PestScenario.smoke()
    if args.scouts is not None:
        scenario = replace(scenario, scouts=args.scouts)
    if args.killers is not None:
        scenario = replace(scenario, killers=args.killers)
    if args.killers_treat_only:
        scenario = replace(scenario, killers_explore=False)
    if args.killer_speed is not None:
        scenario = replace(scenario, speed_mps_ground=args.killer_speed)
    if args.radio is not None:
        scenario = replace(scenario, radio_range_m=args.radio)
    if args.pests_per_hour is not None:
        scenario = replace(scenario, pests_per_hour=args.pests_per_hour)
    if args.spread is not None:
        scenario = replace(scenario, spread_probability=args.spread)
    if args.seed is not None:
        scenario = replace(scenario, seed=args.seed)
    if args.hours is not None:
        scenario = replace(scenario, until=int(args.hours * TimeUnit.HOUR))
    return scenario


def main() -> None:
    args = parse_args()
    scenario = build_scenario(args)
    run = LiveRun.from_config(scenario.into_experiment())

    def panel(snapshot: WorldSnapshot) -> list[str]:
        return panel_lines(run, scenario, snapshot)

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


def panel_lines(run: LiveRun, scenario: PestScenario, snapshot: WorldSnapshot) -> list[str]:
    """The run's settings, its measures, and what the killers do and know."""
    killers = "explore + treat"
    if not scenario.killers_explore:
        killers = "treat only"
    lines = [
        f"time          {_clock(snapshot.time)}",
        f"robots        {scenario.scouts} scouts, {scenario.killers} killers",
        f"killers       {killers}, {scenario.speed_mps_ground:g} m/s",
        f"radio range   {scenario.radio_range_m:.0f} m",
        f"pest rate     {scenario.pests_per_hour:.1f} per hour",
        "",
    ]
    latest = run.metrics.latest()
    if latest is not None:
        lines.extend(
            [
                f"infested      {latest.targets_remaining} cells",
                f"pests         {latest.pests_cleared} of {latest.pests_appeared} cleared",
                f"detection     {_minutes(latest.detection_latency_ms)} (mean)",
                f"news reaches  {_minutes(latest.knowledge_latency_ms)} later",
                f"cleared after {_minutes(latest.response_time_ms)} (mean)",
                f"covered       {latest.coverage:6.1%}",
                f"idleness      {latest.mean_idleness_ms / TimeUnit.MINUTE:5.1f} min (mean)",
                f"redundancy    {latest.redundancy:6.1%}",
                "",
                f"killer time   {latest.killer_driving_share:6.1%} driving",
                f"              {latest.killer_treating_share:6.1%} treating",
                f"              {latest.killer_exploring_share:6.1%} exploring",
                f"              {latest.killer_idle_share:6.1%} idle",
                f"same pest     {latest.duplicate_trip_share:6.1%} of trips",
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
        known = 0
        for pest in agent.memory.infestations():
            if pest.severity > 0:
                known += 1
        rows.append(f"{robot.robot_id:6d}  {goal:8s} {known:5d}")
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
