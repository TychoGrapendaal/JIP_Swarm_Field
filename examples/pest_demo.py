"""Scouts explore and killers explore and treat while pests keep appearing, on ``SimulationViewer``.

Five scouts and three killers start side by side in a corner of a field of 10 m cells with a
river and two rocks. Each is a real ``ComposedAgent`` with a ``UtilityArbiter`` on the real
simulation loop: scouts follow the EXPLORE field, killers weigh it against ``TreatInfestation``
on the TREAT field. With ``--killers-treat-only`` killers do not explore and wait until they hear
of a pest. Pests spawn at random free cells over time, and robots share what they know only
within radio range. ``PestScenario`` describes the run, and ``swarm_field.runner`` assembles and
runs it.

Red cells are the pests in ground truth, trails show where the robots went, and dashed lines the
target each robot is heading for. The panel shows the measures of ``MetricsCollector``, how the
killers spent their time, and what each killer does and knows.

Run from the repository root with:
    uv run python examples/pest_demo.py

Record a video instead (mp4 through ffmpeg, or a gif):
    uv run python examples/pest_demo.py --save videos/pests.mp4
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from swarm_field.contracts import BodyType, TimeUnit, WorldSnapshot
from swarm_field.experiments import LiveRun, PestScenario
from swarm_field.visualization import SimulationViewer

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
    parser.add_argument("--hours", type=float, default=3.0, help="simulated time")
    parser.add_argument("--frame-seconds", type=float, default=20.0, help="simulated per frame")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--save", type=Path, help="write a .mp4 or .gif instead of a window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = PestScenario(
        scouts=args.scouts,
        killers=args.killers,
        killers_explore=not args.killers_treat_only,
        radio_range_m=args.radio,
        pests_per_hour=args.pests_per_hour,
        spread_probability=args.spread,
        seed=args.seed,
    )
    run = LiveRun.from_config(scenario.into_experiment())
    killer_goals: Counter[str] = Counter()

    def panel(snapshot: WorldSnapshot) -> list[str]:
        return panel_lines(run, scenario, snapshot, killer_goals)

    viewer = SimulationViewer(run.snapshot_at, goals=run.targets, panel=panel)
    step = int(args.frame_seconds * TimeUnit.SECOND)
    frames = list(range(0, int(args.hours * TimeUnit.HOUR) + 1, step))
    viewer.animate(frames, fps=args.fps, save=args.save)


def panel_lines(
    run: LiveRun, scenario: PestScenario, snapshot: WorldSnapshot, killer_goals: Counter[str]
) -> list[str]:
    """The run's settings, its measures, and what the killers do and know.

    ``killer_goals`` counts the killers' goals over the frames drawn so far; frames are evenly
    spaced in time, so it says how the killers spent their time.
    """
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
    rows: list[str] = []
    for robot in snapshot.robots:
        if robot.body is not BodyType.KILLER:
            continue
        agent = agents[robot.robot_id]
        goal = GOAL_LABELS.get(agent.inspect().current_goal, "other")
        killer_goals[goal] += 1
        known = 0
        for pest in agent.memory.infestations():
            if pest.severity > 0:
                known += 1
        rows.append(f"{robot.robot_id:6d}  {goal:8s} {known:5d}")
    total = sum(killer_goals.values())
    for goal, count in sorted(killer_goals.items()):
        lines.append(f"killer time   {count / total:6.1%} {goal}")
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
