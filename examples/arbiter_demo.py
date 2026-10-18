"""Fourteen robots arbitrating between exploring, treating and recharging, on ``SimulationViewer``.

Eight scouts and six killers start along the bottom of a field of 10 m cells with a river and two
rocks, and four docks near the corners. Every robot is a real ``ComposedAgent`` with a
``UtilityArbiter`` on the real simulation loop:

- scouts weigh exploring against ``ReturnToDock``;
- killers weigh ``TreatInfestation``, exploring and ``ReturnToDock``.

Pests keep spawning and batteries keep draining, so every rule of the arbiter gets exercised:
overrides (treating the pest underfoot, a critical battery, charging on a dock), a finished
objective freeing the robot, a finished target moving it on within its objective, the minimum
dwell, and the effort margin holding a commitment or letting a rival through.
``ArbiterScenario`` describes the run, and an ``ArbiterLog`` counts the rule behind every
choice.

Green squares are docks, red cells pests, trails show where robots went and dashed lines the
target each robot is heading for. The panel counts the rules, lists the latest notable
decisions, and shows what each robot is doing and its battery.

Run from the repository root with:
    uv run python examples/arbiter_demo.py

Record a video instead (mp4 through ffmpeg, or a gif):
    uv run python examples/arbiter_demo.py --save videos/arbiter.mp4

Only print the rule counts of a run, without drawing it:
    uv run python examples/arbiter_demo.py --report --hours 3
"""

from __future__ import annotations

import argparse
import json
import logging
import time as wallclock
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import (
    BodyType,
    Charge,
    SwitchRole,
    Time,
    TimeUnit,
    Treat,
    WorldSnapshot,
)
from swarm_field.experiments import ArbiterLog, ArbiterScenario, LiveRun
from swarm_field.experiments.arbiter import EXPLORE_BEHAVIOURS, ArbiterDecision, ArbiterRule
from swarm_field.swarm import RebalanceConfig
from swarm_field.visualization import SimulationViewer

GOAL_LABELS: dict[str | None, str] = {
    "follow-field": "explore",
    "explore": "explore",
    "treat-infestation": "treat",
    "return-to-dock": "dock",
    "rebalance-roles": "switch",
    None: "idle",
}

RULE_LABELS: dict[ArbiterRule, str] = {
    ArbiterRule.OVERRIDE: "1 override",
    ArbiterRule.OBJECTIVE_FINISHED: "2 objective done",
    ArbiterRule.DWELL_HOLD: "4 dwell hold",
    ArbiterRule.MARGIN_HOLD: "5 margin hold",
    ArbiterRule.MARGIN_SWITCH: "5 margin switch",
    ArbiterRule.UNCONTESTED: "- uncontested",
    ArbiterRule.FIRST_COMMITMENT: "- first choice",
}

BODY_LETTERS: dict[BodyType, str] = {
    BodyType.SCOUT: "S",
    BodyType.KILLER: "K",
    BodyType.BATTERY: "B",
}


def parse_args() -> argparse.Namespace:
    """The command line; the scenario's settings default to those of ``ArbiterScenario``."""
    defaults = ArbiterScenario()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scouts", type=int, default=defaults.scouts)
    parser.add_argument("--killers", type=int, default=defaults.killers)
    parser.add_argument(
        "--explore",
        choices=EXPLORE_BEHAVIOURS,
        default=defaults.explore_behaviour,
        help="mock explore behaviour (even 0.5 bid) or the scored ExploreBehaviour",
    )
    parser.add_argument("--pests-per-hour", type=float, default=defaults.pests_per_hour)
    parser.add_argument(
        "--radio", type=float, default=defaults.radio_range_m, help="radio range in metres"
    )
    parser.add_argument("--hours", type=float, default=3.0, help="simulated time")
    parser.add_argument("--frame-seconds", type=float, default=20.0, help="simulated per frame")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--save", type=Path, help="write a .mp4 or .gif instead of a window")
    parser.add_argument(
        "--rebalance", action="store_true", help="robots switch between monitor and killer (#64)"
    )
    parser.add_argument(
        "--switch-at-dock", action="store_true", help="with --rebalance, switch only on a dock"
    )
    parser.add_argument("--report", action="store_true", help="run headless, print rule counts")
    parser.add_argument("--out", type=Path, help="write the rule counts as JSON to this file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    # An explore field with no seeds left (everything freshly sensed) makes the brain idle and
    # warn on every decision; that is expected here, so keep the console readable.
    logging.getLogger("swarm_field.swarm.brain").setLevel(logging.ERROR)

    rebalance: RebalanceConfig | None = None
    if args.rebalance:
        rebalance = RebalanceConfig(switch_at_dock=args.switch_at_dock)
    scenario = ArbiterScenario(
        rebalance=rebalance,
        scouts=args.scouts,
        killers=args.killers,
        explore_behaviour=args.explore,
        pests_per_hour=args.pests_per_hour,
        radio_range_m=args.radio,
        until=int(args.hours * TimeUnit.HOUR),
        seed=args.seed,
    )
    if rebalance is not None:
        needed = scenario.monitors_needed()
        scenario = replace(scenario, rebalance=replace(rebalance, min_monitors=needed))
    run = LiveRun.from_config(scenario.into_experiment())
    log = ArbiterLog()
    log.attach(run)
    low_battery = LowestBattery()
    roles = RoleTimeline()
    until = int(args.hours * TimeUnit.HOUR)

    started = wallclock.monotonic()
    if args.report:
        for minute in range(0, until // TimeUnit.MINUTE + 1):
            run.advance_to(minute * TimeUnit.MINUTE)
            snapshot = run.context.world.snapshot(minute * TimeUnit.MINUTE)
            low_battery.update(snapshot)
            roles.update(snapshot)
    else:
        docks = scenario.dock_cells()

        def overlay(snapshot: WorldSnapshot) -> NDArray[np.float64]:
            values = np.full((snapshot.field.width, snapshot.field.height), np.nan)
            for dock in docks:
                values[dock] = 1.0
            return values

        def panel(snapshot: WorldSnapshot) -> list[str]:
            low_battery.update(snapshot)
            roles.update(snapshot)
            return panel_lines(run, scenario, log, low_battery, roles, snapshot)

        viewer = SimulationViewer(
            run.snapshot_at,
            overlay=overlay,
            overlay_label="green = dock",
            overlay_colormap="Greens",
            goals=run.targets,
            panel=panel,
            figsize=(15.5, 9.0),
        )
        step = int(args.frame_seconds * TimeUnit.SECOND)
        viewer.animate(list(range(0, until + 1, step)), fps=args.fps, save=args.save)

    report = report_of(log, low_battery, roles, run, wallclock.monotonic() - started)
    print("\n".join(report_lines(report)))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


class LowestBattery:
    """The lowest battery each robot has reached in the frames seen so far, and the charges.

    Attributes:
        lowest: The lowest battery seen per robot.
        stranded: Robots seen with an empty battery.
    """

    def __init__(self) -> None:
        self.lowest: dict[int, float] = {}
        self.stranded: set[int] = set()

    def update(self, snapshot: WorldSnapshot) -> None:
        for robot in snapshot.robots:
            self.lowest[robot.robot_id] = min(self.lowest.get(robot.robot_id, 1.0), robot.battery)
            if robot.battery <= 0.0:
                self.stranded.add(robot.robot_id)


class RoleTimeline:
    """How many robots had each role, and how many pests there were, at each frame.

    Attributes:
        samples: Per frame: the time, the monitors, the killers and the infested cells.
        switches: Role switches seen so far, over all robots.
    """

    def __init__(self) -> None:
        self.samples: list[tuple[Time, int, int, int]] = []
        self.switches: int = 0
        self._bodies: dict[int, BodyType] = {}

    def update(self, snapshot: WorldSnapshot) -> None:
        monitors = 0
        killers = 0
        for robot in snapshot.robots:
            before = self._bodies.get(robot.robot_id)
            if before is not None and before is not robot.body:
                self.switches += 1
            self._bodies[robot.robot_id] = robot.body
            if robot.body is BodyType.SCOUT:
                monitors += 1
            elif robot.body is BodyType.KILLER:
                killers += 1
        pests = sum(1 for cell in snapshot.cells.values() if cell.infestation > 0)
        self.samples.append((snapshot.time, monitors, killers, pests))

    def latest(self) -> tuple[Time, int, int, int]:
        return self.samples[-1]


def panel_lines(
    run: LiveRun,
    scenario: ArbiterScenario,
    log: ArbiterLog,
    low_battery: LowestBattery,
    roles: RoleTimeline,
    snapshot: WorldSnapshot,
) -> list[str]:
    """The run's settings, the rule counts, the latest notable decisions and every robot."""
    _, monitors, killers, pests = roles.latest()
    mode = "fixed roles"
    if scenario.rebalance is not None:
        mode = f"rebalancing, {roles.switches} switches"
    lines = [
        f"time {_clock(snapshot.time)}   {monitors} monitors  {killers} killers"
        f"  {len(scenario.dock_cells())} docks",
        f"roles {mode}",
        f"pests now {pests:3d}   rate {scenario.pests_per_hour:.0f}/h   radio "
        f"{scenario.radio_range_m:.0f} m",
        "",
        "arbiter rule        choices",
    ]
    total = max(1, sum(log.rules.values()))
    for rule, label in RULE_LABELS.items():
        count = log.rules[rule]
        lines.append(f"  {label:17s} {count:6d} {count / total:6.1%}")
    lines.append(f"  {'3 retarget':17s} {log.retargets:6d}")
    per_job = log.switches_per_job()
    per_job_text = "   -"
    if per_job is not None:
        per_job_text = f"{per_job:4.2f}"
    lines.append(f"  switches/job {per_job_text}   stranded {len(low_battery.stranded)}")
    lines.append("")
    lines.append("latest notable decisions")
    for decision in list(log.notable)[-6:]:
        lines.append("  " + _describe(decision))
    lines.append("")
    lines.append(" id   goal      batt  last rule")
    agents = run.agents()
    for robot in snapshot.robots:
        goal = GOAL_LABELS.get(agents[robot.robot_id].inspect().current_goal, "other")
        if isinstance(robot.current_action, Charge):
            goal = "charging"
        elif isinstance(robot.current_action, SwitchRole):
            goal = "switching"
        elif isinstance(robot.current_action, Treat):
            goal = "treating"
        last = log.last.get(robot.robot_id)
        rule = ""
        if last is not None:
            rule = RULE_LABELS[last.rule][2:]
        letter = BODY_LETTERS[robot.body]
        bar = _battery_bar(robot.battery)
        lines.append(f"{letter}{robot.robot_id:<3d} {goal:9s} {bar} {rule}")
    return lines


def report_of(
    log: ArbiterLog,
    low_battery: LowestBattery,
    roles: RoleTimeline,
    run: LiveRun,
    wall_seconds: float,
) -> dict[str, object]:
    """The rule counts of a run, overall and per body, with the figures that judge them."""
    by_body: dict[str, dict[str, int]] = {}
    bodies = {robot.robot_id: robot.body for robot in run.context.world.public_states()}
    for robot_id, counts in log.rules_by_robot.items():
        body = bodies[robot_id].name.lower()
        totals = by_body.setdefault(body, {})
        for rule, count in counts.items():
            totals[rule.name] = totals.get(rule.name, 0) + count
    latest = run.metrics.latest()
    return {
        "simulated_hours": run.context.now / TimeUnit.HOUR,
        "robots": len(bodies),
        "decisions": log.decisions(),
        "rules": {rule.name: log.rules[rule] for rule in ArbiterRule},
        "rules_by_body": by_body,
        "retargets": log.retargets,
        "switches": log.switches,
        "completed_jobs": log.completed_jobs(),
        "switches_per_job": log.switches_per_job(),
        "winners": dict(log.winners),
        "lowest_battery": min(low_battery.lowest.values(), default=1.0),
        "stranded": len(low_battery.stranded),
        "pests_left": None if latest is None else latest.targets_remaining,
        "coverage": None if latest is None else latest.coverage,
        "role_switches": roles.switches,
        "mean_infested_cells": round(
            sum(p for _, _, _, p in roles.samples) / len(roles.samples), 2
        ),
        "roles_every_15_min": [
            {"minute": t // TimeUnit.MINUTE, "monitors": m, "killers": k, "pests": p}
            for t, m, k, p in roles.samples
            if t % (15 * TimeUnit.MINUTE) == 0
        ],
        "wall_seconds": round(wall_seconds, 1),
    }


def report_lines(report: dict[str, object]) -> list[str]:
    """The report as a short table for the console."""
    rules = report["rules"]
    assert isinstance(rules, dict)
    decided = max(1, sum(rules.values()))
    lines = [
        f"{report['robots']} robots, {report['simulated_hours']:.1f} simulated hours, "
        f"{report['decisions']} decisions ({report['wall_seconds']} s wall clock)",
        "",
        f"{'rule':20s} {'choices':>8s} {'share':>7s}",
    ]
    for rule, label in RULE_LABELS.items():
        count = rules[rule.name]
        lines.append(f"{label:20s} {count:8d} {count / decided:7.1%}")
    lines.append(f"{'3 retarget':20s} {report['retargets']:8d}")
    per_job = report["switches_per_job"]
    per_job_text = "-" if per_job is None else f"{per_job:.2f}"
    lines.extend(
        [
            "",
            f"switches {report['switches']}, completed jobs {report['completed_jobs']}, "
            f"switches per job {per_job_text}",
            f"lowest battery {report['lowest_battery']:.2f}, stranded robots {report['stranded']}",
            f"pests left {report['pests_left']}, coverage {report['coverage']}",
            f"winning behaviours {report['winners']}",
            f"role switches {report['role_switches']}, "
            f"mean infested cells {report['mean_infested_cells']}",
            "",
            "minute  monitors  killers  pests",
        ]
    )
    timeline = report["roles_every_15_min"]
    assert isinstance(timeline, list)
    for row in timeline:
        lines.append(
            f"{row['minute']:6d} {row['monitors']:9d} {row['killers']:8d} {row['pests']:6d}"
        )
    return lines


def _describe(decision: ArbiterDecision) -> str:
    """One notable decision as a short line: when, who, which rule, and what changed."""
    who = f"{_clock(decision.time)} #{decision.robot:<2d}"
    now = GOAL_LABELS.get(decision.source, decision.source)
    if decision.action == "Charge":
        now = "charge"
    elif decision.action == "Treat":
        now = "treat here"
    elif decision.action == "SwitchRole":
        now = "switch role"
    before = GOAL_LABELS.get(decision.previous, decision.previous or "-")
    rule = RULE_LABELS[decision.rule][2:]
    if decision.retargeted and decision.rule is ArbiterRule.UNCONTESTED:
        return f"{who} retarget   {now} next target"
    if decision.rule in (ArbiterRule.DWELL_HOLD, ArbiterRule.MARGIN_HOLD) and decision.rival:
        rival = GOAL_LABELS.get(decision.rival[0], decision.rival[0])
        return f"{who} {rule:13s} {now} {decision.score:.2f} vs {rival} {decision.rival[1]:.2f}"
    if before == now:
        return f"{who} {rule:13s} {now} {decision.score:.2f}"
    return f"{who} {rule:13s} {before} -> {now} {decision.score:.2f}"


def _battery_bar(battery: float) -> str:
    """The battery as a five-block bar and a percentage."""
    filled = round(max(0.0, min(1.0, battery)) * 5)
    return "#" * filled + "." * (5 - filled) + f"{battery:4.0%}"


def _clock(time: Time) -> str:
    """A time as hours and minutes."""
    minutes = int(time // TimeUnit.MINUTE)
    return f"{minutes // 60}:{minutes % 60:02d}"


if __name__ == "__main__":
    main()
