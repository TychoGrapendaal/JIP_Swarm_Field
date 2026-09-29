"""Animated demonstration of the event-driven robot Perceive -> Decide -> Act loop.

This demo uses the real ``SimulationLoop`` and ``RobotDecision`` event. A small scripted agent
moves through a fixed sequence of cells so the event lifecycle is easy to see.

Run:
    uv run --with matplotlib python examples/robot_decision_demo.py

Headless (write PNG frames):
    uv run --with matplotlib python examples/robot_decision_demo.py --frames 180 --save frames
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from random import Random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from swarm_field.contracts import (
    BodyType,
    Decision,
    FieldGeometry,
    Idle,
    Move,
    Perception,
)
from swarm_field.simulation import (
    RobotDecision,
    RobotState,
    SimulationContext,
    SimulationLoop,
    WorldState,
)

FIELD_WIDTH = 10
FIELD_HEIGHT = 8
CELL_SIZE_M = 1.0
FRAME_TIME_MS = 100
DURATION_MS = 20_000
ROBOT_ID = 1
DWELL_MS = 1_500

PATH: tuple[tuple[int, int], ...] = (
    (1, 1),
    (6, 1),
    (6, 6),
    (2, 6),
    (2, 2),
    (8, 2),
    (8, 5),
)


@dataclass(frozen=True)
class DecisionRecord:
    """One perception/decision observed from the robot brain."""

    time: int
    position: tuple[int, int]
    action: Move | Idle


class ScriptedAgent:
    """Tiny deterministic agent used only to make the event lifecycle visible."""

    def __init__(self, path: tuple[tuple[int, int], ...]) -> None:
        self.path = path
        self.index = 1
        self.records: list[DecisionRecord] = []

    def decide(self, perception: Perception) -> Decision:
        """Record the perception and select the next point in the demonstration path."""
        if self.index >= len(self.path):
            action = Idle()
        else:
            action = Move(to=self.path[self.index])
            self.index += 1

        self.records.append(
            DecisionRecord(
                time=perception.time,
                position=perception.self_state.position,
                action=action,
            )
        )
        return Decision(action=action)

    def inspect(self):
        """Return the minimal inspection data needed by the agent protocol."""
        return None


def build_context() -> tuple[SimulationContext, ScriptedAgent]:
    """Build one robot, one scripted brain, and the event-driven simulation context."""
    field = FieldGeometry(
        width=FIELD_WIDTH,
        height=FIELD_HEIGHT,
        cell_size=CELL_SIZE_M,
        row_sp=1.0,
        in_row_sp=1.0,
        headland=0.0,
        connectivity=4,
    )
    robot = RobotState(
        robot_id=ROBOT_ID,
        position=PATH[0],
        body=BodyType.SCOUT,
        speed_mps=1.0,
    )
    world = WorldState(field, robots=[robot])
    agent = ScriptedAgent(PATH)
    ctx = SimulationContext(
        now=0,
        rng=Random(7),
        agents={ROBOT_ID: agent},
        world=world,
    )
    ctx.schedule(RobotDecision(time=0, robot=ROBOT_ID, dwell=DWELL_MS))
    return ctx, agent


def current_record(agent: ScriptedAgent, time: int) -> DecisionRecord | None:
    """Return the latest recorded decision at or before ``time``."""
    visible = [record for record in agent.records if record.time <= time]
    return visible[-1] if visible else None


def render_frame(
    ax,
    ctx: SimulationContext,
    agent: ScriptedAgent,
    time: int,
) -> None:
    """Advance the real event loop and draw the current robot/event state."""
    SimulationLoop().run(ctx, until=time)
    world = ctx.world
    robot = world.robots[ROBOT_ID]

    ax.clear()

    for y in range(FIELD_HEIGHT):
        for x in range(FIELD_WIDTH):
            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=(0.92, 0.86, 0.70),
                    edgecolor=(0.78, 0.73, 0.62),
                    linewidth=0.5,
                )
            )

    x, y = PATH[0]
    ax.scatter(x + 0.5, y + 0.5, marker="x", s=120, linewidths=2, color="black")

    if robot.move_target is not None:
        target = robot.move_target
        ax.plot(
            [robot.position[0] + 0.5, target[0] + 0.5],
            [robot.position[1] + 0.5, target[1] + 0.5],
            linestyle="--",
            color="tab:blue",
            linewidth=1.5,
            alpha=0.7,
        )

    px, py = world.continuous_position(ROBOT_ID, time)
    ax.scatter(
        px,
        py,
        s=180,
        color="tab:blue",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="Robot 1",
    )
    ax.text(px + 0.15, py + 0.15, "R1", fontsize=10, weight="bold", zorder=6)

    record = current_record(agent, time)
    if record is None:
        phase = "WAITING"
        action_text = "—"
        decision_text = "—"
    elif robot.move_target is not None:
        phase = "ACT / MOVE"
        action_text = f"Move → {record.action.to}" if isinstance(record.action, Move) else "Move"
        decision_text = f"t={record.time / 1000:.1f}s"
    elif isinstance(record.action, Idle):
        phase = "ACT / IDLE"
        action_text = "Idle"
        decision_text = f"t={record.time / 1000:.1f}s"
    else:
        phase = "DECIDE / ACT"
        action_text = f"Move → {record.action.to}"
        decision_text = f"t={record.time / 1000:.1f}s"

    ax.set_title(
        "Perceive → Decide → Act → Re-queue\n"
        f"Simulation t={time / 1000:.1f}s | Phase: {phase} | Last decision: {decision_text}"
    )
    ax.text(
        0.02,
        0.98,
        "PERCEIVE\n  ↓\nDECIDE\n  ↓\nACT\n  ↓\nRE-QUEUE",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.text(
        0.98,
        0.02,
        f"Robot 1 position: {robot.position}\n"
        f"Action: {action_text}\n"
        f"Next queued event: {ctx.queue.peek().time / 1000:.1f}s"
        if ctx.queue.peek() is not None
        else "Robot 1 position: done",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    ax.set_xlim(0, FIELD_WIDTH)
    ax.set_ylim(0, FIELD_HEIGHT)
    ax.set_aspect("equal")
    ax.set_xlabel("Field cell X")
    ax.set_ylabel("Field cell Y")
    ax.legend(loc="upper right")
    ax.grid(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=DURATION_MS // FRAME_TIME_MS)
    parser.add_argument("--frame-ms", type=int, default=FRAME_TIME_MS)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="directory to save PNG frames instead of opening a window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_ms <= 0:
        raise SystemExit("--frame-ms must be positive")
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")

    ctx, agent = build_context()
    fig, ax = plt.subplots(figsize=(10, 7))

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for frame in range(args.frames):
            time = frame * args.frame_ms
            render_frame(ax, ctx, agent, time)
            fig.savefig(args.save / f"frame_{frame:04d}.png", dpi=120)
        plt.close(fig)
        print(f"Wrote {args.frames} frames to {args.save}")
        return

    def update(frame: int) -> None:
        render_frame(ax, ctx, agent, frame * args.frame_ms)

    animation = FuncAnimation(fig, update, frames=args.frames, interval=50, repeat=False)
    _ = animation
    plt.show()


if __name__ == "__main__":
    main()
