"""Display an end-to-end battery lifecycle demo in a Matplotlib window.

The demo uses the existing build_agent / UtilityArbiter / potential-field navigation path and
runs the normal discrete-event loop. It shows progressive charging, a per-dock FIFO queue, BATTERY
spare-stock replenishment, and a KILLER that cannot afford its next decision and fails off-dock.

Run from the repository root with:
    uv run --with matplotlib python examples/battery_lifecycle_demo.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.patches import Circle

from swarm_field.contracts import (
    BodyType,
    CropType,
    EnergySpec,
    FieldGeometry,
    RobotId,
    RobotSpec,
    ScenarioConfig,
    SwarmAgent,
    Time,
    TimeUnit,
    WorldSnapshot,
)
from swarm_field.simulation import SimulationContext, SimulationLoop, WorldState, build_world
from swarm_field.simulation.event import MetricSample, RobotDecision
from swarm_field.simulation.metrics import MetricsConfig
from swarm_field.swarm import ReturnToDock, SupplyBattery, TreatInfestation, build_agent

BODY_COLOURS = {
    BodyType.SCOUT: "#168aad",
    BodyType.KILLER: "#e76f51",
    BodyType.BATTERY: "#7b2cbf",
}
BODY_MARKERS = {
    BodyType.SCOUT: "o",
    BodyType.KILLER: "s",
    BodyType.BATTERY: "D",
}


@dataclass(frozen=True)
class RobotFrame:
    """The visualization-only state of a robot at one render tick."""

    robot_id: int
    body: BodyType
    x: float
    y: float
    battery: float
    failed: bool
    charging: bool
    queued: bool
    action: str
    spare_batteries: int


@dataclass(frozen=True)
class DemoFrame:
    """A captured visual state, including information not exported in public snapshots."""

    time: Time
    robots: tuple[RobotFrame, ...]
    active_robots: int
    battery_failures: int


class BatteryDemoCapture:
    """Capture render ticks without modifying the simulation or its agents."""

    def __init__(self, world: WorldState) -> None:
        self.world = world
        self.frames: list[DemoFrame] = []

    def render(self, snapshot: WorldSnapshot) -> None:
        """Store the render snapshot plus charger-queue state for the animation."""
        # The renderer is intentionally a thin observer: it reads the existing ground-truth
        # WorldState for queue details not exposed through the common public snapshot contract.
        world = self.world
        robots: list[RobotFrame] = []
        for public in snapshot.robots:
            state = world.robots[public.robot_id]
            x, y = world.continuous_position(public.robot_id, snapshot.time)
            robots.append(
                RobotFrame(
                    robot_id=public.robot_id,
                    body=public.body,
                    x=x,
                    y=y,
                    battery=public.battery,
                    failed=public.failed,
                    charging=state.charging_updated_at is not None,
                    queued=(
                        state.charging_requested_at is not None
                        and state.charging_updated_at is None
                    ),
                    action=type(public.current_action).__name__
                    if public.current_action is not None
                    else "Idle",
                    spare_batteries=state.spare_batteries,
                )
            )
        self.frames.append(
            DemoFrame(
                time=snapshot.time,
                robots=tuple(robots),
                active_robots=len(world.live_robots()),
                battery_failures=world.battery_failure_count(),
            )
        )


def _make_context() -> tuple[SimulationContext, BatteryDemoCapture]:
    """Create a compact reproducible scenario for the physical battery lifecycle."""
    field = FieldGeometry(
        width=10,
        height=7,
        cell_size=5.0,
        row_sp=2.0,
        in_row_sp=1.0,
        headland=5.0,
        connectivity=4,
    )
    charging_dock = (1, 1)
    support_dock = (7, 5)
    docks = (charging_dock, support_dock)
    config = ScenarioConfig(
        field=field,
        crops={(x, y): CropType.MAIZE for x in range(field.width) for y in range(field.height)},
        robots=[
            RobotSpec(1, BodyType.SCOUT, charging_dock, 8.0, 0.20, 0.0, spare_batteries=0),
            RobotSpec(2, BodyType.KILLER, charging_dock, 1.0, 0.20, 0.0, spare_batteries=0),
            # Start the support robot full and stocked so it can leave the dock to help a peer.
            RobotSpec(3, BodyType.BATTERY, support_dock, 1.0, 1.0, 0.0, spare_batteries=3),
            # A low-battery peer is adjacent to the support robot and will receive a timed exchange.
            RobotSpec(4, BodyType.KILLER, (8, 5), 1.0, 0.20, 0.0, spare_batteries=0),
            # This robot cannot pay its next decision cost away from a dock and must fail safely.
            RobotSpec(5, BodyType.KILLER, (4, 3), 1.0, 0.00005, 0.0, spare_batteries=0),
        ],
        seed=77,
        docks=docks,
        energy=EnergySpec(per_metre=0.0, per_decision=0.0001, treat_per_action=0.0),
    )
    world = build_world(config)
    capture = BatteryDemoCapture(world)
    agents: dict[RobotId, SwarmAgent] = {
        robot.robot_id: build_agent(
            robot.robot_id,
            robot.body,
            field,
            docks=config.docks,
            behaviors=(
                [SupplyBattery(), ReturnToDock(docks=config.docks)]
                if robot.body is BodyType.BATTERY
                else [TreatInfestation()]
                if robot.robot_id == 4
                else [ReturnToDock(docks=config.docks)]
            ),
        )
        for robot in config.robots
    }
    context = SimulationContext(
        now=0,
        rng=Random(config.seed),
        noise=Random(config.seed + 1),
        agents=agents,
        world=world,
        renderer=capture,
    )
    return context, capture


def _status(robot: RobotFrame) -> str:
    if robot.failed:
        return "DEAD_BATTERY"
    if robot.charging:
        return "CHARGING"
    if robot.queued:
        return "WAITING FOR DOCK"
    return robot.action.upper()


def _draw_map(
    ax: Axes, frame: DemoFrame, field: FieldGeometry, docks: tuple[tuple[int, int], ...]
) -> None:
    ax.set_title("Field and charging docks", loc="left", fontsize=12, weight="bold")
    ax.set_xlim(0, field.width)
    ax.set_ylim(0, field.height)
    ax.set_aspect("equal")
    ax.set_xticks(range(field.width + 1), minor=True)
    ax.set_yticks(range(field.height + 1), minor=True)
    ax.grid(which="minor", linewidth=0.6, alpha=0.25)
    ax.set_xticks(range(field.width))
    ax.set_yticks(range(field.height))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Field cell x")
    ax.set_ylabel("Field cell y")

    for dock_index, dock in enumerate(docks):
        dock_centre = (dock[0] + 0.5, dock[1] + 0.5)
        ax.scatter(
            [dock_centre[0]],
            [dock_centre[1]],
            marker="*",
            s=360,
            color="#2a9d8f",
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
            label="Charging dock" if dock_index == 0 else None,
        )

    status_lines: list[str] = []
    co_located: dict[tuple[float, float], list[RobotFrame]] = {}
    for robot in frame.robots:
        co_located.setdefault((round(robot.x, 3), round(robot.y, 3)), []).append(robot)
    display_positions: dict[int, tuple[float, float]] = {}
    for same_cell in co_located.values():
        for index, robot in enumerate(same_cell):
            if len(same_cell) == 1:
                display_positions[robot.robot_id] = (robot.x, robot.y)
            else:
                angle = 2.0 * 3.141592653589793 * index / len(same_cell)
                display_positions[robot.robot_id] = (
                    robot.x + 0.20 * math.cos(angle),
                    robot.y + 0.20 * math.sin(angle),
                )

    for robot in frame.robots:
        colour = BODY_COLOURS[robot.body]
        marker = BODY_MARKERS[robot.body]
        draw_x, draw_y = display_positions[robot.robot_id]
        if robot.failed:
            ax.scatter([draw_x], [draw_y], marker="X", s=150, color="#444444", zorder=6)
        else:
            ax.scatter(
                [draw_x],
                [draw_y],
                marker=marker,
                s=115,
                color=colour,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )
        if robot.charging:
            ax.add_patch(
                Circle((draw_x, draw_y), 0.36, fill=False, edgecolor="#f4a261", linewidth=2.5)
            )
        elif robot.queued:
            ax.add_patch(
                Circle(
                    (draw_x, draw_y),
                    0.36,
                    fill=False,
                    edgecolor="#f4a261",
                    linewidth=2.0,
                    linestyle="--",
                )
            )
        ax.annotate(
            f"R{robot.robot_id}",
            (draw_x, draw_y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
        )
        status_lines.append(
            f"R{robot.robot_id} {robot.body.name}: {_status(robot)}  "
            f"{robot.battery * 100:.0f}%"
            + (f"  spares {robot.spare_batteries}" if robot.body is BodyType.BATTERY else "")
        )

    ax.text(
        0.02,
        0.02,
        "\n".join(status_lines),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.92},
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def _draw_battery_history(ax: Axes, frames: list[DemoFrame], index: int) -> None:
    ax.set_title("Battery charge over simulated time", loc="left", fontsize=12, weight="bold")
    ax.set_xlim(0, max(1.0, frames[-1].time / TimeUnit.MINUTE))
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Simulated time (minutes)")
    ax.set_ylabel("Battery fraction")
    ax.grid(True, alpha=0.25)
    ax.axhline(0.25, color="#e9c46a", linestyle="--", linewidth=1.1, label="Low: 25%")
    ax.axhline(0.10, color="#e76f51", linestyle="--", linewidth=1.1, label="Critical: 10%")

    visible = frames[: index + 1]
    times = [frame.time / TimeUnit.MINUTE for frame in visible]
    for robot in visible[-1].robots:
        values = [
            next(item.battery for item in frame.robots if item.robot_id == robot.robot_id)
            for frame in visible
        ]
        ax.plot(
            times,
            values,
            color=BODY_COLOURS[robot.body],
            linewidth=2.0,
            marker="o",
            markersize=2.5,
            label=f"R{robot.robot_id} {robot.body.name}",
        )
    ax.legend(loc="center right", fontsize=8, framealpha=0.92)


def show_animation(
    frames: list[DemoFrame], field: FieldGeometry, docks: tuple[tuple[int, int], ...]
) -> None:
    """Play the captured simulation in an interactive Matplotlib window."""
    fig, (map_ax, battery_ax) = plt.subplots(
        1,
        2,
        figsize=(13, 6.5),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
        layout="constrained",
    )

    def animate(index: int) -> tuple[object, ...]:
        map_ax.clear()
        battery_ax.clear()
        frame = frames[index]
        _draw_map(map_ax, frame, field, docks=docks)
        _draw_battery_history(battery_ax, frames, index)
        minutes = frame.time / TimeUnit.MINUTE
        fig.suptitle(
            f"Issue #77 — physical battery lifecycle  |  t = {minutes:5.1f} min  |  "
            f"active robots = {frame.active_robots}  |  "
            f"battery failures = {frame.battery_failures}",
            fontsize=13,
            weight="bold",
        )
        return ()

    # Keep a reference alive while show() runs so Matplotlib can continue the animation.
    animation = FuncAnimation(fig, animate, frames=len(frames), interval=125, blit=False)
    plt.show()
    # Explicitly reference the object after show() to make its intended lifetime clear.
    _ = animation


def main() -> None:
    # A concrete capture object is connected to the ordinary simulation renderer port. The
    # context builder fills in its WorldState before any events are processed.
    ctx, capture = _make_context()
    config = MetricsConfig(period=2 * TimeUnit.MINUTE)
    for robot_id in sorted(ctx.world.robots):
        ctx.schedule(
            RobotDecision(
                time=0,
                robot=robot_id,
                decision_interval=5 * TimeUnit.SECOND,
            )
        )
    ctx.schedule(MetricSample(time=0, config=config))

    until = 150 * TimeUnit.MINUTE
    SimulationLoop().run(ctx, until=until)
    if not capture.frames:
        raise RuntimeError("simulation produced no render frames")
    show_animation(capture.frames, ctx.world.field, tuple(sorted(ctx.world.docks)))
    print(f"Displayed {len(capture.frames)} simulation frames.")
    print(f"Battery failures at end: {ctx.world.battery_failure_count()}")


if __name__ == "__main__":
    main()
