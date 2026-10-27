"""Live, architecture-backed field simulation with a potential-field visual style.

The window renders the real ``LiveRun`` produced by ``swarm_field.runner``. Every SCOUT,
KILLER, and BATTERY robot is a composed agent using the repository's behaviours, utility arbiter,
field provider, potential-field navigator, local memory, and radio channel. The plot only observes
that run; it does not supply its own movement controller.

Run from the repository root:
    uv run --with matplotlib python examples/field_simulation_demo.py

Controls:
    space / p  pause or resume
    s          process the next simulation event
    + / -      speed up / slow down simulated time
    0          reset to 1x
    h          hide / show the information panel
    click robot to focus its actual potential field; click an empty walkable cell to inject a pest

For a single static screenshot (useful on headless machines):
    uv run --with matplotlib python examples/field_simulation_demo.py --preview preview.png
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle

from swarm_field.console import InteractiveConsole, show_window
from swarm_field.contracts import (
    BodyType,
    CellId,
    EnergySpec,
    FieldGeometry,
    InjectInfestation,
    PestDynamicsSpec,
    RobotSpec,
    ScenarioConfig,
    TimeUnit,
    WorldSnapshot,
)
from swarm_field.experiments import LiveRun, cropped_cells
from swarm_field.runner import BrainSpec, ExperimentConfig
from swarm_field.simulation import MetricsConfig
from swarm_field.swarm import ArbiterConfig, ExploreConfig
from swarm_field.swarm.seeder import SeededField
from swarm_field.visualization.matplotlib_renderer import cell_rgba_image

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from matplotlib.text import Text

FIELD_WIDTH = 40
FIELD_HEIGHT = 30
CELL_SIZE_M = 10.0
DEFAULT_SPEED = 40.0
FRAME_INTERVAL_MS = 60
TRAIL_LENGTH = 22
LOW_BATTERY_THRESHOLD = 0.25

ROLE_COLOURS = {
    BodyType.SCOUT: "#2878b5",
    BodyType.KILLER: "#d43d5c",
    BodyType.BATTERY: "#ee8b2d",
}
ROLE_MARKERS = {
    BodyType.SCOUT: "o",
    BodyType.KILLER: "o",
    BodyType.BATTERY: "D",
}
ROLE_PREFIX = {
    BodyType.SCOUT: "S",
    BodyType.KILLER: "K",
    BodyType.BATTERY: "B",
}
SEEDER_FOR_GOAL = {
    "treat-infestation": "treat",
    "explore": "explore",
    "supply-battery": "battery-support",
    "return-to-dock": "dock",
}


def _field_and_obstacles() -> tuple[FieldGeometry, tuple[CellId, ...]]:
    field = FieldGeometry(
        width=FIELD_WIDTH,
        height=FIELD_HEIGHT,
        cell_size=CELL_SIZE_M,
        row_sp=2.0,
        in_row_sp=1.0,
        headland=5.0,
        connectivity=8,
    )
    # A two-cell river with two crossings, plus two square obstacles. This makes routing around
    # terrain visible while keeping every component on the normal potential-field path.
    obstacles: set[CellId] = set()
    for y in range(2, FIELD_HEIGHT - 2):
        if not 12 <= y <= 15 and not 22 <= y <= 24:
            obstacles.update({(19, y), (20, y)})
    obstacles.update((x, y) for x in range(10, 13) for y in range(9, 12))
    obstacles.update((x, y) for x in range(30, 33) for y in range(17, 20))
    return field, tuple(sorted(obstacles))


def _infestation_clusters(
    field: FieldGeometry, obstacles: tuple[CellId, ...]
) -> dict[CellId, float]:
    blocked = set(obstacles)
    centres: tuple[CellId, ...] = ((8, 15), (28, 8), (34, 23), (13, 23), (31, 6))
    infestation: dict[CellId, float] = {}
    for centre_x, centre_y in centres:
        for y in range(centre_y - 1, centre_y + 2):
            for x in range(centre_x - 1, centre_x + 2):
                cell = (x, y)
                if 0 <= x < field.width and 0 <= y < field.height and cell not in blocked:
                    infestation[cell] = 0.55 if cell == (centre_x, centre_y) else 0.35
    return infestation


def build_experiment(seed: int = 77, hours: float = 3.0) -> ExperimentConfig:
    """Compose a heterogeneous scenario through the repository's real runner and registries."""
    field, obstacles = _field_and_obstacles()
    docks: tuple[CellId, ...] = ((2, 2), (37, 27))
    infestation = _infestation_clusters(field, obstacles)
    scout_starts: tuple[CellId, ...] = (
        (3, 4),
        (5, 24),
        (8, 16),
        (13, 5),
        (14, 25),
        (24, 4),
        (27, 14),
        (34, 6),
        (36, 22),
        (32, 26),
    )
    killer_starts: tuple[CellId, ...] = ((8, 15), (28, 8), (34, 23))
    battery_start: CellId = (8, 15)

    robots: list[RobotSpec] = []
    for index, start in enumerate(scout_starts, start=1):
        battery = 0.22 if index == 3 else 0.78 + 0.02 * (index % 4)
        robots.append(
            RobotSpec(
                robot_id=index,
                body=BodyType.SCOUT,
                start=start,
                v_max=8.0,
                battery=battery,
                radio_range=80.0,
            )
        )
    for index, start in enumerate(killer_starts, start=11):
        robots.append(
            RobotSpec(
                robot_id=index,
                body=BodyType.KILLER,
                start=start,
                v_max=1.0,
                battery=0.22 if index == 11 else 0.82,
                radio_range=80.0,
            )
        )
    robots.append(
        RobotSpec(
            robot_id=14,
            body=BodyType.BATTERY,
            start=battery_start,
            v_max=1.0,
            battery=0.98,
            radio_range=80.0,
            spare_batteries=3,
        )
    )

    scenario = ScenarioConfig(
        field=field,
        crops=cropped_cells(field, obstacles),
        robots=robots,
        seed=seed,
        initial_infestation=infestation,
        obstacles=obstacles,
        pest_dynamics=PestDynamicsSpec(
            period=TimeUnit.MINUTE,
            spawn_probability=0.00035,
            growth_rate=0.025,
            spread_probability=0.045,
            spawn_severity=0.08,
        ),
        docks=docks,
        energy=EnergySpec(per_metre=2e-6, per_decision=1e-4, treat_per_action=2e-3),
    )

    explore = ExploreConfig(
        revisit_age=20 * TimeUnit.MINUTE,
        rebuild_period=2 * TimeUnit.MINUTE,
        block_side=8,
        spread_peers=True,
        peer_max_age=TimeUnit.MINUTE,
        window_blocks=1,
    )
    arbiter = ArbiterConfig()
    scout_brain = BrainSpec(
        behaviors=("explore", "return-to-dock"),
        seeders=("explore", "dock"),
        params={"explore": explore},
        arbiter=arbiter,
    )
    killer_brain = BrainSpec(
        behaviors=("treat-infestation", "explore", "return-to-dock"),
        seeders=("treat", "explore", "dock"),
        params={"explore": explore},
        arbiter=arbiter,
    )
    battery_brain = BrainSpec(
        behaviors=("supply-battery", "return-to-dock"),
        seeders=("battery-support", "dock"),
        arbiter=arbiter,
    )
    return ExperimentConfig(
        label="Field operations | heterogeneous swarm",
        scenario=scenario,
        brain=scout_brain,
        brains_by_body={BodyType.KILLER: killer_brain, BodyType.BATTERY: battery_brain},
        robots_know_terrain=True,
        stagger_decisions=True,
        until=int(hours * TimeUnit.HOUR),
        metrics=MetricsConfig(
            period=TimeUnit.MINUTE,
            target_threshold=0.05,
            fresh_age=20 * TimeUnit.MINUTE,
        ),
        decision_interval=15 * TimeUnit.SECOND,
    )


def _clock(time: int) -> str:
    seconds = max(time, 0) // TimeUnit.SECOND
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _cell_text(cell: CellId | None) -> str:
    return "—" if cell is None else f"({cell[0]:02d},{cell[1]:02d})"


@dataclass(frozen=True)
class RobotTelemetry:
    """Typed snapshot of robot fields used by the event panel."""

    position: CellId
    battery: float
    failed: bool
    charging: bool
    spare_batteries: int
    status: str
    goal: str | None


class FieldSimulationConsole(InteractiveConsole):
    """An observing frontend over ``LiveRun`` with a potential-field-first visual style."""

    def __init__(self, run: LiveRun, speed: float) -> None:
        from swarm_field.visualization import MatplotlibRenderer

        renderer = MatplotlibRenderer(
            figsize=(15.5, 8.6), show_targets=False, show_metrics=False, show_grid=False
        )
        super().__init__(run, renderer, speed=speed, trail_length=TRAIL_LENGTH, show_targets=False)
        self.figure: Figure = renderer.figure
        self.axes: Axes = renderer.axes
        self.figure.patch.set_facecolor("#ffffff")
        self.figure.subplots_adjust(left=0.055, right=0.655, top=0.865, bottom=0.16)
        self._style_panel: Text = self.figure.text(
            0.685,
            0.895,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.1,
            color="#202a35",
            linespacing=1.18,
        )
        self._subtitle: Text = self.figure.text(
            0.057,
            0.925,
            "Decentralised exploration • infestation treatment • battery support • dock charging",
            ha="left",
            va="center",
            fontsize=9.2,
            color="#586777",
        )
        self.figure.suptitle(
            "JIP SWARM FIELD  /  LIVE SIMULATION",
            x=0.055,
            y=0.975,
            ha="left",
            fontsize=17,
            fontweight="bold",
            color="#17212b",
        )
        self._cbar_ax: Axes = self.figure.add_axes([0.12, 0.073, 0.45, 0.025])
        self._norm: Normalize = Normalize(vmin=-10.0, vmax=50.0)
        self._cmap: Colormap = plt.get_cmap("viridis_r").copy()
        self._cmap.set_bad((0.0, 0.0, 0.0, 0.0))
        self._scalar: ScalarMappable = ScalarMappable(norm=self._norm, cmap=self._cmap)
        self._scalar.set_array(np.asarray([0.0, 1.0]))
        self._colorbar: Colorbar = self.figure.colorbar(
            self._scalar, cax=self._cbar_ax, orientation="horizontal"
        )
        self._colorbar.set_label(
            "Potential cost to selected objective (cells; lower is more attractive)",
            fontsize=8,
            labelpad=2,
        )
        self._robot_labels: dict[int, str] = self._make_robot_labels()
        self._style_trails: dict[int, deque[tuple[float, float]]] = {}
        self._events: deque[str] = deque(maxlen=8)
        self._previous_robots: dict[int, RobotTelemetry] = {}
        self._previous_infestations: dict[CellId, float] = {}
        self._initialized_events: bool = False
        self._cleared_cells: int = 0
        self._current_link_count: int = 0

    def _make_robot_labels(self) -> dict[int, str]:
        counts = {body: 0 for body in BodyType}
        labels: dict[int, str] = {}
        for robot_id, robot in sorted(self.run.context.world.robots.items()):
            counts[robot.body] += 1
            labels[robot_id] = f"{ROLE_PREFIX[robot.body]}{counts[robot.body]:02d}"
        return labels

    @override
    def apply_click(self, x_data: float, y_data: float) -> None:
        """Select the nearest robot, or add an infestation in an empty walkable cell."""
        world = self.run.context.world
        cell = (math.floor(x_data), math.floor(y_data))
        if not world.is_inside(cell):
            return
        click_x, click_y = x_data, y_data
        nearest: tuple[float, int] | None = None
        for robot_id in world.robots:
            rx, ry = world.continuous_position(robot_id, self.time)
            distance = math.dist((click_x, click_y), (rx, ry))
            if nearest is None or distance < nearest[0]:
                nearest = (distance, robot_id)
        if nearest is not None and nearest[0] <= 0.65:
            self.selected = None if self.selected == nearest[1] else nearest[1]
            label = self._robot_labels[nearest[1]]
            self._events.appendleft(f"{_clock(self.time)} focus -> {label}")
            return
        if self.run.replaying or not world.is_walkable(cell):
            return
        self.run.apply_command(InjectInfestation(cell=cell, severity=0.8))
        self._events.appendleft(f"{_clock(self.time)} manual infestation at {_cell_text(cell)}")

    @override
    def draw(self) -> None:
        snapshot = self.run.snapshot()
        self._collect_events(snapshot)
        ax = self.axes
        ax.clear()
        field = snapshot.field
        ax.imshow(
            cell_rgba_image(snapshot),
            origin="lower",
            interpolation="none",
            extent=(0, field.width, 0, field.height),
            zorder=0,
        )

        focus_id, focus_seeder, seeded = self._potential_field(snapshot)
        potential_target: CellId | None = None
        finite: np.ndarray | None = None
        if seeded is not None:
            values = np.full((field.height, field.width), np.nan, dtype=float)
            for y in range(field.height):
                for x in range(field.width):
                    if self.run.context.world.cells[(x, y)].obstacle:
                        continue
                    value = seeded.field.value_at((x, y))
                    if math.isfinite(value):
                        values[y, x] = value
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                low, high = np.percentile(finite_values, (10, 85))
                if high - low < 1e-6:
                    high = low + 1.0
                self._norm.vmin = float(low)
                self._norm.vmax = float(high)
                self._scalar.set_norm(self._norm)
                self._colorbar.update_normal(self._scalar)
                finite = values
                potential_target = seeded.field.target_for(
                    self.run.context.world.robots[focus_id].position
                )
                ax.imshow(
                    np.ma.masked_invalid(values),
                    origin="lower",
                    interpolation="none",
                    extent=(0, field.width, 0, field.height),
                    cmap=self._cmap,
                    norm=self._norm,
                    alpha=0.92,
                    zorder=0.45,
                )

        world = self.run.context.world
        positions = {
            robot_id: world.continuous_position(robot_id, self.time) for robot_id in world.robots
        }
        self._draw_trails(snapshot, positions)
        self._draw_radio_links(positions)
        self._draw_radio_range(focus_id, positions)
        self._draw_targets(snapshot, positions, focus_id)
        self._draw_infestations(snapshot)
        self._draw_docks()
        self._draw_robots(snapshot, positions)

        if seeded is not None and focus_id in self._robot_labels:
            focus_label = self._robot_labels[focus_id]
            ax.set_title(
                f"{focus_label} / {focus_seeder.upper()} potential field   •   "
                f"target {_cell_text(potential_target)}",
                loc="left",
                fontsize=10,
                color="#273746",
                pad=8,
            )
        else:
            ax.set_title("Field state and live robot objectives", loc="left", fontsize=10)

        ax.set_xlim(0, field.width)
        ax.set_ylim(0, field.height)
        ax.set_aspect("equal")
        ax.set_xlabel("Field cell x", fontsize=8)
        ax.set_ylabel("Field cell y", fontsize=8)
        ax.set_xticks(range(0, field.width + 1, 5))
        ax.set_yticks(range(0, field.height + 1, 5))
        ax.set_xticks(range(field.width + 1), minor=True)
        ax.set_yticks(range(field.height + 1), minor=True)
        ax.tick_params(labelsize=7.5, which="major", colors="#485765")
        ax.tick_params(which="minor", length=0)
        ax.grid(which="minor", color="#17212b", linewidth=0.33, alpha=0.22)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#7e8b97")
            spine.set_linewidth(0.7)

        self._style_panel.set_text("\n".join(self._panel_lines(snapshot, focus_id, focus_seeder)))
        self._style_panel.set_visible(self.show_info)
        self._subtitle.set_visible(self.show_info)
        self._cbar_ax.set_visible(finite is not None)
        self._colorbar.set_ticks(np.linspace(self._norm.vmin, self._norm.vmax, 5))
        self.figure.canvas.draw_idle()

    def _potential_field(self, snapshot: WorldSnapshot) -> tuple[int, str, SeededField | None]:
        agents = self.run.agents()
        chosen = self.selected if self.selected in agents else None
        if chosen is None:
            killers = [r for r in snapshot.robots if r.body is BodyType.KILLER and not r.failed]
            focused_killers = [
                robot
                for robot in killers
                if agents[robot.robot_id].inspect().current_goal == "treat-infestation"
            ]
            candidates = focused_killers or killers
            scouts = [r for r in snapshot.robots if r.body is BodyType.SCOUT and not r.failed]
            batteries = [r for r in snapshot.robots if r.body is BodyType.BATTERY and not r.failed]
            order = candidates or scouts or batteries
            if not order:
                return -1, "none", None
            chosen = order[0].robot_id

        public = next((r for r in snapshot.robots if r.robot_id == chosen), None)
        if public is None or public.failed:
            return chosen, "none", None
        agent = agents.get(chosen)
        if agent is None:
            return chosen, "none", None

        # Only render the field belonging to the agent's currently winning behaviour. Do not
        # build a fallback field for another behaviour: architecture docs specify that field
        # construction follows arbitration, rather than influencing behaviour selection.
        goal = agent.inspect().current_goal
        seeder = SEEDER_FOR_GOAL.get(goal or "")
        if seeder is None:
            return chosen, "none", None
        robot_state = self.run.context.world.robots[chosen]
        try:
            field = agent.fields.field_for(seeder, robot_state.position, self.time)
        except KeyError:
            return chosen, seeder, None
        if not isinstance(field, SeededField):
            # A hop field gives a heading, not a field over the cells, so there is none to draw.
            return chosen, seeder, None
        return chosen, seeder, field

    def _draw_trails(
        self, snapshot: WorldSnapshot, positions: dict[int, tuple[float, float]]
    ) -> None:
        ax = self.axes
        active_ids = {robot.robot_id for robot in snapshot.robots}
        for robot_id in list(self._style_trails):
            if robot_id not in active_ids:
                del self._style_trails[robot_id]
        for robot in snapshot.robots:
            trail = self._style_trails.setdefault(robot.robot_id, deque(maxlen=TRAIL_LENGTH))
            pos = positions[robot.robot_id]
            if not trail or math.dist(trail[-1], pos) > 0.06:
                trail.append(pos)
            if len(trail) > 1:
                ax.plot(
                    [point[0] for point in trail],
                    [point[1] for point in trail],
                    color=ROLE_COLOURS[robot.body],
                    linewidth=1.0,
                    alpha=0.42,
                    zorder=0.75,
                )

    def _draw_radio_links(self, positions: dict[int, tuple[float, float]]) -> None:
        world = self.run.context.world
        ids = sorted(world.robots)
        links = 0
        for index, sender_id in enumerate(ids):
            sender = world.robots[sender_id]
            if sender.failed or sender.radio_range_m <= 0:
                continue
            for receiver_id in ids[index + 1 :]:
                receiver = world.robots[receiver_id]
                if receiver.failed:
                    continue
                distance_m = (
                    math.dist(positions[sender_id], positions[receiver_id]) * world.field.cell_size
                )
                if distance_m <= max(sender.radio_range_m, receiver.radio_range_m):
                    x0, y0 = positions[sender_id]
                    x1, y1 = positions[receiver_id]
                    self.axes.plot(
                        [x0, x1],
                        [y0, y1],
                        color="#f5f7fa",
                        linewidth=0.65,
                        alpha=0.50,
                        zorder=0.82,
                    )
                    links += 1
        self._current_link_count = links

    def _draw_radio_range(self, focus_id: int, positions: dict[int, tuple[float, float]]) -> None:
        world = self.run.context.world
        if focus_id not in world.robots:
            return
        robot = world.robots[focus_id]
        if robot.radio_range_m <= 0:
            return
        x, y = positions[focus_id]
        self.axes.add_patch(
            Circle(
                (x, y),
                robot.radio_range_m / world.field.cell_size,
                fill=False,
                edgecolor=ROLE_COLOURS[robot.body],
                linewidth=0.9,
                linestyle=(0, (2, 2)),
                alpha=0.55,
                zorder=0.9,
            )
        )

    def _draw_targets(
        self,
        snapshot: WorldSnapshot,
        positions: dict[int, tuple[float, float]],
        focus_id: int,
    ) -> None:
        targets = self.run.targets()
        by_id = {robot.robot_id: robot for robot in snapshot.robots}
        for robot_id, target in targets.items():
            public = by_id.get(robot_id)
            if public is None or public.failed:
                continue
            # Scouts still use their genuine EXPLORE target; omitting all but the focused scout's
            # line keeps the field view close to the reference style rather than filling it with
            # long exploration vectors. The working and support objectives remain explicit.
            if public.body is BodyType.SCOUT and robot_id != focus_id:
                continue
            x, y = positions[robot_id]
            colour = ROLE_COLOURS[public.body]
            focused = robot_id == focus_id
            self.axes.plot(
                [x, target[0] + 0.5],
                [y, target[1] + 0.5],
                linestyle="-",
                linewidth=1.8 if focused or public.body is BodyType.KILLER else 1.15,
                color=colour,
                alpha=0.9 if focused or public.body is BodyType.KILLER else 0.62,
                zorder=1.3,
            )
            if focused:
                self.axes.scatter(
                    [target[0] + 0.5],
                    [target[1] + 0.5],
                    marker="*",
                    s=105,
                    color=colour,
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=2.4,
                )
            else:
                self.axes.scatter(
                    [target[0] + 0.5],
                    [target[1] + 0.5],
                    marker="x",
                    s=24,
                    color=colour,
                    linewidths=1.2,
                    zorder=2.4,
                )

    def _draw_infestations(self, snapshot: WorldSnapshot) -> None:
        threshold = self.run.config.metrics.target_threshold
        for cell, state in snapshot.cells.items():
            # Outlines emphasize active hotspots, not every low-severity cell in a spreading patch.
            if state.obstacle or state.infestation < max(threshold, 0.30):
                continue
            self.axes.add_patch(
                Rectangle(
                    cell,
                    1,
                    1,
                    fill=False,
                    edgecolor="#e53935",
                    linewidth=1.0,
                    alpha=0.92,
                    zorder=2.0,
                )
            )

    def _draw_docks(self) -> None:
        for x, y in self.run.config.scenario.docks:
            self.axes.scatter(
                [x + 0.5],
                [y + 0.5],
                marker="*",
                s=235,
                color="#18a999",
                edgecolors="white",
                linewidths=0.8,
                zorder=3.2,
            )

    def _draw_robots(
        self, snapshot: WorldSnapshot, positions: dict[int, tuple[float, float]]
    ) -> None:
        grouped: dict[tuple[int, int], list[int]] = {}
        for robot in snapshot.robots:
            x, y = positions[robot.robot_id]
            key = (round(x * 10), round(y * 10))
            grouped.setdefault(key, []).append(robot.robot_id)
        display: dict[int, tuple[float, float]] = {}
        for ids in grouped.values():
            for i, robot_id in enumerate(ids):
                x, y = positions[robot_id]
                if len(ids) > 1:
                    angle = 2.0 * math.pi * i / len(ids)
                    x += 0.45 * math.cos(angle)
                    y += 0.45 * math.sin(angle)
                display[robot_id] = (x, y)

        for robot in snapshot.robots:
            x, y = display[robot.robot_id]
            actual_x, actual_y = positions[robot.robot_id]
            if math.dist((x, y), (actual_x, actual_y)) > 0.01:
                self.axes.plot(
                    [actual_x, x],
                    [actual_y, y],
                    color="#edf2f4",
                    linewidth=0.75,
                    alpha=0.85,
                    zorder=4.8,
                )
            if robot.failed:
                self.axes.scatter(
                    [x],
                    [y],
                    marker="X",
                    s=150,
                    color="#40464d",
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=5.0,
                )
                label_colour = "white"
            else:
                self.axes.scatter(
                    [x],
                    [y],
                    marker=ROLE_MARKERS[robot.body],
                    s=155,
                    color=ROLE_COLOURS[robot.body],
                    edgecolors="white",
                    linewidths=1.2,
                    zorder=5.0,
                )
                label_colour = "white"
            actual_x, actual_y = positions[robot.robot_id]
            label_x, label_y = x, y
            displacement = math.dist((actual_x, actual_y), (x, y))
            if displacement > 0.01:
                label_x += 0.28 * (x - actual_x) / displacement
                label_y += 0.28 * (y - actual_y) / displacement
            self.axes.text(
                label_x,
                label_y,
                self._robot_labels[robot.robot_id],
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="bold",
                color=label_colour,
                zorder=6.0,
                path_effects=[patheffects.withStroke(linewidth=0.6, foreground="#1c2730")],
            )
            # Small battery bar above each symbol makes energy state visible without replacing
            # the ground-truth position or the action chosen by the agent.
            bar_width = 0.60
            bar_y = y + 0.31
            self.axes.add_patch(
                Rectangle((x - bar_width / 2, bar_y), bar_width, 0.075, color="#f1f3f5", zorder=6.2)
            )
            if not robot.failed:
                battery_colour = "#2a9d8f" if robot.battery > LOW_BATTERY_THRESHOLD else "#e76f51"
                self.axes.add_patch(
                    Rectangle(
                        (x - bar_width / 2, bar_y),
                        bar_width * max(0.0, min(1.0, robot.battery)),
                        0.075,
                        color=battery_colour,
                        zorder=6.3,
                    )
                )

    def _status_for(self, robot_id: int) -> str:
        world = self.run.context.world
        state = world.robots[robot_id]
        if state.failed:
            return "DEAD_BATTERY" if state.failure_mode is not None else "FAILED"
        if state.replacing_battery_for is not None:
            return f"REPLACING->{self._robot_labels.get(state.replacing_battery_for, '?')}"
        if state.charging_updated_at is not None:
            return "CHARGING"
        if state.treating_cell is not None:
            return f"TREAT@{_cell_text(state.treating_cell)}"
        if state.move_target is not None:
            return f"MOVE->{_cell_text(state.move_target)}"
        agent = self.run.agents().get(robot_id)
        goal = None if agent is None else agent.inspect().current_goal
        return {
            "explore": "EXPLORE",
            "treat-infestation": "TREAT",
            "supply-battery": "SUPPORT",
            "return-to-dock": "TO DOCK",
        }.get(goal or "", "IDLE")

    def _panel_lines(self, snapshot: WorldSnapshot, focus_id: int, focus_seeder: str) -> list[str]:
        world = self.run.context.world
        total = len(snapshot.robots)
        active = sum(not robot.failed for robot in snapshot.robots)
        counts = {body: sum(robot.body is body for robot in snapshot.robots) for body in BodyType}
        low = sum(not r.failed and r.battery <= LOW_BATTERY_THRESHOLD for r in snapshot.robots)
        failed = total - active
        charging = sum(robot.charging_updated_at is not None for robot in world.robots.values())
        active_infestations = [
            (cell, state.infestation)
            for cell, state in world.cells.items()
            if state.infestation > self.run.config.metrics.target_threshold and not state.obstacle
        ]
        sample = self.run.metrics.latest()
        latest_coverage = 0.0 if sample is None else sample.coverage
        targets = len(active_infestations) if sample is None else sample.targets_remaining
        current_goal = "—"
        current_target = None
        known = 0
        if focus_id in self.run.agents():
            agent = self.run.agents()[focus_id]
            inspect = agent.inspect()
            current_goal = inspect.current_goal or "—"
            current_target = inspect.current_target
            known = sum(fact.severity > 0 for fact in agent.memory.infestations())
        focus_label = self._robot_labels.get(focus_id, "—")
        battery_states = (
            " ".join(
                f"{self._robot_labels[r.robot_id]}:{r.battery:.0%}"
                for r in snapshot.robots
                if r.body is BodyType.BATTERY
            )
            or "none"
        )
        total_spares = sum(
            state.spare_batteries
            for state in world.robots.values()
            if state.body is BodyType.BATTERY
        )
        spare_capacity = sum(3 for state in world.robots.values() if state.body is BodyType.BATTERY)
        lines = [
            "RUN / ARCHITECTURE",
            f"time       {_clock(self.time)} / {_clock(self.run.horizon)}",
            f"speed      {self.speed:g}x   {'PAUSED' if self.paused else 'RUNNING'}",
            f"field      {snapshot.field.width}x{snapshot.field.height} cells",
            f"cell size  {snapshot.field.cell_size:g} m",
            "pipeline   LiveRun > SimulationLoop",
            "           ComposedAgent > UtilityArbiter",
            "           FieldProvider > Navigator",
            "",
            "SWARM / WORLD",
            f"SCOUT {counts[BodyType.SCOUT]:2d} | KILLER {counts[BodyType.KILLER]:2d} | "
            f"BATTERY {counts[BodyType.BATTERY]:2d}",
            f"active     {active:2d}/{total:2d}   failed {failed}",
            f"infested   {len(active_infestations):3d} cells  | targets {targets:3d}",
            f"cleared    {self._cleared_cells:3d} observed",
            f"coverage   {latest_coverage:6.1%}",
            f"radio      sent {world_message_count(self.run, 'sent'):5d} | "
            f"received {world_message_count(self.run, 'received'):5d}",
            f"links      {getattr(self, '_current_link_count', 0):3d} visible pairs",
            "",
            "BATTERY LIFECYCLE",
            f"low peers  {low:2d} | charging {charging:2d}",
            f"failures   {world.battery_failure_count():2d} | "
            f"spares {total_spares}/{spare_capacity}",
            f"support    {battery_states}",
            "",
            f"FOCUS / {focus_label}  FIELD={focus_seeder.upper()}",
            f"goal       {current_goal}",
            f"target     {_cell_text(current_target)} | known pests {known}",
            "",
            "ROBOTS  (battery / current state)",
        ]
        scout_ids = [r.robot_id for r in snapshot.robots if r.body is BodyType.SCOUT]
        battery_by_id = {robot.robot_id: robot.battery for robot in snapshot.robots}
        for index in range(0, len(scout_ids), 2):
            pair = scout_ids[index : index + 2]
            pieces = [
                f"{self._robot_labels[rid]} {battery_by_id[rid]:.0%} {self._status_for(rid)}"
                for rid in pair
            ]
            lines.append("  ".join(pieces))
        for robot in snapshot.robots:
            if robot.body is BodyType.KILLER:
                lines.append(
                    f"{self._robot_labels[robot.robot_id]} {robot.battery:.0%} "
                    f"{self._status_for(robot.robot_id)}"
                )
        for robot in snapshot.robots:
            if robot.body is BodyType.BATTERY:
                lines.append(
                    f"{self._robot_labels[robot.robot_id]} {robot.battery:.0%} "
                    f"{self._status_for(robot.robot_id)}"
                )
        lines.extend(["", "WHAT HAPPENED"])
        lines.extend(f"  {event}" for event in list(self._events)[:6])
        lines.extend(
            [
                "",
                "HOW TO READ",
                "yellow = lower potential cost",
                "red squares = infestation | grey = obstacle",
                "stars = docks / target   dashed ring = focus radio range",
                "controls: space pause | +/- speed | s step | h hide",
                "click robot = focus field | click empty cell = add pest",
            ]
        )
        return lines

    def _collect_events(self, snapshot: WorldSnapshot) -> None:
        world = self.run.context.world
        current: dict[int, RobotTelemetry] = {}
        for robot in snapshot.robots:
            state = world.robots[robot.robot_id]
            agent = self.run.agents().get(robot.robot_id)
            goal = None if agent is None else agent.inspect().current_goal
            current[robot.robot_id] = RobotTelemetry(
                position=robot.position,
                battery=robot.battery,
                failed=robot.failed,
                charging=state.charging_updated_at is not None,
                spare_batteries=state.spare_batteries,
                status=self._status_for(robot.robot_id),
                goal=goal,
            )
        infestations = {
            cell: state.infestation
            for cell, state in world.cells.items()
            if not state.obstacle and state.infestation > self.run.config.metrics.target_threshold
        }
        if not self._initialized_events:
            self._events.appendleft(
                f"{_clock(self.time)} started with {len(snapshot.robots)} robots and "
                f"{len(infestations)} infested cells"
            )
            for robot in snapshot.robots:
                if not robot.failed and robot.battery <= LOW_BATTERY_THRESHOLD:
                    self._events.appendleft(
                        f"{_clock(self.time)} {self._robot_labels[robot.robot_id]} "
                        f"starts low ({robot.battery:.0%})"
                    )
            self._initialized_events = True
        else:
            previous_batteries: dict[int, float] = {
                robot_id: state.battery for robot_id, state in self._previous_robots.items()
            }
            for robot in snapshot.robots:
                robot_id = robot.robot_id
                now_state = current[robot_id]
                old_state = self._previous_robots.get(robot_id)
                label = self._robot_labels[robot_id]
                if old_state is not None:
                    if not old_state.failed and robot.failed:
                        self._events.appendleft(f"{_clock(self.time)} {label} -> DEAD_BATTERY")
                    if old_state.battery > LOW_BATTERY_THRESHOLD >= robot.battery:
                        self._events.appendleft(
                            f"{_clock(self.time)} {label} battery low ({robot.battery:.0%})"
                        )
                    if not old_state.charging and now_state.charging:
                        self._events.appendleft(
                            f"{_clock(self.time)} {label} started dock charging"
                        )
                    if now_state.spare_batteries < old_state.spare_batteries:
                        recipients = [
                            other
                            for other in snapshot.robots
                            if other.robot_id != robot_id
                            and other.battery
                            - previous_batteries.get(other.robot_id, other.battery)
                            > 0.15
                        ]
                        if recipients:
                            recipient = max(
                                recipients,
                                key=lambda other: (
                                    other.battery
                                    - previous_batteries.get(other.robot_id, other.battery)
                                ),
                            )
                            self._events.appendleft(
                                f"{_clock(self.time)} {label} replaced "
                                f"{self._robot_labels[recipient.robot_id]}'s battery"
                            )
                        else:
                            self._events.appendleft(
                                f"{_clock(self.time)} {label} used a spare battery"
                            )
                    if old_state.goal != now_state.goal and now_state.goal in (
                        "treat-infestation",
                        "supply-battery",
                        "return-to-dock",
                    ):
                        self._events.appendleft(
                            f"{_clock(self.time)} {label} objective -> {now_state.goal}"
                        )
            cleared = [cell for cell in self._previous_infestations if cell not in infestations]
            for cell in cleared[:2]:
                self._cleared_cells += 1
                self._events.appendleft(
                    f"{_clock(self.time)} infestation cleared at {_cell_text(cell)}"
                )
            appeared = [cell for cell in infestations if cell not in self._previous_infestations]
            if appeared:
                self._events.appendleft(
                    f"{_clock(self.time)} new infestation at {_cell_text(appeared[0])}"
                )
        self._previous_robots = current
        self._previous_infestations = infestations


def world_message_count(run: LiveRun, direction: str) -> int:
    channel = run.context.channel
    return int(getattr(channel, direction, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=77, help="reproducibility seed")
    parser.add_argument("--hours", type=float, default=3.0, help="simulated duration")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="simulation speed")
    parser.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="save one PNG snapshot and exit (the default opens the live window)",
    )
    parser.add_argument(
        "--preview-minutes", type=float, default=15.0, help="simulation time for --preview"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hours <= 0:
        raise SystemExit("--hours must be positive")
    if args.speed <= 0:
        raise SystemExit("--speed must be positive")
    run = LiveRun.from_config(build_experiment(seed=args.seed, hours=args.hours))
    console = FieldSimulationConsole(run, speed=args.speed)
    if args.preview is not None:
        console.draw()
        run.advance_to(min(int(args.preview_minutes * TimeUnit.MINUTE), run.horizon))
        console.draw()
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        console.figure.savefig(args.preview, dpi=150, bbox_inches="tight")
        plt.close(console.figure)
        print(f"Saved preview to {args.preview}")
        return
    show_window(console)


if __name__ == "__main__":
    main()
