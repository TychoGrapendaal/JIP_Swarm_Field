"""One killer follows the hop field over a lattice of relays, drawn with the overlays.

An explainer for the hop-based potential field (#70). A lattice of relay scouts stands still
over a field with a river and two rocks; in the future real explorers take their place. A mock
source on every pest beacons a hop value of 0, and every relay passes on the lowest value it
hears plus the hop to it. The single killer knows of a pest only through these beacons.

Red cells are the pests in ground truth. The big red dot is the killer, the small grey dots are
the relays. Dots labelled with a hop value are the beacons the killer hears, orange where they
pull it. Orange arrows are the pull of each of those, the thick red arrow is their sum, the
direction the killer walks. With ``--escape``, a killer that circles in place for ten minutes
places an ever stronger escape beacon on the place it circles around (purple dots), and the
purple arrow is their summed push. With ``--show-broadcasts``, blue arrows show a hop value
being passed on: from the beacon a robot took its value from, to the robot, at the moment its
value changes.

Run from the repository root with:
    uv run python examples/hop_explainer.py

Record a video instead, or PNG frames at chosen minutes:
    uv run python examples/hop_explainer.py --show-broadcasts --save videos/hop_explainer.mp4
    uv run python examples/hop_explainer.py --frames-at 5 20 --save videos/hop_explainer_frames
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from math import hypot
from pathlib import Path

from swarm_field.contracts import (
    BeaconKind,
    BodyType,
    CellId,
    RobotId,
    Time,
    TimeUnit,
    WorldSnapshot,
)
from swarm_field.experiments import (
    FAST_TREATMENT,
    RELAY,
    HopKillerPreset,
    LiveRun,
    MapConfig,
    PestConfig,
    RobotConfig,
    SwarmScenario,
)
from swarm_field.swarm import (
    BeaconConfig,
    BeaconMemory,
    BeaconView,
    ComposedAgent,
    EscapeConfig,
    treat_beacon,
)
from swarm_field.swarm.hop import (
    beacon_strength,
    downhill_beacons,
    fresh_beacons,
    hop_pull,
    obstacle_push,
    own_value,
)
from swarm_field.visualization import (
    Arrow,
    Marker,
    MatplotlibRenderer,
    RobotStyle,
    SidePanel,
    draw_arrows,
    draw_markers,
    draw_targets,
    save_frames,
)

OBSTACLE_SHOWN_DECAYS = 3.0
"""Obstacle beacons older than this many decay times push next to nothing and are not drawn."""
PULL_ARROW_CELLS = 4.0
"""How long the arrow of the killer's summed pull is drawn, in cells."""
BROADCAST_FRAMES = 3
"""For how many frames a passed-on hop value stays drawn, fading out."""
MIN_DROP = 0.05
"""How much a robot's hop value must fall to count as new information passed on."""


def parse_args() -> argparse.Namespace:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--relays", type=int, default=25)
    parser.add_argument("--radio", type=float, default=100.0, help="radio range in metres")
    parser.add_argument("--pests-per-hour", type=float, default=6.0)
    parser.add_argument("--cap", type=float, default=8.0, help="the highest hop value relayed")
    parser.add_argument(
        "--escape", action="store_true", help="a killer circling in place pushes itself out"
    )
    parser.add_argument("--show-broadcasts", action="store_true", help="draw passed-on values")
    parser.add_argument("--video-seconds", type=float, default=45.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-seconds", type=float, default=5.0, help="simulated per frame")
    parser.add_argument(
        "--start-minute", type=float, default=0.0, help="simulated minute of the first frame"
    )
    parser.add_argument(
        "--frames-at", type=float, nargs="+", help="draw only these minutes, e.g. a smoke test"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=Path, help="write a .mp4, a .gif or a directory of PNGs")
    return parser.parse_args()


class HopExplainer:
    """The arrows and the panel of the explainer, read from the robots' memories each frame.

    Attributes:
        show_broadcasts: Whether to draw hop values being passed on.
        treated: How many pests have been cleared so far.
    """

    def __init__(self, run: LiveRun, config: BeaconConfig, show_broadcasts: bool) -> None:
        self.show_broadcasts: bool = show_broadcasts
        self.treated: int = 0
        self._run: LiveRun = run
        self._config: BeaconConfig = config
        self._values: dict[RobotId, float] = {}
        self._broadcasts: list[tuple[Arrow, int]] = []
        self._infested: set[CellId] = set()

    def arrows(self, snapshot: WorldSnapshot) -> list[Arrow]:
        """The killer's pull and, with ``show_broadcasts``, the values passed on lately."""
        self._count_treated(snapshot)
        found: list[Arrow] = []
        if self.show_broadcasts:
            found.extend(self._broadcast_arrows(snapshot))
        for robot in snapshot.robots:
            if robot.body is BodyType.KILLER:
                found.extend(self._pull_arrows(robot.robot_id, robot.position, snapshot.time))
        return found

    def markers(self, snapshot: WorldSnapshot) -> list[Marker]:
        """The beacons the killer hears, labelled with their hop value; the pulling ones orange."""
        found: list[Marker] = []
        agents = self._run.agents()
        for robot in snapshot.robots:
            if robot.body is not BodyType.KILLER:
                continue
            memory = _beacons_of(agents[robot.robot_id])
            now = snapshot.time
            pulling: set[CellId] = set()
            for heard in downhill_beacons(memory, robot.position, now, self._config):
                pulling.add(heard.beacon.cell)
            for heard in fresh_beacons(memory, BeaconKind.TREAT, now, self._config):
                cell = heard.beacon.cell
                color = "khaki"
                if cell in pulling:
                    color = "darkorange"
                position = (cell[0] + 0.5, cell[1] + 0.5)
                found.append(Marker(position, color, size=45.0, label=f"{heard.beacon.value:.1f}"))
            for placed in memory.placed_beacons():
                cell = placed.cell
                age = (now - placed.placed_at) / placed.decay
                if age > OBSTACLE_SHOWN_DECAYS:
                    continue
                position = (cell[0] + 0.5, cell[1] + 0.5)
                label = f"{placed.strength:g}"
                found.append(Marker(position, "mediumorchid", size=35.0, label=label))
        return found

    def panel(self, snapshot: WorldSnapshot) -> list[str]:
        """What the killer knows and does, and a legend."""
        lines = [
            f"time            {_clock(snapshot.time)}",
            f"pests now       {len(self._infested)}",
            f"pests treated   {self.treated}",
            "",
        ]
        agents = self._run.agents()
        for robot in snapshot.robots:
            if robot.body is not BodyType.KILLER:
                continue
            agent = agents[robot.robot_id]
            now = snapshot.time
            heard = fresh_beacons(_beacons_of(agent), BeaconKind.TREAT, now, self._config)
            value = "-"
            if heard:
                value = f"{own_value(heard, robot.position, self._config):.2f}"
            goal = agent.inspect().current_goal or "idle"
            lines.extend(
                [
                    f"killer {robot.robot_id}",
                    f"  goal          {goal}",
                    f"  beacons heard {len(heard)}",
                    f"  own hop value {value}",
                    "",
                ]
            )
        lines.extend(
            [
                "big red dot     the killer",
                "grey dots       relays",
                "orange dots     beacons pulling the killer",
                "pale dots       beacons it hears, not pulling",
                "numbers         their hop values",
                "red cells       pests",
                "orange arrows   pull of each beacon",
                "red arrow       summed pull",
                "purple dots     escape beacons, with push",
                "purple arrow    their push",
            ]
        )
        if self.show_broadcasts:
            lines.append("blue arrows     hop value passed on")
        return lines

    def _count_treated(self, snapshot: WorldSnapshot) -> None:
        """Count the pests cleared since the last frame; only treatment clears a pest."""
        infested: set[CellId] = set()
        for cell, state in snapshot.cells.items():
            if state.infestation > 0:
                infested.add(cell)
        self.treated += len(self._infested - infested)
        self._infested = infested

    def _pull_arrows(self, killer: RobotId, position: CellId, now: Time) -> list[Arrow]:
        """The killer's summed pull, and on top one arrow per beacon pulling it, by strength."""
        memory = _beacons_of(self._run.agents()[killer])
        pull = hop_pull(memory, position, now, self._config)
        if pull is None:
            return []
        start = (position[0] + 0.5, position[1] + 0.5)
        length = hypot(pull.vector[0], pull.vector[1])
        end = (
            start[0] + pull.vector[0] / length * PULL_ARROW_CELLS,
            start[1] + pull.vector[1] / length * PULL_ARROW_CELLS,
        )
        arrows = [Arrow(start, end, "red", width=3.0)]
        push = obstacle_push(memory, position, now, self._config)
        push_length = hypot(push[0], push[1])
        if push_length > 0:
            scale = PULL_ARROW_CELLS * push_length / pull.strength
            tip = (
                start[0] + push[0] / push_length * scale,
                start[1] + push[1] / push_length * scale,
            )
            arrows.append(Arrow(start, tip, "mediumorchid", width=2.0))
        for heard in downhill_beacons(memory, position, now, self._config):
            cell = heard.beacon.cell
            dx = cell[0] - position[0]
            dy = cell[1] - position[1]
            scale = PULL_ARROW_CELLS * beacon_strength(heard, now, self._config) / pull.strength
            distance = hypot(dx, dy)
            tip = (start[0] + dx / distance * scale, start[1] + dy / distance * scale)
            arrows.append(Arrow(start, tip, "darkorange", width=1.5))
        return arrows

    def _broadcast_arrows(self, snapshot: WorldSnapshot) -> list[Arrow]:
        """Arrows from where each robot's new hop value came from, fading over a few frames."""
        agents = self._run.agents()
        for robot in snapshot.robots:
            agent = agents[robot.robot_id]
            memory = _beacons_of(agent)
            now = snapshot.time
            beacon = treat_beacon(memory, robot.position, now, self._config)
            if beacon is None:
                self._values.pop(robot.robot_id, None)
                continue
            previous = self._values.get(robot.robot_id)
            self._values[robot.robot_id] = beacon.value
            if beacon.value == 0 or (previous is not None and beacon.value > previous - MIN_DROP):
                continue
            origin = _value_origin(memory, robot.position, now, self._config)
            if origin is None:
                continue
            end = (robot.position[0] + 0.5, robot.position[1] + 0.5)
            arrow = Arrow((origin[0] + 0.5, origin[1] + 0.5), end, "royalblue", width=1.0)
            self._broadcasts.append((arrow, 0))

        drawn: list[Arrow] = []
        kept: list[tuple[Arrow, int]] = []
        for arrow, age in self._broadcasts:
            fade = 1.0 - age / BROADCAST_FRAMES
            drawn.append(Arrow(arrow.start, arrow.end, arrow.color, arrow.width, fade))
            if age + 1 < BROADCAST_FRAMES:
                kept.append((arrow, age + 1))
        self._broadcasts = kept
        return drawn


def _beacons_of(agent: ComposedAgent) -> BeaconMemory:
    """A robot's beacons; every robot of the explainer relays them."""
    if agent.beacon_link is None:
        raise ValueError(f"robot {agent.robot_id} keeps no beacons")
    return agent.beacon_link.memory


def _value_origin(
    memory: BeaconView, position: CellId, now: Time, config: BeaconConfig
) -> CellId | None:
    """The cell of the beacon a robot's own value comes from: the lowest value plus hop."""
    origin: CellId | None = None
    lowest = 0.0
    for heard in fresh_beacons(memory, BeaconKind.TREAT, now, config):
        value = own_value([heard], position, config)
        if origin is None or value < lowest:
            origin = heard.beacon.cell
            lowest = value
    return origin


def _clock(time: float) -> str:
    """A time as hours, minutes and seconds."""
    seconds = int(time // TimeUnit.SECOND)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def frame_times(args: argparse.Namespace) -> Sequence[Time]:
    """The simulated time of every frame: the minutes asked for, or a video's worth from the start
    minute on."""
    if args.frames_at:
        return [int(minute * TimeUnit.MINUTE) for minute in args.frames_at]
    start = int(args.start_minute * TimeUnit.MINUTE)
    step = int(args.frame_seconds * TimeUnit.SECOND)
    frames = int(args.video_seconds * args.fps)
    return [start + index * step for index in range(frames)]


def main() -> None:
    args = parse_args()
    escape: EscapeConfig | None = None
    if args.escape:
        escape = EscapeConfig()
    field = MapConfig()
    config = BeaconConfig(hop_cells=args.radio / field.cell_size, max_value=args.cap, escape=escape)
    pests = PestConfig(
        pests_per_hour=args.pests_per_hour,
        treatment_rate=FAST_TREATMENT,
        source_range_m=args.radio,
    )
    robots = RobotConfig(radio_range_m=args.radio, knows_terrain=False, lattice_starts=True)
    scenario = (
        SwarmScenario(map=field, pests=pests, robots=robots, seed=args.seed)
        .add_robots(RELAY, args.relays)
        .add_robots(HopKillerPreset(beacons=config), 1)
    )
    run = LiveRun.from_config(scenario.into_experiment("hop explainer"))
    explainer = HopExplainer(run, config, args.show_broadcasts)

    def killer_targets() -> dict[RobotId, CellId]:
        targets = run.targets()
        killers: dict[RobotId, CellId] = {}
        for robot_id, robot in run.context.world.robots.items():
            if robot.body is BodyType.KILLER and robot_id in targets:
                killers[robot_id] = targets[robot_id]
        return killers

    renderer = MatplotlibRenderer(
        (12.0, 8.0),
        show_targets=False,
        body_styles={
            BodyType.KILLER: RobotStyle("red", size=160.0, label="killer"),
            BodyType.SCOUT: RobotStyle("slategray", size=40.0, alpha=0.7, label="relay"),
        },
    )
    panel = SidePanel(renderer)
    title = (
        f"#70 hop-based potential field: 1 killer, {args.relays} relays, "
        f"{args.radio:g} m radio, seed {args.seed}"
    )
    times = frame_times(args)

    def draw_frame(index: int) -> None:
        snapshot = run.snapshot_at(times[index])
        renderer.render(snapshot)
        axes = renderer.axes
        draw_targets(axes, snapshot, killer_targets())
        draw_markers(axes, explainer.markers(snapshot))
        draw_arrows(axes, explainer.arrows(snapshot))
        axes.set_title(f"{title}\n{axes.get_title()}")
        panel.show(explainer.panel(snapshot))

    if args.save is not None:
        save_frames(renderer.figure, draw_frame, len(times), args.save, fps=args.fps)
        return
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    animation = FuncAnimation(
        renderer.figure, draw_frame, frames=len(times), interval=1000 / args.fps, repeat=False
    )
    _ = animation
    plt.show()


if __name__ == "__main__":
    main()
