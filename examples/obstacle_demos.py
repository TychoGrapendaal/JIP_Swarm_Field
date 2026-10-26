"""Demo videos of the obstacle comparison (#70): record a run once, render it as often as wanted.

Each pest response of ``CONFIGS``, from ``pest_response_configurations``, runs on ``RIVER_SMALL``
or ``RIVERS_LARGE``; ``--config seeded-no-yield`` records the seeded response without the claims.
``record`` runs it once through ``LiveRun``, one frame per decision interval, and writes a
``Trace``: everything the videos might show, so that new visuals need no new run. ``render``
turns traces into videos with ``MatplotlibRenderer`` and the overlays, without the simulation.

Red cells are the pests in ground truth. Big red dots are the killers, small grey dots the
relays. A black arrow points from each killer to where it walks: along its heading for the hop
configurations, to its target cell for the seeded one. Purple dots are the escape beacons a
killer of ``hop-tuned`` placed where it circled. The number next to a killer is the doses of
treatment it has left, and a gold square marks a pest several killers treat at once.

With ``--explain``, ``record`` also keeps why each killer walks where it does, and ``render``
draws it. For a hop killer: the beacons it hears with their hop values, the pull of each one,
and their sum, its heading. For a seeded killer: its TREAT field over the quadtree as the
overlay, and its route through the field to its target. Blue relays show how the news of a pest
spreads: those with a hop value, or those that know of pests.

Run from the repository root, e.g.:

    uv run python examples/obstacle_demos.py record --seed 0
    uv run python examples/obstacle_demos.py render runs/70-obstacle-comparison/traces
    uv run python examples/obstacle_demos.py record --scenario rivers-large \\
        --config seeded hop-tuned --killers 1 --pests-per-hour 3 --hours 6 --explain \\
        --treat-window-blocks 0
    uv run python examples/obstacle_demos.py render --fps 20 \\
        runs/70-obstacle-comparison/traces/explainer_1-killer_rivers-large_seeded_seed0.pkl.gz
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from math import hypot, inf
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import (
    BeaconKind,
    BodyType,
    CellId,
    CellState,
    FieldGeometry,
    RobotId,
    RobotPublicState,
    Time,
    TimeUnit,
    Vector2,
    WorldSnapshot,
)
from swarm_field.experiments import (
    RIVER_SMALL,
    RIVERS_LARGE,
    SEEDED_KILLER,
    LiveRun,
    SeededKillerPreset,
    SwarmScenario,
    pest_response_configurations,
)
from swarm_field.swarm import BeaconLink, Heading, SeededField, treat_beacon
from swarm_field.swarm.field import Leaf
from swarm_field.swarm.hop import (
    beacon_strength,
    downhill_beacons,
    fresh_beacons,
    obstacle_push,
    obstacle_strength,
)
from swarm_field.visualization import (
    Arrow,
    Marker,
    MatplotlibRenderer,
    Polyline,
    RobotStyle,
    SidePanel,
    draw_arrows,
    draw_cell_overlay,
    draw_markers,
    draw_polylines,
    save_frames,
)

logger = logging.getLogger(__name__)

SCENARIOS: dict[str, SwarmScenario] = {"river-small": RIVER_SMALL, "rivers-large": RIVERS_LARGE}
"""The scenarios by the name used in file names and titles."""
CONFIGS = ("seeded", "seeded-best", "seeded-no-yield", "hop", "hop-repel", "hop-tuned", "hop-best")
"""Every configuration that can be recorded."""
DEFAULT_CONFIGS = ("seeded", "hop", "hop-repel", "hop-tuned")
"""The configurations recorded when none is named: the four of the comparison."""
TRACES = Path("runs/70-obstacle-comparison/traces")
VIDEOS = Path("videos/70-obstacle-comparison")
SHOWN_PUSH = 0.05
"""Escape beacons that push less than this are not drawn."""


@dataclass(frozen=True)
class HeardMark:
    """A TREAT beacon a killer hears.

    Attributes:
        cell: Where it was sent from.
        value: Its hop value.
        pulling: Whether it pulls the killer: it is below the killer's own value.
    """

    cell: CellId
    value: float
    pulling: bool


@dataclass(frozen=True)
class HopDetails:
    """Why a hop killer walks where it does.

    Attributes:
        heard: Every fresh TREAT beacon it hears.
        pulls: Each pulling beacon's cell and the strength of its pull, as ``hop_pull`` weighs
            it; a negative strength pushes.
        push: The summed push of its escape beacons.
    """

    heard: tuple[HeardMark, ...]
    pulls: tuple[tuple[CellId, float], ...]
    push: Vector2


@dataclass(frozen=True)
class FieldDetails:
    """The seeded field a killer followed at its last decision.

    Attributes:
        values: The field's potential per cell, indexed ``[x, y]``; NaN where no target is
            reachable. Outside the search window a cell has its quadtree leaf's value.
        route: The centres of the nodes from the killer's node to its target: cells inside the
            window, leaves outside it, and the target cell last.
    """

    values: NDArray[np.float32]
    route: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class KillerFrame:
    """What one killer does at a frame.

    Attributes:
        goal: The behaviour that won its last decision, or ``None`` when idle.
        target: The cell it reported as its target at that decision.
        heading: The offset in cells it walked along at that decision, if it followed a heading.
        hop_value: Its own TREAT beacon value, if it has one.
        escapes: Each escape beacon it placed, with its current push.
        doses_left: The treatments left in its tank.
        doses_used: The treatments it has started since the run began.
        hop: Why it walks where it does, for a hop killer recorded with ``--explain``.
        field: The field it followed, for a seeded killer recorded with ``--explain``.
        treating: The cell it is treating, if any.
    """

    goal: str | None
    target: CellId | None
    heading: Vector2 | None
    hop_value: float | None
    escapes: tuple[tuple[CellId, float], ...]
    doses_left: int
    doses_used: int
    hop: HopDetails | None = None
    field: FieldDetails | None = None
    treating: CellId | None = None


@dataclass(frozen=True)
class Frame:
    """The changing part of the world at one time.

    Attributes:
        time: The simulated time.
        positions: Every robot's position in cells, between two cells while it moves.
        pests: Every infested cell, with its severity.
        killers: What each killer does.
        relay_values: Each relay's own TREAT beacon value, for relays that have one.
        relay_pests: How many infested cells each relay that keeps a map knows of, for relays
            that know of any.
    """

    time: Time
    positions: dict[RobotId, tuple[float, float]]
    pests: dict[CellId, float]
    killers: dict[RobotId, KillerFrame]
    relay_values: dict[RobotId, float]
    relay_pests: dict[RobotId, int] | None = None


@dataclass(frozen=True)
class Trace:
    """One recorded run: its fixed part once, and one ``Frame`` per decision interval.

    Attributes:
        scenario: The scenario's name, a key of ``SCENARIOS``.
        config: The configuration's name, one of ``CONFIGS``.
        seed: The run's seed.
        field: The field's geometry.
        cells: Every cell at the start, without pests; a frame's ``pests`` are laid over it.
        bodies: Every robot's body, by id.
        frames: The frames in time order.
        explain: Whether the frames hold why each killer walks where it does.
        hop_cap: The highest hop value a relay passes on, for the hop configurations.
    """

    scenario: str
    config: str
    seed: int
    field: FieldGeometry
    cells: dict[CellId, CellState]
    bodies: dict[RobotId, BodyType]
    frames: list[Frame]
    explain: bool = False
    hop_cap: float | None = None

    def killers(self) -> int:
        return sum(body is BodyType.KILLER for body in self.bodies.values())

    def file_stem(self) -> str:
        stem = f"{self.scenario}_{self.config}_seed{self.seed}"
        if self.explain:
            return f"explainer_{self.killers()}-killer_{stem}"
        return stem

    def snapshot(self, frame: Frame) -> WorldSnapshot:
        """The ground truth of ``frame``, for the renderer; robots stand on the cell they are in."""
        cells = dict(self.cells)
        for cell, severity in frame.pests.items():
            cells[cell] = replace(self.cells[cell], infestation=severity)
        robots: list[RobotPublicState] = []
        for robot_id, body in self.bodies.items():
            x, y = frame.positions[robot_id]
            robots.append(
                RobotPublicState(robot_id, (int(x), int(y)), body, 0.0, 1.0, 1.0, None, None)
            )
        return WorldSnapshot(frame.time, self.field, cells, robots)


@dataclass(frozen=True)
class Recording:
    """What to record: one configuration on one scenario.

    Attributes:
        scenario: The scenario's name, a key of ``SCENARIOS``.
        config: The configuration's name, one of ``CONFIGS``.
        seed: The run's seed.
        hours: How long the run lasts.
        traces: The directory the trace is written to.
        killers: How many killers, or ``None`` for the scenario's own.
        pests_per_hour: The pest rate, or ``None`` for the scenario's own.
        explain: Whether to keep why each killer walks where it does.
        treat_window_blocks: The seeded killers' TREAT field window, or ``None`` for the
            scenario's own.
    """

    scenario: str
    config: str
    seed: int
    hours: float
    traces: Path
    killers: int | None = None
    pests_per_hour: float | None = None
    explain: bool = False
    treat_window_blocks: int | None = None

    def into_scenario(self) -> SwarmScenario:
        base = SCENARIOS[self.scenario]
        if self.killers is not None:
            base = base.without_robots(BodyType.KILLER).add_robots(SEEDED_KILLER, self.killers)
        scenario = replace(
            pest_response_configurations(base, seeded_no_yield=True)[self.config],
            seed=self.seed,
            until=int(self.hours * TimeUnit.HOUR),
        )
        scenario = scenario.with_metrics(replace(scenario.metrics, period=TimeUnit.HOUR))
        if self.pests_per_hour is not None:
            scenario = scenario.with_pests(
                replace(scenario.pests, pests_per_hour=self.pests_per_hour)
            )
        if self.treat_window_blocks is not None and self.config.startswith("seeded"):
            killer = SeededKillerPreset(
                claims=self.config == "seeded", window_blocks=self.treat_window_blocks
            )
            others = scenario.without_robots(BodyType.KILLER)
            scenario = others.add_robots(killer, scenario.killers)
        return scenario


def record(recording: Recording) -> Path:
    """Make one run and write its trace; returns the trace's path."""
    scenario = recording.into_scenario()
    run = LiveRun.from_config(scenario.into_experiment(recording.config))
    world = run.context.world
    cells: dict[CellId, CellState] = {}
    for cell, state in world.cells.items():
        cells[cell] = replace(state, infestation=0.0)
    bodies: dict[RobotId, BodyType] = {}
    for robot_id, robot in world.robots.items():
        bodies[robot_id] = robot.body
    hop_cap: float | None = None
    beacons = scenario.relaying().beacons
    if beacons is not None:
        hop_cap = beacons.max_value
    trace = Trace(
        recording.scenario,
        recording.config,
        recording.seed,
        world.field,
        cells,
        bodies,
        [],
        recording.explain,
        hop_cap,
    )
    step = scenario.robots.decision_interval
    for now in range(0, scenario.until + 1, step):
        run.advance_to(now)
        trace.frames.append(_frame(run, now, recording.explain))
    path = recording.traces / f"{trace.file_stem()}.pkl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as file:
        pickle.dump(trace, file)
    logger.info("Recorded %s, %d frames", path, len(trace.frames))
    return path


def _frame(run: LiveRun, now: Time, explain: bool) -> Frame:
    """Everything the videos might show at ``now``, read from the world and the robots."""
    world = run.context.world
    world.advance_to(now)
    positions: dict[RobotId, tuple[float, float]] = {}
    for robot_id in world.robots:
        positions[robot_id] = world.continuous_position(robot_id, now)
    pests: dict[CellId, float] = {}
    for cell, state in world.cells.items():
        if state.infestation > 0.0:
            pests[cell] = state.infestation
    killers: dict[RobotId, KillerFrame] = {}
    relay_values: dict[RobotId, float] = {}
    relay_pests: dict[RobotId, int] = {}
    for robot_id, agent in run.agents().items():
        robot = world.robots[robot_id]
        hop_value: float | None = None
        escapes: list[tuple[CellId, float]] = []
        link = agent.beacon_link
        if link is not None:
            beacon = treat_beacon(link.memory, robot.position, now, link.config)
            if beacon is not None:
                hop_value = beacon.value
            for placed in link.memory.placed_beacons():
                escapes.append((placed.cell, obstacle_strength(placed, now)))
        if robot.body is not BodyType.KILLER:
            if hop_value is not None:
                relay_values[robot_id] = hop_value
            known = 0
            for pest in agent.memory.infestations():
                if pest.severity > 0:
                    known += 1
            if known > 0:
                relay_pests[robot_id] = known
            continue
        guidance = agent.guidance
        offset: Vector2 | None = None
        if isinstance(guidance, Heading):
            offset = guidance.offset
        hop: HopDetails | None = None
        field: FieldDetails | None = None
        if explain and link is not None:
            hop = _hop_details(link, robot.position, now)
        if explain and isinstance(guidance, SeededField):
            field = _field_details(guidance, robot.position, world.field)
        inspection = agent.inspect()
        killers[robot_id] = KillerFrame(
            goal=inspection.current_goal,
            target=inspection.current_target,
            heading=offset,
            hop_value=hop_value,
            escapes=tuple(escapes),
            doses_left=robot.doses_left,
            doses_used=robot.doses_used,
            treating=robot.treating_cell,
            hop=hop,
            field=field,
        )
    return Frame(now, positions, pests, killers, relay_values, relay_pests)


def _hop_details(link: BeaconLink, robot: CellId, now: Time) -> HopDetails:
    """The beacons a killer hears and the pull of each, weighed as ``hop_pull`` weighs them."""
    config = link.config
    memory = link.memory
    downhill = downhill_beacons(memory, robot, now, config)
    pulling: set[CellId] = set()
    for heard in downhill:
        pulling.add(heard.beacon.cell)
    marks: list[HeardMark] = []
    for heard in fresh_beacons(memory, BeaconKind.TREAT, now, config):
        cell = heard.beacon.cell
        marks.append(HeardMark(cell, heard.beacon.value, cell in pulling))
    killers: frozenset[RobotId] = frozenset()
    if config.killer_repulsion > 0:
        killers = memory.killers()
    pulls: list[tuple[CellId, float]] = []
    for heard in downhill:
        cell = heard.beacon.cell
        distance = hypot(cell[0] - robot[0], cell[1] - robot[1])
        strength = beacon_strength(heard, now, config) / max(distance, 1.0) ** config.distance_power
        if heard.sender in killers:
            strength *= -config.killer_repulsion
        pulls.append((cell, strength))
    return HopDetails(tuple(marks), tuple(pulls), obstacle_push(memory, robot, now, config))


def _field_details(seeded: SeededField, robot: CellId, geometry: FieldGeometry) -> FieldDetails:
    """The field's value at every cell, and the route through its nodes from ``robot``."""
    field = seeded.field
    values = np.full((geometry.width, geometry.height), np.nan, dtype=np.float32)
    for x in range(geometry.width):
        for y in range(geometry.height):
            value = field.value_at((x, y))
            if value != inf:
                values[x, y] = value
    route: list[tuple[float, float]] = []
    node = field.graph.node_at(robot)
    origin: CellId | None = None
    while node is not None:
        settled = field.settled.get(node)
        if settled is None:
            break
        route.append(_node_centre(node))
        origin = settled.origin
        node = settled.parent
    if origin is not None:
        route.append((origin[0] + 0.5, origin[1] + 0.5))
    return FieldDetails(values, tuple(route))


def _node_centre(node: CellId | Leaf) -> tuple[float, float]:
    if isinstance(node, Leaf):
        return node.centre
    return (node[0] + 0.5, node[1] + 0.5)


def load_trace(path: Path) -> Trace:
    with gzip.open(path, "rb") as file:
        trace = pickle.load(file)
    if not isinstance(trace, Trace):
        raise TypeError(f"{path} holds no trace")
    return trace


@dataclass(frozen=True)
class FrameVisuals:
    """What is drawn over the field at one frame.

    Attributes:
        arrows: Arrows between points of the field.
        markers: Dots at points of the field.
        lines: Lines through points of the field.
    """

    arrows: tuple[Arrow, ...]
    markers: tuple[Marker, ...]
    lines: tuple[Polyline, ...]


def frame_visuals(frame: Frame, explain: bool) -> FrameVisuals:
    """The arrows, markers and lines of one frame.

    Without ``explain``: a black arrow from each killer to where it walks, along its heading or
    to its target cell's centre, and its escape beacons. Every killer is labelled with the doses
    it has left, and a pest that several killers treat at once gets a gold square labelled ``xN``.
    With ``explain``, a hop killer gets the beacons it hears, an orange arrow for each beacon's
    pull and a red one for its heading, and a seeded killer its route through the field to its
    target.
    """
    arrows: list[Arrow] = []
    markers: list[Marker] = []
    lines: list[Polyline] = []
    treaters: dict[CellId, int] = {}
    for robot_id, killer in frame.killers.items():
        start = frame.positions[robot_id]
        markers.append(Marker(start, "red", size=0.0, label=f"{killer.doses_left}"))
        if killer.treating is not None:
            treaters[killer.treating] = treaters.get(killer.treating, 0) + 1
        for cell, push in killer.escapes:
            if push >= SHOWN_PUSH:
                markers.append(Marker((cell[0] + 0.5, cell[1] + 0.5), "mediumorchid", size=40.0))
        if explain and killer.hop is not None:
            visuals = _hop_visuals(start, killer.hop, killer.heading)
            arrows.extend(visuals.arrows)
            markers.extend(visuals.markers)
            continue
        if explain and killer.field is not None:
            lines.append(Polyline((start, *killer.field.route[1:]), "black", 2.0, "--"))
            target = killer.field.route[-1]
            markers.append(Marker(target, "white", size=130.0))
            continue
        end: tuple[float, float] | None = None
        if killer.heading is not None:
            end = (start[0] + killer.heading[0], start[1] + killer.heading[1])
        elif killer.target is not None:
            end = (killer.target[0] + 0.5, killer.target[1] + 0.5)
        if end is not None and end != start:
            arrows.append(Arrow(start, end, "black", width=1.8))
    for cell, count in treaters.items():
        if count > 1:
            x, y = cell
            ring = ((x - 0.4, y - 0.4), (x + 1.4, y - 0.4), (x + 1.4, y + 1.4), (x - 0.4, y + 1.4))
            lines.append(Polyline((*ring, ring[0]), "gold", 3.0))
            markers.append(Marker((x + 1.4, y + 1.4), "gold", 0.0, f"x{count}"))
    return FrameVisuals(tuple(arrows), tuple(markers), tuple(lines))


def _hop_visuals(
    start: tuple[float, float], hop: HopDetails, heading: Vector2 | None
) -> FrameVisuals:
    """The beacons a hop killer hears, each beacon's pull and its heading.

    The heading reaches as far as the strongest beacon, so each pull is drawn at the length of
    its strength relative to that beacon's: the strongest one's arrow reaches its beacon.
    """
    markers: list[Marker] = []
    for mark in hop.heard:
        color = "khaki"
        if mark.pulling:
            color = "darkorange"
        position = (mark.cell[0] + 0.5, mark.cell[1] + 0.5)
        markers.append(Marker(position, color, size=45.0, label=f"{mark.value:.1f}"))
    arrows: list[Arrow] = []
    strongest = 0.0
    for _, strength in hop.pulls:
        strongest = max(strongest, strength)
    if heading is None or strongest == 0:
        return FrameVisuals((), tuple(markers), ())
    reach = hypot(heading[0], heading[1])
    for cell, strength in hop.pulls:
        dx = cell[0] + 0.5 - start[0]
        dy = cell[1] + 0.5 - start[1]
        distance = hypot(dx, dy)
        if distance == 0:
            continue
        length = reach * strength / strongest
        tip = (start[0] + dx / distance * length, start[1] + dy / distance * length)
        arrows.append(Arrow(start, tip, "darkorange", width=1.5))
    push = hypot(hop.push[0], hop.push[1])
    if push > 0:
        length = reach * push / strongest
        tip = (start[0] + hop.push[0] / push * length, start[1] + hop.push[1] / push * length)
        arrows.append(Arrow(start, tip, "mediumorchid", width=2.0))
    end = (start[0] + heading[0], start[1] + heading[1])
    arrows.append(Arrow(start, end, "red", width=3.0))
    return FrameVisuals(tuple(arrows), tuple(markers), ())


def relay_knowledge(frame: Frame, hop_cap: float | None) -> list[Marker]:
    """A blue dot on every relay that knows of a pest, darker the more it knows.

    A hop relay is darker the lower its hop value, from its full colour at 0 to its palest at
    ``hop_cap``. A relay of the seeded configuration is darker the larger the share of the
    current pests it knows of.
    """
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    blues = colormaps["Blues"]
    shades: dict[RobotId, float] = {}
    for robot_id, value in frame.relay_values.items():
        cap = hop_cap or 1.0
        shades[robot_id] = 1.0 - min(value / cap, 1.0)
    for robot_id, known in (frame.relay_pests or {}).items():
        shades[robot_id] = min(known / max(len(frame.pests), 1), 1.0)
    markers: list[Marker] = []
    for robot_id, shade in shades.items():
        color = to_hex(blues(0.3 + 0.5 * shade))
        markers.append(Marker(frame.positions[robot_id], color, size=40.0))
    return markers


class TracePlayback:
    """Draws a trace frame by frame: the snapshot, what is drawn over it, and the panel.

    Attributes:
        trace: The run played.
        treated: How many pests had been cleared by each frame's time; only treatment clears one.
    """

    def __init__(self, trace: Trace) -> None:
        self.trace: Trace = trace
        self._frames: dict[Time, Frame] = {}
        self.treated: dict[Time, int] = {}
        cleared = 0
        before: set[CellId] = set()
        for frame in trace.frames:
            self._frames[frame.time] = frame
            now = set(frame.pests)
            cleared += len(before - now)
            before = now
            self.treated[frame.time] = cleared

    def times(self) -> list[Time]:
        return [frame.time for frame in self.trace.frames]

    def draw(self, renderer: MatplotlibRenderer, panel: SidePanel, time: Time) -> None:
        """Draw the frame of ``time``: robots between cells, the overlay, the visuals, the panel."""
        snapshot = self.snapshot(time)
        renderer.render(snapshot, self.positions(snapshot))
        axes = renderer.axes
        colormap = "Reds"
        if self.shows_field():
            colormap = "viridis_r"
        draw_cell_overlay(axes, snapshot, self.overlay(snapshot), colormap)
        visuals = self.visuals(snapshot)
        draw_polylines(axes, visuals.lines)
        draw_markers(axes, visuals.markers)
        draw_arrows(axes, visuals.arrows)
        axes.set_title(f"{self.title(snapshot)}\n{axes.get_title()}")
        panel.show(self.panel(snapshot))

    def snapshot(self, time: Time) -> WorldSnapshot:
        return self.trace.snapshot(self._frames[time])

    def shows_field(self) -> bool:
        """Whether the overlay is a seeded killer's field rather than the pests."""
        return self.trace.explain and self.trace.config == "seeded"

    def overlay(self, snapshot: WorldSnapshot) -> NDArray[np.float64]:
        """The pests, or with ``shows_field`` the first killer's field, from 0 to 1 per cell."""
        if self.shows_field():
            return self._field_overlay(snapshot)
        return self._pest_overlay(snapshot)

    def _pest_overlay(self, snapshot: WorldSnapshot) -> NDArray[np.float64]:
        """Every pest cell, from 0.6 at a new pest to 1 at a full one; other cells undrawn.

        A new pest barely tints its cell in the renderer, so this marks it clearly.
        """
        field = self.trace.field
        values = np.full((field.width, field.height), np.nan)
        for cell, severity in self._frames[snapshot.time].pests.items():
            values[cell] = 0.6 + 0.4 * severity
        return values

    def _field_overlay(self, snapshot: WorldSnapshot) -> NDArray[np.float64]:
        """The first killer's field, scaled to 0 at its lowest and 1 at its highest value."""
        field = self.trace.field
        for killer in self._frames[snapshot.time].killers.values():
            if killer.field is None:
                continue
            values = killer.field.values.astype(np.float64)
            low = float(np.nanmin(values))
            high = float(np.nanmax(values))
            return (values - low) / max(high - low, 1e-9)
        return np.full((field.width, field.height), np.nan)

    def positions(self, snapshot: WorldSnapshot) -> dict[RobotId, tuple[float, float]]:
        return self._frames[snapshot.time].positions

    def visuals(self, snapshot: WorldSnapshot) -> FrameVisuals:
        """The frame's visuals; explained, with what the relays know, under everything else.

        Over a field overlay, the pests are red dots on top, so the target's ring shows around
        its pest.
        """
        frame = self._frames[snapshot.time]
        visuals = frame_visuals(frame, self.trace.explain)
        if not self.trace.explain:
            return visuals
        markers = relay_knowledge(frame, self.trace.hop_cap)
        markers.extend(visuals.markers)
        if self.shows_field():
            for cell in frame.pests:
                markers.append(Marker((cell[0] + 0.5, cell[1] + 0.5), "red", size=50.0))
        return replace(visuals, markers=tuple(markers))

    def title(self, snapshot: WorldSnapshot) -> str:
        trace = self.trace
        title = f"{trace.scenario} | {trace.config} | seed {trace.seed} | {_clock(snapshot.time)}"
        if trace.explain:
            return f"explainer | {trace.killers()} killer | {title}"
        return title

    def panel(self, snapshot: WorldSnapshot) -> list[str]:
        """The pests and what the killers do, and a legend."""
        frame = self._frames[snapshot.time]
        treating = 0
        empty = 0
        for killer in frame.killers.values():
            if killer.doses_left <= 0:
                empty += 1
            elif killer.goal is not None and killer.goal.startswith("treat"):
                treating += 1
        idle = len(frame.killers) - treating - empty
        doses = 0
        treaters: dict[CellId, int] = {}
        for killer in frame.killers.values():
            doses += killer.doses_used
            if killer.treating is not None:
                treaters[killer.treating] = treaters.get(killer.treating, 0) + 1
        shared = 0
        for count in treaters.values():
            if count > 1:
                shared += 1
        informed = len(frame.relay_values) + len(frame.relay_pests or {})
        relays = len(self.trace.bodies) - len(frame.killers)
        lines = [
            f"time            {_clock(snapshot.time)}",
            f"pests now       {len(frame.pests)}",
            f"pests treated   {self.treated[snapshot.time]}",
            "",
            f"killers         {len(frame.killers)}",
            f"  treating      {treating}",
            f"  idle          {idle}",
            f"  empty         {empty}",
            f"doses used      {doses}",
            f"pests with 2+   {shared}",
            "  killers on it",
            "",
        ]
        if self.trace.explain:
            lines.extend([f"relays informed {informed}/{relays}", ""])
        lines.extend(self._legend())
        return lines

    def _legend(self) -> list[str]:
        """What the drawn elements mean."""
        legend = ["big red dots    killers", "grey dots       relays"]
        if not self.trace.explain:
            legend.extend(
                [
                    "red cells       pests",
                    "black arrows    where a killer walks",
                    "numbers         doses a killer has left",
                    "gold squares    a pest treated by several",
                    "                killers at once",
                ]
            )
        elif self.shows_field():
            legend.extend(
                [
                    "red dots        pests",
                    "blue dots       relays that know of pests,",
                    "                darker: more of them",
                    "colours         killer's TREAT field,",
                    "                yellow lowest, near pests;",
                    "                squares are quadtree leaves",
                    "dashed line     its route to the target",
                    "white ring      its target",
                ]
            )
        else:
            legend.extend(
                [
                    "red cells       pests",
                    "blue dots       relays with a hop value,",
                    "                darker: lower, nearer a pest",
                    "orange dots     beacons pulling the killer",
                    "pale dots       beacons it hears, not pulling",
                    "numbers         their hop values",
                    "orange arrows   pull of each beacon",
                    "red arrow       heading: summed pull",
                    "purple arrow    push of escape beacons",
                ]
            )
        legend.append("purple dots     escape beacons")
        return legend


def render(path: Path, videos: Path, fps: int, every: int, png: bool) -> Path:
    """Render one trace as a video, or as PNG frames with ``png``; returns where it was written."""
    import matplotlib

    matplotlib.use("Agg")
    playback = TracePlayback(load_trace(path))
    renderer = MatplotlibRenderer(
        (12.0, 8.0),
        show_targets=False,
        body_styles={
            BodyType.KILLER: RobotStyle("red", size=160.0, label="killer"),
            BodyType.SCOUT: RobotStyle("slategray", size=40.0, alpha=0.6, label="relay"),
        },
    )
    panel = SidePanel(renderer)
    times = playback.times()[::every]

    def draw_frame(index: int) -> None:
        playback.draw(renderer, panel, times[index])

    out = videos / f"{playback.trace.file_stem()}.mp4"
    if png:
        out = videos / playback.trace.file_stem()
    save_frames(renderer.figure, draw_frame, len(times), out, fps=fps)
    logger.info("Rendered %s", out)
    return out


def _clock(time: float) -> str:
    """A time as hours, minutes and seconds."""
    seconds = int(time // TimeUnit.SECOND)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _render_job(job: tuple[Path, Path, int, int, bool]) -> Path:
    return render(*job)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    record_parser = commands.add_parser("record", help="run and write traces")
    record_parser.add_argument("--scenario", nargs="+", choices=list(SCENARIOS), default=None)
    record_parser.add_argument("--config", nargs="+", choices=CONFIGS, default=None)
    record_parser.add_argument("--seed", type=int, default=0)
    record_parser.add_argument("--hours", type=float, default=3.0)
    record_parser.add_argument("--traces", type=Path, default=TRACES)
    record_parser.add_argument("--killers", type=int, help="override the scenario's killers")
    record_parser.add_argument("--pests-per-hour", type=float, help="override its pest rate")
    record_parser.add_argument(
        "--explain", action="store_true", help="keep why each killer walks where it does"
    )
    record_parser.add_argument(
        "--treat-window-blocks", type=int, help="override the seeded field's window"
    )
    render_parser = commands.add_parser("render", help="turn traces into videos")
    render_parser.add_argument("paths", type=Path, nargs="+", help="traces or their directories")
    render_parser.add_argument("--videos", type=Path, default=VIDEOS)
    render_parser.add_argument("--fps", type=int, default=10)
    render_parser.add_argument("--every", type=int, default=1, help="render every n-th frame")
    render_parser.add_argument("--png", action="store_true", help="PNG frames, not a video")
    parser.add_argument("--jobs", type=int, default=8, help="processes")
    args = parser.parse_args()

    if args.command == "record":
        scenarios = args.scenario or list(SCENARIOS)
        configs = args.config or list(DEFAULT_CONFIGS)
        recordings: list[Recording] = []
        # The large field first, the seeded configuration first: they take longest.
        for scenario, config in itertools.product(reversed(scenarios), configs):
            recordings.append(
                Recording(
                    scenario,
                    config,
                    args.seed,
                    args.hours,
                    args.traces,
                    args.killers,
                    args.pests_per_hour,
                    args.explain,
                    args.treat_window_blocks,
                )
            )
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(record, recordings))
        return
    traces: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            traces.extend(sorted(path.glob("*.pkl.gz")))
        else:
            traces.append(path)
    renders: list[tuple[Path, Path, int, int, bool]] = []
    for trace in traces:
        renders.append((trace, args.videos, args.fps, args.every, args.png))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(_render_job, renders))


if __name__ == "__main__":
    main()
