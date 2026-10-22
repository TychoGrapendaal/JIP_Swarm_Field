"""Watch one killer find pests by hop counts, with every robot's count drawn.

A slowed-down, annotated version of ``hop_demo.py`` for explaining the two hop methods. Scouts
explore and one killer explores until it hears of a pest. Grey lines join robots in radio
range. The number above every robot is its hop count for the pest in focus (the one the killer
follows, or else the newest one); a dash means it has no count.

- ``--navigation gradient`` (default): real hop-gradient navigation. The scout that finds a
  pest stays on it as the anchor (gold ring) and sends hop 0 every round; every robot keeps
  the newest round and passes on its count. The orange arrows point from every robot to its
  parent, the neighbour it heard its count from, so they form a tree down to the anchor. The
  killer drives to its parent (dashed line), hop by hop, and never learns where the pest is.
- ``--navigation hop``: hop-limited beacons. A beacon carries the pest's cell and a hop count;
  the counts only rank the pests, and the killer drives to the pest's cell (dashed line).
  Orange arrows show where each hop of the news happened.

Run from the repository root with:
    uv run python examples/hop_explainer.py
    uv run python examples/hop_explainer.py --navigation hop

``--view field`` colours the map by the potential field the killer follows, the cost to its
current target, and draws the path the field predicts instead of trails. ``--arrows`` adds
arrows for how the counts were passed from robot to robot.

Record a video instead, or print only the event log and the numbers:
    uv run python examples/hop_explainer.py --save videos/gradient_explainer.mp4
    uv run python examples/hop_explainer.py --events
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from math import dist, isfinite, isnan
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from swarm_field.contracts import BodyType, CellId, PestId, RobotId, TimeUnit, Treat
from swarm_field.experiments import HopPestScenario, LiveRun
from swarm_field.swarm import (
    ComposedAgent,
    GradientAgent,
    HopAgent,
    chosen_beacon,
    chosen_gradient,
)
from swarm_field.swarm.field import GridPotentialField
from swarm_field.swarm.gradient import parent_cell

SEEDER_OF_GOAL: dict[str, str] = {
    "follow-gradient": "gradient",
    "follow-beacon": "beacon",
    "follow-field": "explore",
    "explore": "explore",
}
"""The seeder whose field a killer follows, by the behaviour that won its last decision."""

KILLER_COLOURS = ["#e03131", "#f76707", "#d6336c", "#ae3ec9", "#fab005", "#0ca678"]
"""Each killer's own colour, for its body, trail and way, in order of robot id."""

HOP_COLOURS = ["#f5c400", "#f08c00", "#e8590c", "#c2255c", "#862e9c", "#5f3dc4", "#364fc7"]
"""Badge colours by hop count: 0 is gold, further hops get darker."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--navigation", choices=("gradient", "hop"), default="gradient")
    parser.add_argument("--size", type=int, default=32, help="the field's side in cells of 10 m")
    parser.add_argument(
        "--layout",
        choices=("edge", "centre"),
        default="edge",
        help="where robots start: along the bottom edge, or around the middle",
    )
    parser.add_argument("--scouts", type=int, default=6)
    parser.add_argument("--killers", type=int, default=1)
    parser.add_argument("--radio", type=float, default=70.0, help="radio range in metres")
    parser.add_argument("--pests-per-hour", type=float, default=6.0)
    parser.add_argument("--killer-speed", type=float, default=0.5, help="in m/s")
    parser.add_argument("--hours", type=float, default=1.5, help="simulated time")
    parser.add_argument("--frame-seconds", type=float, default=15.0, help="simulated per frame")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--start", type=float, default=0.0, help="simulated minutes to run before drawing"
    )
    parser.add_argument("--save", type=Path, help="write a .mp4 or .gif instead of a window")
    parser.add_argument("--events", action="store_true", help="print the events, draw nothing")
    parser.add_argument(
        "--view",
        choices=("clean", "field"),
        default="clean",
        help="clean: green field, the killer's last cells; field: the killer's potential field "
        "as colour and its predicted path",
    )
    parser.add_argument(
        "--arrows", action="store_true", help="also draw how counts were passed robot to robot"
    )
    parser.add_argument("--trail-cells", type=int, default=5, help="killer cells kept as trail")
    parser.add_argument(
        "--no-contact",
        action="store_true",
        help="let robots drift out of radio contact (by default each keeps a neighbour)",
    )
    return parser.parse_args()


def pest_name(pest: PestId) -> str:
    """A short name for a pest: P, its finder and the finder's number for it."""
    return f"P{pest.finder}.{pest.number}"


@dataclass(frozen=True)
class Known:
    """What one robot knows of one pest, in either method.

    Attributes:
        pest: The pest.
        hops: The robot's hop count.
        cleared: Whether the robot knows the pest is gone.
        cell: The pest's cell, which only a hop beacon carries.
        parent: The neighbour the count came from, which only the gradient keeps.
        found_at: When the pest was found, for picking the newest; 0 if unknown.
    """

    pest: PestId
    hops: int
    cleared: bool
    cell: CellId | None = None
    parent: RobotId | None = None
    found_at: int = 0


def knowledge_of(agent: ComposedAgent, now: int) -> list[Known]:
    """What a robot knows of every pest, for either kind of hop agent."""
    known: list[Known] = []
    if isinstance(agent, HopAgent):
        for beacon in agent.beacons.known(now):
            known.append(
                Known(beacon.pest, beacon.hops, beacon.cleared, beacon.cell, None, beacon.found_at)
            )
    if isinstance(agent, GradientAgent):
        for entry in agent.gradients.live(now):
            known.append(Known(entry.pest, entry.hops, False, None, entry.parent))
        for pest in agent.gradients.cleared(now):
            known.append(Known(pest, 0, True))
    return known


def followed_by(agent: ComposedAgent, now: int) -> Known | None:
    """The pest a killer follows, in either method."""
    if isinstance(agent, HopAgent):
        beacon = chosen_beacon(agent.beacons, agent.memory, agent.robot_id, now)
        if beacon is not None:
            return Known(beacon.pest, beacon.hops, False, beacon.cell)
    if isinstance(agent, GradientAgent):
        entry = chosen_gradient(agent.gradients, now)
        if entry is not None and entry.hops > 0:
            return Known(entry.pest, entry.hops, False, None, entry.parent)
    return None


@dataclass(frozen=True)
class Style:
    """How a frame is drawn.

    Attributes:
        view: ``"clean"`` for a green field with the killer's last cells as its trail, or
            ``"field"`` for the killer's potential field as colour and its predicted path.
        arrows: Whether to draw how the counts were passed from robot to robot.
        trail_cells: How many of the killer's last cells the clean view draws as its trail.
    """

    view: str = "clean"
    arrows: bool = False
    trail_cells: int = 5


@dataclass
class Frame:
    """What the swarm looks like and knows at one moment.

    Attributes:
        time: The moment.
        positions: Every robot's position in cell coordinates.
        bodies: Every robot's body.
        known: What every robot knows of every pest.
        chosen: The pest each killer follows, if any.
        heading: Where each following killer drives to: the pest's cell, or its parent's cell.
        anchors: The robots standing on a pest as its anchor, with the pest's cell.
        treating: The killers that are treating.
        pests: The infested cells in ground truth.
        focus: The pest whose hop counts are drawn.
        fields: The potential field each killer follows, as it last built it.
        paths: Each killer's route along its field, from its cell to the field's target.
    """

    time: int
    positions: dict[RobotId, tuple[float, float]]
    bodies: dict[RobotId, BodyType]
    known: dict[RobotId, list[Known]]
    chosen: dict[RobotId, Known]
    heading: dict[RobotId, CellId]
    anchors: dict[RobotId, CellId]
    treating: set[RobotId]
    pests: set[CellId]
    focus: PestId | None
    fields: dict[RobotId, GridPotentialField] = field(default_factory=dict)
    paths: dict[RobotId, list[CellId]] = field(default_factory=dict)


def predicted_path(potential: GridPotentialField, start: CellId) -> list[CellId]:
    """The cells a robot at ``start`` steps through when it keeps going downhill on the field.

    Ends at the field's target, joined by a straight step if the field stops guiding first.
    """
    path = [start]
    visited = {start}
    for _ in range(500):
        step = potential.next_cell(path[-1])
        if step is None or step in visited:
            break
        path.append(step)
        visited.add(step)
    target = potential.target_for(start)
    if target is not None and path[-1] != target:
        path.append(target)
    return path


@dataclass(frozen=True)
class Relay:
    """One hop of a pest's news in the hop method: who passed it to whom, when and where.

    Attributes:
        sender: The robot the news most likely came from: one in range, one hop nearer.
        receiver: The robot that heard it.
        time: When the receiver had it.
        hops: The receiver's hop count.
        start: The sender's position then.
        end: The receiver's position then.
    """

    sender: RobotId | None
    receiver: RobotId
    time: int
    hops: int
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass
class Tracker:
    """Advances the run frame by frame, and logs what happens to the news of each pest.

    Attributes:
        run: The run.
        radio_cells: The radio range in cells.
        events: The event log, oldest first.
        trails: Recent positions per robot.
        stats: Counters for the numbers in the panel and the summary.
        relays: Hop method only: per pest and robot, the hop that last improved its count.
        cells: The cells each robot has stood in, oldest first, each once per visit.
    """

    run: LiveRun
    radio_cells: float
    events: list[str] = field(default_factory=list)
    trails: dict[RobotId, list[tuple[float, float]]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    relays: dict[PestId, dict[RobotId, Relay]] = field(default_factory=dict)
    _seen: dict[RobotId, set[PestId]] = field(default_factory=dict)
    _cleared: set[PestId] = field(default_factory=set)
    _started: dict[PestId, int] = field(default_factory=dict)
    _chosen: dict[RobotId, tuple[PestId, RobotId | None] | None] = field(default_factory=dict)
    _anchors: set[RobotId] = field(default_factory=set)
    _treating: set[RobotId] = field(default_factory=set)
    _pests_before: set[CellId] = field(default_factory=set)
    _heard: list[tuple[int, int]] = field(default_factory=list)
    _best_hops: dict[tuple[RobotId, PestId], int] = field(default_factory=dict)
    cells: dict[RobotId, list[CellId]] = field(default_factory=dict)

    def step(self, time: int) -> Frame:
        """Advance to ``time``, log the events since the last frame, and return the frame."""
        self.run.advance_to(time)
        world = self.run.context.world
        positions = _spread_stacked(
            {rid: world.continuous_position(rid, time) for rid in world.robots}
        )
        bodies = {rid: robot.body for rid, robot in world.robots.items()}
        pests = {cell for cell, state in world.cells.items() if state.infestation > 0}
        self._count("appeared", len(pests - self._pests_before))
        self._pests_before = pests
        known: dict[RobotId, list[Known]] = {}
        chosen: dict[RobotId, Known] = {}
        heading: dict[RobotId, CellId] = {}
        anchors: dict[RobotId, CellId] = {}
        treating: set[RobotId] = set()
        fields: dict[RobotId, GridPotentialField] = {}
        paths: dict[RobotId, list[CellId]] = {}
        for rid, agent in self.run.agents().items():
            known[rid] = knowledge_of(agent, time)
            self._log_news(rid, bodies[rid], known[rid], world.robots[rid].position, time)
            if agent.inspect().current_goal == "anchor":
                anchors[rid] = world.robots[rid].position
            if bodies[rid] is BodyType.KILLER:
                followed = followed_by(agent, time)
                if followed is not None:
                    chosen[rid] = followed
                    target = self._heading_of(agent, followed)
                    if target is not None:
                        heading[rid] = target
                self._log_killer(rid, followed, bodies, world.robots[rid].current_action, time)
                if isinstance(world.robots[rid].current_action, Treat):
                    treating.add(rid)
                potential = _followed_field(agent)
                if potential is not None:
                    fields[rid] = potential
                    paths[rid] = predicted_path(potential, world.robots[rid].position)
            self.trails.setdefault(rid, []).append(positions[rid])
            visited = self.cells.setdefault(rid, [])
            if not visited or visited[-1] != world.robots[rid].position:
                visited.append(world.robots[rid].position)
        self._log_anchors(anchors, time)
        self._log_relays(known, positions, bodies, time)
        focus = _newest_live(known)
        for followed in chosen.values():
            focus = followed.pest
        return Frame(
            time,
            positions,
            bodies,
            known,
            chosen,
            heading,
            anchors,
            treating,
            pests,
            focus,
            fields,
            paths,
        )

    def _heading_of(self, agent: ComposedAgent, followed: Known) -> CellId | None:
        if followed.cell is not None:
            return followed.cell
        return parent_cell(agent.memory, followed.parent)

    def _count(self, name: str, amount: int = 1) -> None:
        self.stats[name] = self.stats.get(name, 0) + amount

    def _log_news(
        self, rid: RobotId, body: BodyType, known: list[Known], cell: CellId, time: int
    ) -> None:
        seen = self._seen.setdefault(rid, set())
        who = _who(rid, body)
        for item in known:
            if item.cleared and item.pest not in self._cleared:
                self._cleared.add(item.pest)
                self._count("cleared")
                self._log(time, f"{who} knows {pest_name(item.pest)} is gone: cleared")
            if item.pest in seen or item.cleared:
                continue
            seen.add(item.pest)
            if item.pest.finder == rid and item.hops == 0:
                self._started[item.pest] = time
                self._count("found")
                self._log(time, f"{who} senses a pest at {cell}: {pest_name(item.pest)}, hop 0")
            elif body is BodyType.KILLER:
                delay = time - self._started.get(item.pest, time)
                self._heard.append((item.hops, delay))
                self._log(
                    time,
                    f"{who} hears of {pest_name(item.pest)} at {_hops(item.hops)}, "
                    f"{delay / TimeUnit.MINUTE:.1f} min after it was sensed",
                )

    def _log_anchors(self, anchors: dict[RobotId, CellId], time: int) -> None:
        for rid, cell in anchors.items():
            if rid not in self._anchors:
                self._log(time, f"S{rid} stays on {cell} as the anchor, sending hop 0")
        for rid in self._anchors - set(anchors):
            self._log(time, f"S{rid} stops anchoring and explores again")
        self._anchors = set(anchors)

    def _log_relays(
        self,
        known: dict[RobotId, list[Known]],
        positions: dict[RobotId, tuple[float, float]],
        bodies: dict[RobotId, BodyType],
        time: int,
    ) -> None:
        """Hop method only: record every robot whose count of a pest dropped, and from whom.

        The simulation does not say which message carried a beacon, so the sender is taken to
        be the nearest robot that had the pest one hop nearer its finder. The gradient keeps
        its parents itself, so nothing needs recording for it.
        """
        before = dict(self._best_hops)
        for rid, items in known.items():
            for item in items:
                if item.cleared or item.cell is None:
                    continue
                key = (rid, item.pest)
                if key in self._best_hops and self._best_hops[key] <= item.hops:
                    continue
                self._best_hops[key] = item.hops
                sender = self._sender(rid, item, before, positions)
                start = positions[rid]
                if sender is not None:
                    start = positions[sender]
                relay = Relay(sender, rid, time, item.hops, start, positions[rid])
                self.relays.setdefault(item.pest, {})[rid] = relay
                if sender is not None and item.hops > 0:
                    self._log(
                        time,
                        f"{_who(sender, bodies[sender])} -> {_who(rid, bodies[rid])}: "
                        f"{pest_name(item.pest)} now {_hops(item.hops)}",
                    )

    def _sender(
        self,
        rid: RobotId,
        item: Known,
        before: dict[tuple[RobotId, PestId], int],
        positions: dict[RobotId, tuple[float, float]],
    ) -> RobotId | None:
        if item.hops == 0:
            return None
        nearest: RobotId | None = None
        nearest_distance = float("inf")
        for other, position in positions.items():
            if other == rid:
                continue
            hops = before.get((other, item.pest), self._best_hops.get((other, item.pest)))
            if hops != item.hops - 1:
                continue
            distance = dist(position, positions[rid])
            if distance < nearest_distance:
                nearest = other
                nearest_distance = distance
        return nearest

    def news_path(self, pest: PestId, robot: RobotId) -> list[Relay]:
        """Hop method: the hops that brought the news of ``pest`` to ``robot``, finder first."""
        path: list[Relay] = []
        relays = self.relays.get(pest, {})
        current: RobotId | None = robot
        while current is not None and current in relays and len(path) < 30:
            relay = relays[current]
            path.append(relay)
            current = relay.sender
        path.reverse()
        return path

    def _log_killer(
        self,
        rid: RobotId,
        followed: Known | None,
        bodies: dict[RobotId, BodyType],
        action: object,
        time: int,
    ) -> None:
        state: tuple[PestId, RobotId | None] | None = None
        if followed is not None:
            state = (followed.pest, followed.parent)
        if self._chosen.get(rid) != state:
            self._chosen[rid] = state
            if followed is None:
                self._log(time, f"K{rid} has no live count: explores")
            elif followed.cell is not None:
                self._log(
                    time,
                    f"K{rid} heads for {pest_name(followed.pest)} at {followed.cell} "
                    f"({_hops(followed.hops)})",
                )
            elif followed.parent is not None:
                parent = _who(followed.parent, bodies[followed.parent])
                self._log(
                    time,
                    f"K{rid} at {_hops(followed.hops)} from {pest_name(followed.pest)}: "
                    f"drives to its parent {parent}",
                )
        if isinstance(action, Treat) and rid not in self._treating:
            self._treating.add(rid)
            self._count("treated")
            self._count(f"treated by {rid}")
            self._log(time, f"K{rid} stands on the pest at {action.cell} and treats it")
        if not isinstance(action, Treat):
            self._treating.discard(rid)

    def _log(self, time: int, text: str) -> None:
        self.events.append(f"{_clock(time)}  {text}")

    def summary(self) -> list[str]:
        """The run's numbers, as lines."""
        lines = [
            f"pests appeared            {self.stats.get('appeared', 0)}",
            f"hop-0 reports (sensings)  {self.stats.get('found', 0)}",
            f"treatments by killers     {self.stats.get('treated', 0)}",
            f"pests known cleared       {self.stats.get('cleared', 0)}",
        ]
        if self._heard:
            hops = sum(h for h, _ in self._heard) / len(self._heard)
            delay = sum(d for _, d in self._heard) / len(self._heard) / TimeUnit.MINUTE
            lines.append(f"news reaching the killer  {len(self._heard)} pests")
            lines.append(f"  mean hops               {hops:.1f}")
            lines.append(f"  mean delay              {delay:.1f} min after sensing")
        return lines


def _spread_stacked(
    positions: dict[RobotId, tuple[float, float]],
) -> dict[RobotId, tuple[float, float]]:
    """The positions, with robots on the same spot moved apart a little so each one shows."""
    spread: dict[RobotId, tuple[float, float]] = {}
    taken: list[tuple[float, float]] = []
    for rid in sorted(positions):
        x, y = positions[rid]
        stacked = 0
        for other in taken:
            if dist(other, (x, y)) < 0.3:
                stacked += 1
        taken.append((x, y))
        spread[rid] = (x + 0.7 * stacked, y - 0.7 * stacked)
    return spread


def _followed_field(agent: ComposedAgent) -> GridPotentialField | None:
    """The potential field the robot followed at its last decision, if it followed one."""
    goal = agent.inspect().current_goal
    if goal is None or goal not in SEEDER_OF_GOAL:
        return None
    seeded = agent.fields.cached_field(SEEDER_OF_GOAL[goal])
    if seeded is None:
        return None
    return seeded.field


def _newest_live(known: dict[RobotId, list[Known]]) -> PestId | None:
    """The live pest found last that some robot knows of and none knows cleared.

    The gradient sends no time of finding, so its pests are ordered by their numbers.
    """
    cleared: set[PestId] = set()
    for items in known.values():
        cleared.update(item.pest for item in items if item.cleared)
    newest: Known | None = None
    for items in known.values():
        for item in items:
            if item.cleared or item.pest in cleared:
                continue
            if newest is None or _newer(item, newest):
                newest = item
    if newest is None:
        return None
    return newest.pest


def _newer(item: Known, than: Known) -> bool:
    if item.found_at != than.found_at:
        return item.found_at > than.found_at
    return item.hops < than.hops


def _who(rid: RobotId, body: BodyType) -> str:
    if body is BodyType.KILLER:
        return f"K{rid}"
    return f"S{rid}"


def _hops(count: int) -> str:
    if count == 1:
        return "1 hop"
    return f"{count} hops"


def _clock(time: float) -> str:
    seconds = int(time // TimeUnit.SECOND)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _badge(colour: str) -> dict[str, str]:
    return {"boxstyle": "circle,pad=0.25", "fc": colour, "ec": "black"}


def counts_for(frame: Frame, pest: PestId | None) -> dict[RobotId, Known]:
    """Each robot's live knowledge of ``pest``, for the robots that have a count."""
    found: dict[RobotId, Known] = {}
    if pest is None:
        return found
    for rid, items in frame.known.items():
        for item in items:
            if item.pest == pest and not item.cleared:
                found[rid] = item
    return found


def nearest_counts(frame: Frame) -> dict[RobotId, Known]:
    """Each robot's smallest live hop count over all pests: its hops to the nearest anchor."""
    cleared: set[PestId] = set()
    for items in frame.known.values():
        cleared.update(item.pest for item in items if item.cleared)
    found: dict[RobotId, Known] = {}
    for rid, items in frame.known.items():
        for item in items:
            if item.cleared or item.pest in cleared:
                continue
            best = found.get(rid)
            if best is None or item.hops < best.hops:
                found[rid] = item
    return found


def out_of_contact(frame: Frame, radio_cells: float) -> list[RobotId]:
    """The robots with no other robot within radio range, in ground truth."""
    alone: list[RobotId] = []
    for rid, position in frame.positions.items():
        linked = False
        for other, other_position in frame.positions.items():
            if other != rid and dist(position, other_position) <= radio_cells:
                linked = True
                break
        if not linked:
            alone.append(rid)
    return alone


def killer_colours(frame: Frame) -> dict[RobotId, str]:
    """Each killer's colour, by robot id."""
    colours: dict[RobotId, str] = {}
    killers = [rid for rid, body in sorted(frame.bodies.items()) if body is BodyType.KILLER]
    for index, rid in enumerate(killers):
        colours[rid] = KILLER_COLOURS[index % len(KILLER_COLOURS)]
    return colours


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    main: bool,
    dashed: bool,
    colour: str = "#ff922b",
) -> None:
    width = 1.2
    alpha = 0.6
    scale = 10
    if main:
        width = 2.8
        alpha = 1.0
        scale = 15
    style = "-"
    if dashed:
        style = "--"
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            color=colour,
            lw=width,
            alpha=alpha,
            linestyle=style,
            shrinkA=8,
            shrinkB=8,
        )
    )


def draw(
    field_ax: Axes,
    panel_ax: Axes,
    frame: Frame,
    tracker: Tracker,
    scenario: HopPestScenario,
    blocked: tuple[CellId, ...],
    style: Style,
    colour_ax: Axes | None = None,
) -> None:
    """Draw one frame: the field with links, counts and the killer's way, and the panel."""
    field_ax.clear()
    panel_ax.clear()
    gradient = scenario.navigation == "gradient"
    size = scenario.size
    field_ax.set_xlim(0, size)
    field_ax.set_ylim(0, size)
    field_ax.set_aspect("equal")
    field_ax.set_facecolor("#3d8b37")
    field_ax.set_xticks([])
    field_ax.set_yticks([])
    if style.view == "field":
        _draw_potential(field_ax, colour_ax, frame, size)
    for x, y in blocked:
        field_ax.add_patch(Rectangle((x, y), 1, 1, color="#555555", lw=0))
    for x, y in frame.pests:
        pest_style = {"color": "#e03131", "lw": 0}
        if style.view == "field":
            pest_style = {"fill": False, "ec": "#e03131", "lw": 2.5}
        field_ax.add_patch(Rectangle((x, y), 1, 1, **pest_style))

    # Hop method: the pests the swarm knows of by beacon, with the cell the beacon carries.
    labelled: set[PestId] = set()
    for items in frame.known.values():
        for item in items:
            if item.cell is None or item.pest in labelled:
                continue
            labelled.add(item.pest)
            x, y = item.cell
            if item.cleared:
                field_ax.plot(x + 0.5, y + 0.5, marker="x", color="#dddddd", ms=9, mew=2)
                continue
            field_ax.add_patch(
                Rectangle((x - 0.15, y - 0.15), 1.3, 1.3, fill=False, ec="#ffe066", lw=2)
            )
            field_ax.text(
                x + 0.5, y + 1.4, pest_name(item.pest), color="#ffe066", ha="center", fontsize=8
            )

    # Radio links between every pair of robots in range.
    robots = sorted(frame.positions)
    for i, a in enumerate(robots):
        for b in robots[i + 1 :]:
            pa, pb = frame.positions[a], frame.positions[b]
            if dist(pa, pb) <= tracker.radio_cells:
                field_ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#d0d0d0", lw=1, alpha=0.6)

    colours = killer_colours(frame)
    counts = nearest_counts(frame)
    for rid in frame.anchors:
        x, y = frame.positions[rid]
        field_ax.add_patch(Circle((x, y), 1.1, fill=False, ec="#ffd43b", lw=2.5))
        field_ax.text(x, y - 1.6, "anchor", color="#ffd43b", ha="center", fontsize=8)
    if style.arrows and gradient:
        # Each killer's chain down its pest's gradient, in the killer's colour.
        for killer, followed in frame.chosen.items():
            pest_counts = counts_for(frame, followed.pest)
            current: RobotId | None = killer
            seen: set[RobotId] = set()
            while current is not None and current in pest_counts and current not in seen:
                seen.add(current)
                parent = pest_counts[current].parent
                if parent is None or parent not in frame.positions:
                    break
                start = frame.positions[current]
                end = frame.positions[parent]
                out_of_range = dist(start, end) > tracker.radio_cells
                _arrow(field_ax, start, end, True, out_of_range, colours[killer])
                current = parent
    elif style.arrows:
        # The hop method: where each hop of the news to each killer happened.
        for killer, followed in frame.chosen.items():
            for relay in tracker.news_path(followed.pest, killer):
                if relay.sender is not None:
                    _arrow(field_ax, relay.start, relay.end, True, False, colours[killer])

    # The killer's way: its last cells and where it drives to, or its predicted path.
    if style.view == "field":
        for rid, path in frame.paths.items():
            xs = [cell[0] + 0.5 for cell in path]
            ys = [cell[1] + 0.5 for cell in path]
            field_ax.plot(xs, ys, color="white", lw=4.5, solid_capstyle="round")
            field_ax.plot(xs, ys, color=colours[rid], lw=2.5, marker="o", ms=3)
            field_ax.plot(xs[-1], ys[-1], marker="*", color=colours[rid], ms=17, mec="white")
    else:
        for rid, visited in tracker.cells.items():
            if frame.bodies[rid] is not BodyType.KILLER:
                continue
            recent = visited[-style.trail_cells :]
            xs = [cell[0] + 0.5 for cell in recent] + [frame.positions[rid][0]]
            ys = [cell[1] + 0.5 for cell in recent] + [frame.positions[rid][1]]
            field_ax.plot(xs, ys, color=colours[rid], lw=3, solid_capstyle="round", alpha=0.8)
        for rid, cell in frame.heading.items():
            start = frame.positions[rid]
            end = (cell[0] + 0.5, cell[1] + 0.5)
            field_ax.plot([start[0], end[0]], [start[1], end[1]], "--", color="white", lw=3)
            field_ax.plot([start[0], end[0]], [start[1], end[1]], "--", color=colours[rid], lw=1.8)
    for rid, body in frame.bodies.items():
        if body is BodyType.KILLER:
            field_ax.add_patch(
                Circle(
                    frame.positions[rid],
                    tracker.radio_cells,
                    fill=False,
                    ec=colours[rid],
                    ls=":",
                    lw=1,
                    alpha=0.7,
                )
            )

    # Robots with no other robot in radio range: a red dashed ring.
    for rid in out_of_contact(frame, tracker.radio_cells):
        field_ax.add_patch(
            Circle(frame.positions[rid], 1.3, fill=False, ec="#ff0000", lw=2, ls="--")
        )

    # The robots, with their count for the pest in focus. On a big field, or with many robots,
    # robots are drawn larger, only killers are labelled, and robots without a count get no
    # badge, so that the hop counts stay readable.
    scale = max(1.0, size / 32)
    crowded = len(robots) > 16
    for rid in robots:
        x, y = frame.positions[rid]
        killer = frame.bodies[rid] is BodyType.KILLER
        radius = 0.55 * scale
        colour = "#1c7ed6"
        if killer:
            radius = 0.8 * scale
            colour = colours[rid]
        field_ax.add_patch(Circle((x, y), radius, color=colour, ec="white", lw=1))
        if killer or not crowded:
            field_ax.text(
                x,
                y,
                _who(rid, frame.bodies[rid]),
                color="white",
                ha="center",
                va="center",
                fontsize=6.5,
                weight="bold",
            )
        if not counts:
            continue
        badge = "-"
        badge_colour = "#868e96"
        item = counts.get(rid)
        if item is not None:
            badge = str(item.hops)
            badge_colour = HOP_COLOURS[min(item.hops, len(HOP_COLOURS) - 1)]
        elif crowded:
            continue
        field_ax.text(
            x + 0.9 * scale,
            y + 0.9 * scale,
            badge,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            color="black",
            bbox=_badge(badge_colour),
        )
    title = "no hop counts yet"
    if counts and gradient:
        title = "number = each robot's hops to the nearest anchor"
    elif counts:
        title = "number = fewest hops the news of any pest took to reach each robot"
    field_ax.set_title(title, fontsize=10)
    panel_ax.axis("off")
    lines = panel_lines(frame, tracker, scenario, gradient, style)
    panel_ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8)


def panel_lines(
    frame: Frame, tracker: Tracker, scenario: HopPestScenario, gradient: bool, style: Style
) -> list[str]:
    """The panel: settings, numbers, every robot's counts, the killer's chain and the log."""
    method = "hop-limited beacons (pest cell in the message)"
    if gradient:
        method = "hop gradient (no position in any message)"
    lines = [
        f"time {_clock(frame.time)}    {method}",
        f"{_count_of(scenario.scouts, 'scout')}, {_count_of(scenario.killers, 'killer')}, "
        f"radio {scenario.radio_range_m:.0f} m ({tracker.radio_cells:.0f} cells), "
        f"keep contact: {_on_off(scenario.keep_connected)}",
        "",
        "NUMBERS",
        f"pests in the field now     {len(frame.pests)}",
        f"pests appeared so far      {tracker.stats.get('appeared', 0)}",
        f"hop-0 reports (sensings)   {tracker.stats.get('found', 0)}",
        f"treatments by killers      {tracker.stats.get('treated', 0)}",
        f"pests known cleared        {tracker.stats.get('cleared', 0)}",
        f"robots out of contact now  {len(out_of_contact(frame, tracker.radio_cells))}",
        f"anchors (one per pest)     {len(frame.anchors)}",
        "",
        "KILLERS",
    ]
    for rid, body in frame.bodies.items():
        if body is not BodyType.KILLER:
            continue
        doing = "explores (no live count)"
        followed = frame.chosen.get(rid)
        if rid in frame.treating:
            doing = "treats the pest it stands on"
        elif followed is not None:
            doing = f"-> {pest_name(followed.pest)}: {_chain_text(frame, tracker, rid, followed)}"
        treated = tracker.stats.get(f"treated by {rid}", 0)
        lines.append(f"K{rid} [{treated} treated] {doing}")
    shown_robots = sorted(frame.positions)
    title = "WHAT EACH ROBOT KNOWS  (pest:hops<parent, x = cleared)"
    if len(shown_robots) > 16:
        shown_robots = [rid for rid in shown_robots if frame.bodies[rid] is BodyType.KILLER]
        title = "WHAT EACH KILLER KNOWS  (pest:hops<parent, x = cleared)"
    lines.extend(["", title])
    for rid in shown_robots:
        shown: list[str] = []
        for item in frame.known.get(rid, [])[:4]:
            if item.cleared:
                shown.append(f"x{pest_name(item.pest)}")
            elif item.parent is not None:
                shown.append(f"{pest_name(item.pest)}:{item.hops}<{item.parent}")
            else:
                shown.append(f"{pest_name(item.pest)}:{item.hops}")
        lines.append(f"{_who(rid, frame.bodies[rid]):>3}  {' '.join(shown) or '-'}")
    lines.extend(["", "HOW TO READ"])
    lines.extend(_legend(gradient, style))
    lines.extend(["", "WHAT HAPPENED"])
    lines.extend(tracker.events[-10:])
    return lines


def _on_off(flag: bool) -> str:
    if flag:
        return "on"
    return "off"


def _count_of(number: int, thing: str) -> str:
    if number == 1:
        return f"1 {thing}"
    return f"{number} {thing}s"


def _legend(gradient: bool, style: Style) -> list[str]:
    """The panel's key to the drawing."""
    lines = ["badge  = hops to the nearest anchor (0 = an anchor, on a pest)"]
    if not gradient:
        lines = ["badge  = fewest hops any pest's news took (0 = sensed it)"]
    lines.append("big circles = killers, each in its own colour; blue = scouts")
    if style.arrows:
        lines.append("arrows = each killer's chain of hops, in its colour")
    if style.view == "field":
        lines.append("colour = cost to the nearest killer target (bright = near);")
        lines.append("         every killer steps downhill on its own field")
        lines.append("line + star = the path a killer's field predicts, its target")
        lines.append("red box = a pest in ground truth")
    elif gradient:
        lines.append(f"short line = a killer's last {style.trail_cells} cells")
        lines.append("dashed = where a killer drives: its parent's cell")
    else:
        lines.append(f"short line = a killer's last {style.trail_cells} cells")
        lines.append("dashed = the pest's cell from the beacon")
    lines.append("grey   = robots in radio range now")
    lines.append("red dashed ring = a robot out of contact with every other")
    return lines


def _draw_potential(field_ax: Axes, colour_ax: Axes | None, frame: Frame, size: int) -> None:
    """The killer's potential field as colour, with a colour bar; plain green without one."""
    if colour_ax is not None:
        colour_ax.clear()
        colour_ax.set_axis_off()
    if not frame.fields:
        return
    values = np.full((size, size), np.nan)
    for potential in frame.fields.values():
        for x in range(size):
            for y in range(size):
                value = potential.value_at((x, y))
                if not isfinite(value):
                    continue
                if isnan(values[y, x]) or value < values[y, x]:
                    values[y, x] = value
    if np.all(np.isnan(values)):
        return
    image = field_ax.imshow(
        np.ma.masked_invalid(values),
        origin="lower",
        extent=(0, size, 0, size),
        cmap="viridis_r",
        interpolation="nearest",
        zorder=0,
    )
    if colour_ax is not None:
        colour_ax.set_axis_on()
        bar = colour_ax.figure.colorbar(image, cax=colour_ax, orientation="horizontal")
        bar.set_label(
            "potential: cost to the nearest killer's current target, in cells", fontsize=8
        )
        bar.ax.tick_params(labelsize=7)


def _chain_text(frame: Frame, tracker: Tracker, killer: RobotId, followed: Known) -> str:
    """A killer's chain to the pest it follows: robots and their hops, ending at the pest."""
    names: list[str] = []
    if followed.cell is None:
        counts = counts_for(frame, followed.pest)
        current: RobotId | None = killer
        while current is not None and current in counts and len(names) < 12:
            names.append(f"{_who(current, frame.bodies[current])}({counts[current].hops})")
            current = counts[current].parent
        names.append("pest")
        return "-".join(names)
    for relay in tracker.news_path(followed.pest, killer):
        names.append(_who(relay.receiver, frame.bodies[relay.receiver]))
    return f"{_hops(followed.hops)}, news came " + "-".join(names)


def main() -> None:
    args = parse_args()
    logging.disable(logging.WARNING)
    scenario = HopPestScenario(
        navigation=args.navigation,
        size=args.size,
        layout=args.layout,
        scouts=args.scouts,
        killers=args.killers,
        radio_range_m=args.radio,
        pests_per_hour=args.pests_per_hour,
        speed_mps_ground=args.killer_speed,
        until=int(args.hours * TimeUnit.HOUR),
        seed=args.seed,
        keep_connected=not args.no_contact,
    )
    tracker = Tracker(
        LiveRun.from_config(scenario.into_experiment()), args.radio / scenario.cell_size
    )
    step = int(args.frame_seconds * TimeUnit.SECOND)
    times = list(range(0, scenario.until + 1, step))
    start = int(args.start * TimeUnit.MINUTE)
    if args.events:
        for time in times:
            tracker.step(time)
        print("\n".join(tracker.events))
        print()
        print("\n".join(tracker.summary()))
        return

    for time in times:
        if time < start:
            tracker.step(time)
    times = [time for time in times if time >= start]
    blocked = scenario.blocked_cells()
    figure, (field_ax, panel_ax) = plt.subplots(
        1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    style = Style(view=args.view, arrows=args.arrows, trail_cells=args.trail_cells)
    colour_ax: Axes | None = None
    bottom = 0.03
    if style.view == "field":
        bottom = 0.1
        colour_ax = figure.add_axes((0.06, 0.045, 0.43, 0.018))
    figure.subplots_adjust(left=0.02, right=0.99, top=0.94, bottom=bottom, wspace=0.03)

    def update(time: int) -> list[object]:
        frame = tracker.step(time)
        draw(field_ax, panel_ax, frame, tracker, scenario, blocked, style, colour_ax)
        return []

    animation = FuncAnimation(figure, update, frames=times, interval=1000 / args.fps, repeat=False)
    if args.save is None:
        plt.show()
        return
    args.save.parent.mkdir(parents=True, exist_ok=True)
    if args.save.suffix == ".gif":
        animation.save(str(args.save), writer=PillowWriter(fps=args.fps))
    else:
        animation.save(str(args.save), writer=FFMpegWriter(fps=args.fps, bitrate=2400))
    print("\n".join(tracker.summary()))


if __name__ == "__main__":
    main()
