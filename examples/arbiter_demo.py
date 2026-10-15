"""The utility arbiter at work: a swarm explores a field and treats the pests it finds.

Every robot is a ``ComposedAgent`` with a ``UtilityArbiter``, run in the discrete-event loop and
drawn through the ``MatplotlibRenderer``. Drones (blue) only explore. Ground robots (red) explore
and treat, so their arbiter weighs a pest against the unexplored ground around them. On top of
the renderer's frame, the demo marks each robot's current objective (E = explore, T = treat),
the target it is committed to (x), and darkens the cells no robot has seen yet.

Ground robots treat with ``TreatInfestation`` and the ``TreatSeeder``. Exploring is a small
stand-in defined here until the EXPLORE behaviour (#41) exists. Both score from memory alone,
never from a field. Every robot starts with the wall in its memory, since a ground robot never
observes an obstacle. The robots have no radio range, so a ground robot mostly treats pests it
found itself.

Run:
    uv run python examples/arbiter_demo.py

Headless (write frames as PNGs instead of opening a window):
    uv run python examples/arbiter_demo.py --frames 150 --save frames
"""

from __future__ import annotations

import argparse
from collections.abc import Hashable, Sequence
from pathlib import Path
from random import Random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from swarm_field.contracts import (
    BodyType,
    CellId,
    FieldGeometry,
    Terrain,
    TerrainObservation,
    Time,
)
from swarm_field.simulation import (
    RobotDecision,
    RobotState,
    SimulationContext,
    SimulationLoop,
    WorldState,
)
from swarm_field.swarm import (
    ActionCandidate,
    AgentMemory,
    ArbiterChoice,
    Behavior,
    ComposedAgent,
    DecisionContext,
    FieldProvider,
    FollowField,
    MemoryView,
    Navigator,
    TreatInfestation,
    TreatSeeder,
    UtilityArbiter,
)
from swarm_field.swarm.field import Seed
from swarm_field.visualization import MatplotlibRenderer

FIELD_WIDTH = 30
FIELD_HEIGHT = 20
CELL_SIZE_M = 2.0
DECISION_INTERVAL_MS = 2_000
FRAME_TIME_MS = 2_000

PESTS: dict[CellId, float] = {
    (6, 15): 0.9,
    (7, 15): 0.6,
    (22, 4): 0.8,
    (23, 5): 0.5,
    (14, 9): 0.7,
    (26, 16): 0.9,
    (3, 4): 0.4,
}
WALL: tuple[CellId, ...] = tuple((12, y) for y in range(4, 16))
ROBOTS: tuple[tuple[BodyType, CellId], ...] = (
    (BodyType.SCOUT, (0, 0)),
    (BodyType.SCOUT, (29, 19)),
    (BodyType.KILLER, (0, 19)),
    (BodyType.KILLER, (29, 0)),
    (BodyType.KILLER, (15, 10)),
)
SPEED_BY_BODY: dict[BodyType, float] = {BodyType.SCOUT: 1.5, BodyType.KILLER: 0.8}
GOAL_LABELS: dict[str, str] = {"explore": "E", TreatInfestation.name: "T"}


class UnseenCells:
    """Stand-in EXPLORE seeder: every cell the robot has not observed, worth its trip only."""

    name: str = "explore"
    window_blocks: int = 1

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        return tuple(_unseen_cells(memory))

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        return tuple(Seed(cell, 0.0) for cell in _unseen_cells(memory))


class ExploreUnseen:
    """Stand-in EXPLORE behaviour: follow the ``UnseenCells`` field while any cell is unseen.

    It bids half the fraction of the field still unseen, so exploring matters less as the map
    fills in, and a known pest wins more often.
    """

    name: str = "explore"

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        unseen = len(_unseen_cells(ctx.memory))
        if unseen == 0:
            return ()
        score = 0.5 * unseen / (FIELD_WIDTH * FIELD_HEIGHT)
        return (ActionCandidate(FollowField(UnseenCells.name), score, self.name, None),)


class CountingArbiter(UtilityArbiter):
    """A ``UtilityArbiter`` that counts how often the robot abandons a commitment."""

    def __init__(self) -> None:
        super().__init__()
        self.switches: int = 0

    def choose(self, candidates: Sequence[ActionCandidate], now: Time) -> ArbiterChoice | None:
        choice = super().choose(candidates, now)
        if choice is not None and choice.switched:
            self.switches += 1
        return choice


def _unseen_cells(memory: MemoryView) -> list[CellId]:
    """The field cells the robot has no terrain fact for, in a fixed order."""
    cells: list[CellId] = []
    for y in range(FIELD_HEIGHT):
        for x in range(FIELD_WIDTH):
            if memory.terrain_at((x, y)) is None:
                cells.append((x, y))
    return cells


def _wall_facts(robot_id: int) -> list[TerrainObservation]:
    """The wall as terrain facts, for a robot's memory to start with."""
    facts: list[TerrainObservation] = []
    for cell in WALL:
        facts.append(
            TerrainObservation(
                cell=cell, confidence=1.0, observed_at=0, source=robot_id, terrain=Terrain.OBSTACLE
            )
        )
    return facts


def build_world() -> WorldState:
    """A cropped field with a wall across the middle, scattered pests, drones and ground robots."""
    field = FieldGeometry(
        width=FIELD_WIDTH,
        height=FIELD_HEIGHT,
        cell_size=CELL_SIZE_M,
        row_sp=0.75,
        in_row_sp=0.2,
        headland=4.0,
    )
    robots = [
        RobotState(robot_id=index, position=start, body=body, speed_mps=SPEED_BY_BODY[body])
        for index, (body, start) in enumerate(ROBOTS, start=1)
    ]
    world = WorldState(field, robots=robots)
    for y in range(FIELD_HEIGHT):
        for x in range(FIELD_WIDTH):
            world.cell((x, y)).growth_stage = 0.6
    for cell in WALL:
        world.cell(cell).obstacle = True
    for cell, severity in PESTS.items():
        world.cell(cell).infestation = severity
    return world


def build_agent(robot: RobotState, field: FieldGeometry) -> ComposedAgent:
    """A brain for one robot: drones explore, ground robots explore and treat."""
    memory = AgentMemory(robot.robot_id, field)
    memory.merge(_wall_facts(robot.robot_id))
    # Drones plan around obstacles too: the world does not model flying yet and refuses a move
    # into an obstacle cell, so a drone planning over the wall would stall against it.
    fields = FieldProvider(memory, flies=False, seeders=[TreatSeeder(), UnseenCells()])
    behaviors: list[Behavior] = [ExploreUnseen()]
    if robot.body is BodyType.KILLER:
        behaviors.insert(0, TreatInfestation())
    navigator = Navigator(fields.quadtree)
    return ComposedAgent(robot.robot_id, behaviors, memory, fields, navigator, CountingArbiter())


def build_simulation() -> SimulationContext:
    """The world, one brain per robot, and each robot's first decision event."""
    world = build_world()
    agents = {robot_id: build_agent(robot, world.field) for robot_id, robot in world.robots.items()}
    ctx = SimulationContext(now=0, rng=Random(7), agents=dict(agents), world=world)
    for robot_id in world.robots:
        ctx.schedule(
            RobotDecision(time=robot_id, robot=robot_id, decision_interval=DECISION_INTERVAL_MS)
        )
    return ctx


def draw(renderer: MatplotlibRenderer, ctx: SimulationContext, time: Time) -> None:
    """Render the world, then overlay what each robot's arbiter has chosen."""
    snapshot = ctx.world.snapshot(time)
    renderer.render(snapshot)
    axes = renderer.axes
    agents = _composed_agents(ctx)

    unseen = set(_cells_of_field())
    for agent in agents.values():
        unseen.difference_update(cell for cell in _cells_of_field() if _seen(agent, cell))
    for x, y in unseen:
        axes.add_patch(plt.Rectangle((x, y), 1, 1, color="black", alpha=0.25, linewidth=0))

    switches = 0
    for robot in snapshot.robots:
        agent = agents[robot.robot_id]
        goal = agent.inspect().current_goal
        x, y = robot.position[0] + 0.5, robot.position[1] + 0.5
        axes.annotate(
            GOAL_LABELS.get(goal or "", "-"),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )
        arbiter = agent.arbiter
        if isinstance(arbiter, CountingArbiter):
            switches += arbiter.switches
            commitment = arbiter.commitment
            if commitment is not None and commitment.target is not None:
                tx, ty = commitment.target
                axes.plot(tx + 0.5, ty + 0.5, marker="x", color="black", markersize=6)

    pests_left = sum(1 for cell in PESTS if ctx.world.cell(cell).infestation > 0)
    explored = 100 * (1 - len(unseen) / (FIELD_WIDTH * FIELD_HEIGHT))
    axes.set_title(
        f"t={time / 1000:.0f} s | pests left {pests_left}/{len(PESTS)} | "
        f"explored {explored:.0f}% | switches {switches}"
    )


def _composed_agents(ctx: SimulationContext) -> dict[int, ComposedAgent]:
    """The brains of the simulation, as the ``ComposedAgent``s this demo built."""
    agents: dict[int, ComposedAgent] = {}
    for robot_id, agent in ctx.agents.items():
        if isinstance(agent, ComposedAgent):
            agents[robot_id] = agent
    return agents


def _cells_of_field() -> list[CellId]:
    return [(x, y) for y in range(FIELD_HEIGHT) for x in range(FIELD_WIDTH)]


def _seen(agent: ComposedAgent, cell: CellId) -> bool:
    return agent.memory.terrain_at(cell) is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300, help="number of frames")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="directory to write PNG frames into instead of showing a window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = build_simulation()
    loop = SimulationLoop()
    renderer = MatplotlibRenderer(figsize=(11.0, 7.5))

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for index in range(args.frames):
            time = index * FRAME_TIME_MS
            loop.run(ctx, until=time)
            draw(renderer, ctx, time)
            renderer.figure.savefig(args.save / f"frame_{index:04d}.png")
        print(f"Wrote {args.frames} frames to {args.save}")
        return

    frame_time = {"value": 0}

    def update(_frame: int) -> None:
        frame_time["value"] += FRAME_TIME_MS
        loop.run(ctx, until=frame_time["value"])
        draw(renderer, ctx, frame_time["value"])

    animation = FuncAnimation(
        renderer.figure, update, frames=args.frames, interval=60, repeat=False
    )
    _ = animation
    plt.show()


if __name__ == "__main__":
    main()
