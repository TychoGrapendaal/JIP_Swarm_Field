"""Animated demonstration of Regroup bringing two robots back into radio range.

A killer knows of one pest far across the field; a scout has nothing to do and stays put. The
killer walks out of radio range to treat the pest, so both robots fall silent. After the silence
timeout each follows the field of its RegroupSeeder to where it last knew the other to be, until
they hear each other again. Every change of goal and of radio contact is logged.

Run from the repository root with:
    uv run python examples/regroup_demo.py
    uv run python examples/regroup_demo.py --headless   # log the timeline only, no window
"""

import argparse
import logging
from math import dist
from random import Random

from swarm_field.contracts import (
    BodyType,
    CellId,
    FieldGeometry,
    InfestationObservation,
    RobotId,
    SwarmAgent,
    Time,
    TimeUnit,
)
from swarm_field.simulation import (
    RobotDecision,
    RobotState,
    SimulationContext,
    SimulationLoop,
    WorldState,
)
from swarm_field.swarm import (
    AgentMemory,
    Behavior,
    ComposedAgent,
    FieldProvider,
    FieldSeeder,
    Navigator,
    Regroup,
    RegroupSeeder,
    TreatInfestation,
    TreatSeeder,
    UtilityArbiter,
)

logger = logging.getLogger(__name__)

KILLER_ID = 1
SCOUT_ID = 2
KILLER_START = (5, 6)
SCOUT_START = (3, 6)
PEST = (26, 6)
RALLY_POINT = (15, 6)
PEST_SEVERITY = 0.01
"""Treating takes an hour per unit of severity, so this pest is cleared in about 36 s."""

RADIO_RANGE_M = 12.0  # 6 cells
SILENCE_TIMEOUT = 30 * TimeUnit.SECOND
DECISION_INTERVAL = 1 * TimeUnit.SECOND
RUN_UNTIL = 2 * TimeUnit.MINUTE
FRAME_MS = 250


def make_world() -> WorldState:
    field = FieldGeometry(
        width=30,
        height=12,
        cell_size=2.0,
        row_sp=2.0,
        in_row_sp=1.0,
        headland=5.0,
        connectivity=8,
    )
    world = WorldState(
        field,
        robots=[
            RobotState(KILLER_ID, KILLER_START, BodyType.KILLER, 2.0, radio_range_m=RADIO_RANGE_M),
            RobotState(SCOUT_ID, SCOUT_START, BodyType.SCOUT, 2.0, radio_range_m=RADIO_RANGE_M),
        ],
    )
    world.cell(PEST).infestation = PEST_SEVERITY
    return world


def make_agent(robot_id: RobotId, body: BodyType, field: FieldGeometry) -> ComposedAgent:
    """A brain that regroups; the killer also treats, and already knows of the pest."""
    memory = AgentMemory(robot_id, field)
    regroup_seeder = RegroupSeeder(rally_point=RALLY_POINT)
    seeders: list[FieldSeeder] = [regroup_seeder]
    behaviors: list[Behavior] = [Regroup(regroup_seeder.name, SILENCE_TIMEOUT, RALLY_POINT)]
    if body is BodyType.KILLER:
        memory.merge(
            [InfestationObservation(PEST, 1.0, 0, robot_id, severity=PEST_SEVERITY, pest=None)]
        )
        seeders.append(TreatSeeder())
        behaviors.append(TreatInfestation())
    fields = FieldProvider(memory, flies=body is BodyType.SCOUT, seeders=seeders)
    return ComposedAgent(
        robot_id=robot_id,
        behaviors=behaviors,
        memory=memory,
        fields=fields,
        navigator=Navigator(fields.quadtree),
        arbiter=UtilityArbiter(),
    )


def make_context() -> SimulationContext:
    world = make_world()
    agents: dict[RobotId, SwarmAgent] = {}
    for robot in world.robots.values():
        agents[robot.robot_id] = make_agent(robot.robot_id, robot.body, world.field)
    ctx = SimulationContext(now=0, rng=Random(0), agents=agents, world=world)
    for robot_id in agents:
        ctx.schedule(RobotDecision(time=0, robot=robot_id, decision_interval=DECISION_INTERVAL))
    return ctx


class TimelineLogger:
    """Logs each robot's change of goal, and each change of radio contact between the two."""

    def __init__(self) -> None:
        self._goals: dict[RobotId, str | None] = {}
        self._in_range: bool | None = None

    def log_changes(self, ctx: SimulationContext, time: Time) -> None:
        in_range = robots_in_radio_range(ctx.world, time)
        if in_range != self._in_range:
            self._in_range = in_range
            logger.info("t=%5.1fs  radio contact: %s", time / TimeUnit.SECOND, in_range)
        for robot_id, agent in ctx.agents.items():
            goal = agent.inspect().current_goal
            if robot_id not in self._goals or goal != self._goals[robot_id]:
                self._goals[robot_id] = goal
                logger.info(
                    "t=%5.1fs  robot %d at %s: goal -> %s",
                    time / TimeUnit.SECOND,
                    robot_id,
                    ctx.world.robots[robot_id].position,
                    goal,
                )


def robots_in_radio_range(world: WorldState, time: Time) -> bool:
    killer = world.continuous_position(KILLER_ID, time)
    scout = world.continuous_position(SCOUT_ID, time)
    return dist(killer, scout) * world.field.cell_size <= RADIO_RANGE_M


def goal_of(ctx: SimulationContext, robot_id: RobotId) -> str | None:
    return ctx.agents[robot_id].inspect().current_goal


def run_headless() -> None:
    ctx = make_context()
    timeline = TimelineLogger()
    loop = SimulationLoop()
    for time in range(0, RUN_UNTIL + 1, DECISION_INTERVAL):
        loop.run(ctx, time)
        timeline.log_changes(ctx, time)
    logger.info("pest left at %s: %.1f", PEST, ctx.world.cell(PEST).infestation)


def run_animated() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    from swarm_field.visualization import MatplotlibRenderer

    ctx = make_context()
    timeline = TimelineLogger()
    loop = SimulationLoop()
    renderer = MatplotlibRenderer(figsize=(12.0, 5.5), show_targets=False)
    range_cells = RADIO_RANGE_M / ctx.world.field.cell_size

    def cell_centre(position: tuple[float, float] | CellId) -> tuple[float, float]:
        return (position[0] + 0.5, position[1] + 0.5)

    def update(frame: int) -> None:
        time = frame * FRAME_MS
        loop.run(ctx, time)
        ctx.world.advance_to(time)
        timeline.log_changes(ctx, time)

        renderer.render(ctx.world.snapshot(time))
        for robot_id in ctx.agents:
            centre = cell_centre(ctx.world.continuous_position(robot_id, time))
            renderer.axes.add_patch(Circle(centre, range_cells, fill=False, linestyle="--"))
        renderer.axes.plot(*cell_centre(RALLY_POINT), marker="x", color="black")
        renderer.axes.set_title(
            f"t={time / TimeUnit.SECOND:.0f}s | "
            f"radio contact: {robots_in_radio_range(ctx.world, time)} | "
            f"killer: {goal_of(ctx, KILLER_ID)} | scout: {goal_of(ctx, SCOUT_ID)}"
        )

    animation = FuncAnimation(
        renderer.figure, update, frames=RUN_UNTIL // FRAME_MS + 1, interval=40, repeat=False
    )
    _ = animation
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="log the timeline, no window")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.headless:
        run_headless()
    else:
        run_animated()


if __name__ == "__main__":
    main()
