"""Interactive swarm-field console (C.1) on a small hand-built world.

Three robots wander a 24 by 16 field while pests spread. Unlike ``scenario_viewer.py``, which
drives a real scenario preset, this world is built here, so it exercises the ``Command`` port
on its own. The console itself (time controls, injecting, inspecting, recording and replaying)
is ``swarm_field.console``; see the README for every flag.

- Click an empty cell to inject an infestation; click a robot to inspect it.
- space / p pause, s steps one event, + / - change the speed, 0 resets it to 1x.

Run in a window, until it is closed:
    uv run python examples/interactive_demo.py

Record a session, then replay it in a window or into a video:
    uv run python examples/interactive_demo.py --record recordings/demo.json
    uv run python examples/interactive_demo.py --replay recordings/demo.json
    uv run python examples/interactive_demo.py --replay recordings/demo.json --save demo.mp4
"""

from __future__ import annotations

import argparse
from collections.abc import Hashable
from random import Random

from swarm_field.console import (
    InteractiveConsole,
    add_session_arguments,
    format_time,
    save_session,
    show_window,
    validate_session_arguments,
    write_recording,
)
from swarm_field.contracts import (
    BodyType,
    CellId,
    CommandLog,
    FieldGeometry,
    RobotId,
    SwarmAgent,
    Time,
    TimeUnit,
)
from swarm_field.simulation import (
    CommandReplay,
    PestDynamicsConfig,
    RobotDecision,
    RobotState,
    SimulationContext,
    WorldState,
    WorldTick,
)
from swarm_field.swarm import (
    ActionCandidate,
    AgentMemory,
    ComposedAgent,
    DecisionContext,
    FieldProvider,
    FollowField,
    MemoryView,
    Navigator,
)
from swarm_field.swarm.field import Seed
from swarm_field.visualization import MatplotlibRenderer

FIELD_WIDTH = 24
FIELD_HEIGHT = 16
CELL_SIZE_M = 2.0
WORLD_TICK_PERIOD = 3_000
DEFAULT_SEED = 7
DEFAULT_SPEED = 1.0
SAVED_SESSION_HOURS = 2 * TimeUnit.MINUTE / TimeUnit.HOUR
"""Length of a new session saved without a window, which has no window to close."""
SPEED_BY_BODY = {
    BodyType.SCOUT: 2.0,
    BodyType.KILLER: 1.2,
    BodyType.BATTERY: 0.8,
}


class _WanderSeeder:
    """Stand-in for the EXPLORE seeder (#41): a random walk over believed ground.

    It names one random neighbouring cell as the wander field's only target, and asks for a new
    one at every decision point by making the field key the decision time. The navigator then
    rebuilds the field per tick and the robot steps towards it. The target is clamped into the
    field, so it is never the unreachable outside cell a robot standing at the edge would pick.

    A real exploration algorithm would seed unvisited or stale cells instead of a random
    neighbour, and would keep them long enough to be worth walking to.
    """

    name: str = "wander"

    def __init__(self, field: FieldGeometry, rng: Random) -> None:
        self._field: FieldGeometry = field
        self._rng: Random = rng

    def field_key(self, memory: MemoryView, now: Time) -> Hashable:
        return now

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        dx, dy = self._rng.choice(((1, 0), (0, 1), (-1, 0), (0, -1)))
        x = min(max(robot[0] + dx, 0), self._field.width - 1)
        y = min(max(robot[1] + dy, 0), self._field.height - 1)
        return (Seed((x, y), 0.0),)


class WanderBehavior:
    """A minimal exploration behavior: bid to follow the wander field.

    The behaviour never moves the robot itself; it names the seeder whose field to follow and
    the navigator takes the step. Standing on the field's target needs no handling here: the
    navigator returns ``Idle`` for that case.
    """

    name: str = "wander"

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        candidate = ActionCandidate(
            action=FollowField(_WanderSeeder.name),
            score=1.0,
            source=self.name,
            target=None,
        )
        return (candidate,)


def build_context(seed: int) -> SimulationContext:
    """Build a seeded field with an initial pest patch, obstacles and one agent per robot."""
    field = FieldGeometry(
        width=FIELD_WIDTH,
        height=FIELD_HEIGHT,
        cell_size=CELL_SIZE_M,
        row_sp=3.0,
        in_row_sp=1.0,
        headland=2.0,
        connectivity=4,
    )
    bodies = [BodyType.SCOUT, BodyType.KILLER, BodyType.BATTERY]
    robots = [
        RobotState(
            robot_id=robot_id,
            position=start,
            body=body,
            speed_mps=SPEED_BY_BODY[body],
        )
        for robot_id, (body, start) in enumerate(zip(bodies, ((2, 2), (3, 3), (2, 4)), strict=True))
    ]
    world = WorldState(field, robots=robots)

    for cell, severity in {
        (14, 6): 0.6,
        (15, 7): 0.5,
        (18, 10): 0.4,
    }.items():
        world.cell(cell).infestation = severity

    for cell in ((20, 5), (20, 6), (20, 7), (21, 5), (21, 6), (21, 7)):
        world.cell(cell).obstacle = True

    rng = Random(seed)
    agents: dict[RobotId, SwarmAgent] = {}
    for robot_id in range(len(robots)):
        memory = AgentMemory(robot_id, field)
        fields = FieldProvider(
            memory,
            flies=False,
            # Its own random source, so a robot's walk does not depend on how many draws its
            # peers happened to make first.
            seeders=[_WanderSeeder(field, Random(seed + robot_id))],
        )
        agents[robot_id] = ComposedAgent(
            robot_id=robot_id,
            behaviors=[WanderBehavior()],
            memory=memory,
            fields=fields,
            navigator=Navigator(fields.quadtree),
        )
    ctx = SimulationContext(
        now=0,
        rng=rng,
        agents=agents,
        world=world,
    )
    for robot_id in range(len(robots)):
        ctx.schedule(RobotDecision(time=0, robot=robot_id))
    ctx.schedule(
        WorldTick(
            time=0,
            config=PestDynamicsConfig(
                period=WORLD_TICK_PERIOD,
                spawn_probability=0.008,
                growth_rate=0.03,
                spread_probability=0.28,
                spawn_severity=0.10,
            ),
        )
    )
    return ctx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_session_arguments(parser, default_speed=DEFAULT_SPEED, default_hours=None)
    args = parser.parse_args()
    validate_session_arguments(parser, args, default_hours=None)
    return args


def main() -> None:
    args = parse_args()
    log = CommandLog(seed=DEFAULT_SEED, until=0, commands=())
    if args.seed is not None:
        log = CommandLog(seed=args.seed, until=0, commands=())
    horizon: Time | None = None
    if args.replay is not None:
        log = CommandLog.from_json(args.replay.read_text(encoding="utf-8"))
        horizon = log.until
    elif args.hours is not None:
        horizon = int(args.hours * TimeUnit.HOUR)
    elif args.save is not None:
        horizon = int(SAVED_SESSION_HOURS * TimeUnit.HOUR)

    console = InteractiveConsole(
        CommandReplay(build_context(log.seed), log),
        MatplotlibRenderer(),
        label=f"interactive/{log.seed}",
        horizon=horizon,
        replaying=args.replay is not None,
        speed=args.speed,
    )
    if args.save is None:
        show_window(console)
    else:
        frames = save_session(console, args.save)
        print(f"Wrote {frames} frames up to {format_time(console.time)} to {args.save}")
    if args.record is not None:
        write_recording(console, args.record)
        print(f"Recorded {len(console.recorded)} commands up to {format_time(console.time)}")


if __name__ == "__main__":
    main()
