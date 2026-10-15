"""A runnable exploration experiment: a swarm with one objective on an open or obstructed field.

Every robot is a real ``ComposedAgent`` with one behaviour, ``FollowSeederField``, which follows
the field of one seeder: the EXPLORE seeder, or ``RandomWaypointSeeder`` as a baseline. With a
single behaviour, the highest bid always wins, so no arbiter is needed. The simulation is the
real discrete-event loop; a ``CoverageTracker`` measures the swarm from ground truth.

Robots start with the field's terrain map, since they only sense the cell they stand on and
would never see an obstacle they cannot enter. They broadcast their status (cell and target)
over the real channel. Belief sharing between robots is not built yet (#34), so
``share_observations`` stands in for it: every infestation fact a robot senses reaches every
other robot at its next decision, as with perfect communication.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from random import Random

from swarm_field.contracts import (
    AgentInspection,
    BodyType,
    CellId,
    Decision,
    FieldGeometry,
    InfestationObservation,
    Observation,
    Perception,
    RobotId,
    Terrain,
    TerrainObservation,
    Time,
    TimeUnit,
    WorldSnapshot,
)
from swarm_field.simulation import (
    CoverageSample,
    CoverageSummary,
    CoverageTracker,
    RobotDecision,
    RobotState,
    SimulationContext,
    SimulationLoop,
    WorldState,
)
from swarm_field.swarm import (
    ActionCandidate,
    AgentMemory,
    ComposedAgent,
    DecisionContext,
    FieldProvider,
    FieldSeeder,
    FollowField,
    MemoryView,
    Navigator,
    UtilityArbiter,
)
from swarm_field.swarm.field import Seed

SeederFactory = Callable[[RobotId], FieldSeeder]
"""Makes the seeder of one robot; each robot gets its own."""


@dataclass(frozen=True)
class ExploreScenario:
    """The field, the swarm and the communication of one exploration run.

    Attributes:
        width: Field width in cells.
        height: Field height in cells.
        robots: How many robots start, side by side from cell ``(0, 0)`` along the bottom row.
        obstacles: The blocked cells.
        cell_size: The side of a cell in metres.
        speed_mps: The robots' speed; a straight move takes ``cell_size / speed_mps`` seconds.
        decision_interval: The time between two decisions of a robot.
        radio_range_m: How far a robot's status broadcast reaches.
        share_observations: Whether every sensed infestation fact reaches every robot.
        sample_period: The time between two coverage samples; shorter than any move.
        fresh_age: A visit to a cell visited less than this before is redundant.
        measure_from: Idleness is averaged from this time on.
        seed: Draws when each robot decides first, within one decision interval, so runs with
            different seeds interleave the robots' decisions differently.
    """

    width: int = 32
    height: int = 32
    robots: int = 5
    obstacles: tuple[CellId, ...] = ()
    cell_size: float = 10.0
    speed_mps: float = 1.0
    decision_interval: Time = 2_500
    radio_range_m: float = float("inf")
    share_observations: bool = True
    sample_period: Time = TimeUnit.SECOND
    fresh_age: Time = 30 * TimeUnit.MINUTE
    measure_from: Time = 0
    seed: int = 0

    def field_geometry(self) -> FieldGeometry:
        return FieldGeometry(
            width=self.width,
            height=self.height,
            cell_size=self.cell_size,
            row_sp=0.0,
            in_row_sp=0.0,
            headland=0.0,
        )

    def starts(self) -> list[CellId]:
        """The robots' starting cells, side by side from ``(0, 0)``, skipping obstacles."""
        free = [(x, 0) for x in range(self.width) if (x, 0) not in self.obstacles]
        if len(free) < self.robots:
            raise ValueError(f"the bottom row has room for {len(free)} robots, not {self.robots}")
        return free[: self.robots]


class FollowSeederField:
    """Always bid a fixed score to follow one seeder's field.

    As the only behaviour of a robot it always wins, so its score does not matter.
    """

    name: str = "follow-field"

    def __init__(self, seeder: str) -> None:
        self._seeder: str = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        return (ActionCandidate(FollowField(self._seeder), 0.5, self.name, target=None),)


class RandomWaypointSeeder:
    """A baseline: one target cell drawn at random, and a new one once the robot stands on it."""

    name: str = "random-waypoint"
    window_blocks: int = 1

    def __init__(self, field: FieldGeometry, obstacles: tuple[CellId, ...], seed: int) -> None:
        blocked = set(obstacles)
        self._cells: list[CellId] = []
        for x in range(field.width):
            self._cells.extend((x, y) for y in range(field.height) if (x, y) not in blocked)
        self._rng: Random = Random(seed)
        self._target: CellId = self._rng.choice(self._cells)

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        if robot == self._target:
            self._target = self._rng.choice(self._cells)
        return self._target

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        return (Seed(self._target, 0.0),)


class SharedObservations:
    """Every infestation fact a robot senses, in the order sensed: perfect belief sharing.

    A stand-in for #34. Each robot reads the facts added since its last decision.
    """

    def __init__(self) -> None:
        self.facts: list[Observation] = []


class SharingAgent:
    """A ``ComposedAgent`` that also reads and feeds a ``SharedObservations`` pool."""

    def __init__(self, agent: ComposedAgent, pool: SharedObservations) -> None:
        self.agent: ComposedAgent = agent
        self._pool: SharedObservations = pool
        self._read: int = 0

    def decide(self, perception: Perception) -> Decision:
        self.agent.memory.merge(self._pool.facts[self._read :])
        self._read = len(self._pool.facts)
        decision = self.agent.decide(perception)
        for fact in perception.sensed:
            if isinstance(fact, InfestationObservation):
                self._pool.facts.append(fact)
        return decision

    def inspect(self) -> AgentInspection:
        return self.agent.inspect()


@dataclass
class ExploreRun:
    """One assembled run, advanced in time by the caller.

    Attributes:
        scenario: What was assembled.
        context: The simulation's state; its world is the ground truth.
        agents: The robots' brains, by id.
        tracker: The coverage measures, sampled from the world.
        loop: The event loop that advances the run.
    """

    scenario: ExploreScenario
    context: SimulationContext
    agents: dict[RobotId, ComposedAgent]
    tracker: CoverageTracker
    loop: SimulationLoop = field(default_factory=SimulationLoop)

    def advance_to(self, time: Time) -> None:
        """Process every event up to ``time``."""
        self.loop.run(self.context, until=time)

    def snapshot_at(self, time: Time) -> WorldSnapshot:
        """Advance to ``time`` and return the ground truth then, for rendering."""
        self.advance_to(time)
        return self.context.world.snapshot(time)

    def targets(self) -> dict[RobotId, CellId]:
        """The cell each robot is heading for, from its last decision."""
        found: dict[RobotId, CellId] = {}
        for robot_id, agent in self.agents.items():
            target = agent.inspect().current_target
            if target is not None:
                found[robot_id] = target
        return found

    def summary(self) -> CoverageSummary:
        return self.tracker.summary()


def build_explore_run(scenario: ExploreScenario, seeder_for: SeederFactory) -> ExploreRun:
    """Assemble the world, the robots' brains, their decision events and the coverage samples."""
    geometry = scenario.field_geometry()
    world = WorldState(geometry)
    for cell in scenario.obstacles:
        world.cell(cell).obstacle = True
    terrain_map = _terrain_map(world)

    agents: dict[RobotId, ComposedAgent] = {}
    pool = SharedObservations()
    rng = Random(scenario.seed)
    context = SimulationContext(now=0, rng=rng, agents={}, world=world)
    for index, start in enumerate(scenario.starts()):
        robot_id = index + 1
        world.add_robot(
            RobotState(
                robot_id=robot_id,
                position=start,
                body=BodyType.SCOUT,
                speed_mps=scenario.speed_mps,
                radio_range_m=scenario.radio_range_m,
            )
        )
        memory = AgentMemory(robot_id, geometry)
        memory.merge(terrain_map)
        seeder = seeder_for(robot_id)
        fields = FieldProvider(memory, flies=False, seeders=[seeder])
        agent = ComposedAgent(
            robot_id,
            [FollowSeederField(seeder.name)],
            memory,
            fields,
            Navigator(fields.quadtree),
            UtilityArbiter(),
        )
        agents[robot_id] = agent
        context.agents[robot_id] = agent
        if scenario.share_observations:
            context.agents[robot_id] = SharingAgent(agent, pool)
        first = rng.randrange(scenario.decision_interval)
        context.schedule(RobotDecision(first, robot_id, scenario.decision_interval))

    tracker = CoverageTracker(
        world, fresh_age=scenario.fresh_age, measure_from=scenario.measure_from
    )
    context.schedule(CoverageSample(0, scenario.sample_period, tracker))
    return ExploreRun(scenario, context, agents, tracker)


def river_and_rocks(size: int) -> tuple[CellId, ...]:
    """A river across the middle of the field with one bridge, and a few square rocks."""
    river_x = size // 2
    bridge_y = 3 * size // 4
    cells: list[CellId] = []
    for y in range(size):
        if abs(y - bridge_y) > 1:
            cells.extend([(river_x, y), (river_x + 1, y)])
    for corner_x, corner_y in ((size // 4, size // 3), (3 * size // 4, size // 5)):
        for dx in range(3):
            cells.extend((corner_x + dx, corner_y + dy) for dy in range(3))
    return tuple(cells)


def _terrain_map(world: WorldState) -> list[TerrainObservation]:
    """The terrain of every cell, as facts observed at time 0."""
    facts: list[TerrainObservation] = []
    for cell, state in world.cells.items():
        terrain = Terrain.FREE
        if state.obstacle:
            terrain = Terrain.OBSTACLE
        facts.append(
            TerrainObservation(cell=cell, confidence=1.0, observed_at=0, source=0, terrain=terrain)
        )
    return facts
