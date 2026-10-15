"""A runnable pest experiment: scouts explore, killers explore and treat, pests keep appearing.

Every robot is a real ``ComposedAgent`` with a ``UtilityArbiter``. Scouts carry one behaviour,
``FollowSeederField`` on the EXPLORE field (``ExploreSeeder``), the mock explore behaviour that
always bids 0.5. Killers carry ``TreatInfestation`` on the TREAT field (``TreatSeeder``) and, with
``killers_explore``, the same explore behaviour, so their arbiter weighs exploring against
treating. Without it a killer waits until it hears of a pest.

The simulation is the real discrete-event loop. ``WorldTick`` spawns pests on random free cells
at ``pests_per_hour``, and grows and spreads them as ``PestDynamicsConfig`` says. Robots share
their memory and status only through the real channel, within ``radio_range_m``, so a killer
learns of a pest when a robot that knows of it comes in range. From ground truth, a
``CoverageTracker`` measures the swarm's visits and an ``InfestationTracker`` its response to the
pests.

Robots start with the field's terrain map, since they only sense the cell they stand on and
would never see an obstacle they cannot enter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from swarm_field.contracts import (
    BodyType,
    CellId,
    FieldGeometry,
    RobotId,
    TerrainObservation,
    Time,
    TimeUnit,
    WorldSnapshot,
)
from swarm_field.experiments.explore import FollowSeederField, river_and_rocks, terrain_map
from swarm_field.simulation import (
    CoverageSample,
    CoverageTracker,
    InfestationSample,
    InfestationTracker,
    PestDynamicsConfig,
    RobotDecision,
    RobotState,
    SimulationContext,
    SimulationLoop,
    WorldState,
    WorldTick,
)
from swarm_field.swarm import (
    AgentMemory,
    ArbiterConfig,
    Behavior,
    ComposedAgent,
    ExploreConfig,
    ExploreSeeder,
    FieldProvider,
    FieldSeeder,
    Navigator,
    TreatInfestation,
    TreatSeeder,
    UtilityArbiter,
)


@dataclass(frozen=True)
class PestScenario:
    """The field, the swarm, the pests and the communication of one pest run.

    Attributes:
        size: The field's side in cells.
        obstacles: The blocked cells; by default the river with one bridge and two rocks of
            ``river_and_rocks``.
        scouts: How many scouts start, side by side from cell ``(0, 0)`` along the bottom row.
        killers: How many killers start, along the bottom row after the scouts.
        killers_explore: Whether killers also carry the explore behaviour.
        cell_size: The side of a cell in metres.
        speed_mps: The robots' speed; a straight move takes ``cell_size / speed_mps`` seconds.
        decision_interval: The time between two decisions of a robot.
        radio_range_m: How far a robot's broadcasts reach.
        pests_per_hour: How many pests spawn per hour, on average, on the free healthy cells.
        pest_period: The time between two pest updates (``WorldTick``).
        growth_rate: How much a pest's severity grows per pest update.
        spread_probability: The chance per pest update that a pest of severity 1 infests one
            neighbour; 0 keeps every pest where it spawned.
        spawn_severity: The severity of a new pest.
        explore: The parameters of every robot's EXPLORE field.
        arbiter: The overrides and hysteresis of every robot's ``UtilityArbiter``.
        sample_period: The time between two coverage and pest samples; shorter than any move.
        seed: Seeds the pests and when each robot decides first.
    """

    size: int = 32
    obstacles: tuple[CellId, ...] | None = None
    scouts: int = 5
    killers: int = 3
    killers_explore: bool = True
    cell_size: float = 10.0
    speed_mps_drone: float = 1.5
    speed_mps_ground: float = 0.1
    decision_interval: Time = TimeUnit.SECOND * 15
    radio_range_m: float = 300.0
    pests_per_hour: float = 15.0
    pest_period: Time = TimeUnit.MINUTE
    growth_rate: float = 0.05
    spread_probability: float = 0.0
    spawn_severity: float = 0.1
    explore: ExploreConfig = field(default_factory=ExploreConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    sample_period: Time = TimeUnit.SECOND
    seed: int = 0

    def field_geometry(self) -> FieldGeometry:
        return FieldGeometry(
            width=self.size,
            height=self.size,
            cell_size=self.cell_size,
            row_sp=0.0,
            in_row_sp=0.0,
            headland=0.0,
        )

    def blocked_cells(self) -> tuple[CellId, ...]:
        """The obstacles, or ``river_and_rocks`` when none are given."""
        if self.obstacles is None:
            return river_and_rocks(self.size)
        return self.obstacles

    def bodies(self) -> list[BodyType]:
        """The body of every robot, in the order of their ids: scouts first, then killers."""
        return [BodyType.SCOUT] * self.scouts + [BodyType.KILLER] * self.killers

    def body_speed(self, body: BodyType) -> float:
        return self.speed_mps_drone if body == BodyType.SCOUT else self.speed_mps_ground

    def starts(self) -> list[CellId]:
        """The robots' starting cells, side by side from ``(0, 0)``, skipping obstacles."""
        blocked = set(self.blocked_cells())
        free = [(x, 0) for x in range(self.size) if (x, 0) not in blocked]
        robots = self.scouts + self.killers
        if len(free) < robots:
            raise ValueError(f"the bottom row has room for {len(free)} robots, not {robots}")
        return free[:robots]

    def pest_dynamics(self) -> PestDynamicsConfig:
        """``pests_per_hour`` as the chance per free cell and pest update that a pest spawns."""
        free_cells = self.size * self.size - len(set(self.blocked_cells()))
        updates_per_hour = TimeUnit.HOUR / self.pest_period
        return PestDynamicsConfig(
            period=self.pest_period,
            spawn_probability=self.pests_per_hour / updates_per_hour / free_cells,
            growth_rate=self.growth_rate,
            spread_probability=self.spread_probability,
            spawn_severity=self.spawn_severity,
        )


@dataclass
class PestRun:
    """One assembled pest run, advanced in time by the caller.

    Attributes:
        scenario: What was assembled.
        context: The simulation's state; its world is the ground truth.
        agents: The robots' brains, by id.
        tracker: The coverage measures, sampled from the world.
        pests: The pest measures, sampled from the world.
        loop: The event loop that advances the run.
    """

    scenario: PestScenario
    context: SimulationContext
    agents: dict[RobotId, ComposedAgent]
    tracker: CoverageTracker
    pests: InfestationTracker
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

    def infested_cells(self) -> list[CellId]:
        """The cells infested in ground truth."""
        cells: list[CellId] = []
        for cell, state in self.context.world.cells.items():
            if state.infestation > 0:
                cells.append(cell)
        return cells


def build_pest_run(scenario: PestScenario) -> PestRun:
    """Assemble the world, the robots' brains, their decisions, the pests and the samples."""
    geometry = scenario.field_geometry()
    world = WorldState(geometry)
    for cell in scenario.blocked_cells():
        world.cell(cell).obstacle = True
    terrain = terrain_map(world)

    agents: dict[RobotId, ComposedAgent] = {}
    rng = Random(scenario.seed)
    context = SimulationContext(now=0, rng=rng, agents={}, world=world)
    for index, (body, start) in enumerate(zip(scenario.bodies(), scenario.starts(), strict=True)):
        robot_id = index + 1
        world.add_robot(
            RobotState(
                robot_id=robot_id,
                position=start,
                body=body,
                speed_mps=scenario.body_speed(body),
                radio_range_m=scenario.radio_range_m,
            )
        )
        agent = _build_agent(robot_id, body, geometry, terrain, scenario)
        agents[robot_id] = agent
        context.agents[robot_id] = agent
        first = rng.randrange(scenario.decision_interval)
        context.schedule(RobotDecision(first, robot_id, scenario.decision_interval))

    dynamics = scenario.pest_dynamics()
    context.schedule(WorldTick(dynamics.period, dynamics))
    tracker = CoverageTracker(world, fresh_age=scenario.explore.revisit_age)
    context.schedule(CoverageSample(0, scenario.sample_period, tracker))
    pests = InfestationTracker()
    context.schedule(InfestationSample(0, scenario.sample_period, pests))
    return PestRun(scenario, context, agents, tracker, pests)


def _build_agent(
    robot_id: RobotId,
    body: BodyType,
    geometry: FieldGeometry,
    terrain: list[TerrainObservation],
    scenario: PestScenario,
) -> ComposedAgent:
    """A scout explores; a killer treats and, with ``killers_explore``, also explores.

    The robot starts with ``terrain`` in its memory.
    """
    memory = AgentMemory(robot_id, geometry)
    memory.merge(terrain)
    explore_seeder = ExploreSeeder(scenario.explore)
    seeders: list[FieldSeeder] = []
    behaviors: list[Behavior] = []
    if body is BodyType.KILLER:
        seeders.append(TreatSeeder())
        behaviors.append(TreatInfestation())
    if body is not BodyType.KILLER or scenario.killers_explore:
        seeders.append(explore_seeder)
        behaviors.append(FollowSeederField(explore_seeder.name))
    fields = FieldProvider(memory, flies=False, seeders=seeders)
    return ComposedAgent(
        robot_id,
        behaviors,
        memory,
        fields,
        Navigator(fields.quadtree),
        UtilityArbiter(scenario.arbiter),
    )
