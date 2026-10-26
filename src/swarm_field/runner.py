"""The composition root: the one place that knows every concrete class (architecture §I.2).

Everything else depends on ports. This module depends on the implementations: it resolves
named brain parts from registries, builds one brain per robot, seeds the run's first events
and drives ``SimulationLoop``. Because it is the only such place, comparing algorithms is a
change to a ``BrainSpec`` rather than to any component.

A run is reproducible by construction. One ``Random``, seeded from ``ScenarioConfig.seed``, is
created here and threaded into the context, and every part that needs randomness takes its
draw from it at assembly time in a fixed order. The context's ``rng`` is then left to the pest
dynamics, which draw from it in event order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from random import Random
from typing import Any, TypeVar

from swarm_field.contracts import (
    BodyType,
    CellId,
    FieldGeometry,
    MetricsSample,
    Renderer,
    RobotId,
    ScenarioConfig,
    ScenarioId,
    SwarmAgent,
    Terrain,
    TerrainObservation,
    Time,
    TimeUnit,
)
from swarm_field.simulation import (
    FailureCheck,
    FailureConfig,
    MetricSample,
    MetricsCollector,
    MetricsConfig,
    PestDynamicsConfig,
    PestSourceBroadcast,
    RobotDecision,
    SimulationContext,
    SimulationLoop,
    WorldState,
    WorldTick,
    build_scenario,
    build_world,
)
from swarm_field.swarm import (
    AgentMemory,
    ArbiterConfig,
    BatteryRequestSeeder,
    BeaconConfig,
    BeaconLink,
    BeaconView,
    Behavior,
    ComposedAgent,
    DockSeeder,
    ExploreBehaviour,
    ExploreConfig,
    ExploreSeeder,
    FieldProvider,
    FieldSeeder,
    HopField,
    KnowledgeSnapshot,
    Navigator,
    ObjectiveField,
    RechargeConfig,
    Regroup,
    RegroupSeeder,
    ReturnToDock,
    SensedSurroundings,
    SharingConfig,
    SupplyBattery,
    TreatByBeacon,
    TreatInfestation,
    TreatSeeder,
    UtilityArbiter,
    capabilities_for,
)

T = TypeVar("T")
"""What a ``Registry`` holds: a swappable brain part."""

PartFactory = Callable[[dict[str, Any], Random], T]
"""Builds one brain part from its parameters and the run's random source."""


class Registry[T]:
    """Named factories for swappable brain parts, so a spec can be data rather than code."""

    def __init__(self) -> None:
        self._factories: dict[str, PartFactory[T]] = {}

    def register(self, name: str) -> Callable[[PartFactory[T]], PartFactory[T]]:
        """Return a decorator that files a part factory under ``name``."""

        def add(factory: PartFactory[T]) -> PartFactory[T]:
            self._factories[name] = factory
            return factory

        return add

    def build(self, name: str, params: dict[str, Any], rng: Random) -> T:
        """Build the part registered under ``name``."""
        return self._factories[name](params, rng)

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def names(self) -> tuple[str, ...]:
        """Every registered name, sorted, for error messages and documentation."""
        return tuple(sorted(self._factories))


BEHAVIORS: Registry[Behavior] = Registry()
SEEDERS: Registry[FieldSeeder] = Registry()
FIELDS: Registry[ObjectiveField] = Registry()
"""Objective fields that are not seeders, such as the hop field."""

_NOISE_SEED_OFFSET = 0x5EED
"""Keeps the fault-injection random stream distinct from the world's for a given seed."""


@BEHAVIORS.register("treat-infestation")
def _build_treat_infestation(params: dict[str, Any], rng: Random) -> TreatInfestation:
    """Build the pest-response behaviour, following the TREAT seeder the spec names first.

    It leaves a pest to a peer already treating it with ``"treat_yield"``.

    Raises:
        ValueError: If the spec names no seeder, so there is no field to follow.
    """
    seeders: tuple[str, ...] = tuple(params.get("seeders", (TreatSeeder.name,)))
    if not seeders:
        raise ValueError("treat-infestation needs a seeder to follow")
    yield_to_treating_peer: bool = params.get("treat_yield", False)
    return TreatInfestation(seeders[0], yield_to_treating_peer=yield_to_treating_peer)


@BEHAVIORS.register(TreatByBeacon.name)
def _build_treat_by_beacon(params: dict[str, Any], rng: Random) -> TreatByBeacon:
    """Build the pest response on hop beacons, following the field the spec names first."""
    seeders: tuple[str, ...] = tuple(params.get("seeders", (HopField.name,)))
    if not seeders:
        raise ValueError(f"{TreatByBeacon.name} needs a field to follow")
    config, beacons = _beacon_parts(params, TreatByBeacon.name)
    return TreatByBeacon(config, beacons, seeders[0])


@FIELDS.register(HopField.name)
def _build_hop_field(params: dict[str, Any], rng: Random) -> HopField:
    """Build the hop field of the TREAT beacons, over the robot's own beacons."""
    config, beacons = _beacon_parts(params, f"the {HopField.name} field")
    return HopField(config, beacons)


def _beacon_parts(params: dict[str, Any], part: str) -> tuple[BeaconConfig, BeaconView]:
    """The spec's ``BeaconConfig``, and the robot's beacons from its ``BeaconLink``.

    Raises:
        ValueError: If the spec has no ``BeaconConfig``, so its robots keep and broadcast no
            beacons.
    """
    config: BeaconConfig | None = params.get("beacons")
    beacons: BeaconView | None = params.get("beacon_view")
    if config is None or beacons is None:
        raise ValueError(f"{part} needs BrainSpec.beacons")
    return config, beacons


@SEEDERS.register("treat")
def _build_treat_seeder(params: dict[str, Any], rng: Random) -> TreatSeeder:
    """Build the TREAT seeder, with ``"treat_window_blocks"`` and ``"treat_yield"`` if given."""
    window_blocks: int = params.get("treat_window_blocks", 2)
    yield_to_closer: bool = params.get("treat_yield", False)
    return TreatSeeder(window_blocks=window_blocks, yield_to_closer=yield_to_closer)


@BEHAVIORS.register("supply-battery")
def _build_supply_battery(params: dict[str, Any], rng: Random) -> SupplyBattery:
    """Build battery support, following the battery-support seeder named by the spec."""
    seeders: tuple[str, ...] = tuple(params.get("seeders", (BatteryRequestSeeder.name,)))
    seeder = next((name for name in seeders if name == BatteryRequestSeeder.name), None)
    if seeder is None:
        raise ValueError("supply-battery needs the battery-support seeder")
    return SupplyBattery(seeder)


@BEHAVIORS.register("return-to-dock")
def _build_return_to_dock(params: dict[str, Any], rng: Random) -> ReturnToDock:
    """Build recharge behaviour with the scenario's docks and optional policy."""
    docks: tuple[CellId, ...] = tuple(params.get("docks", ()))
    recharge: RechargeConfig | None = params.get("recharge")
    seeders: tuple[str, ...] = tuple(params.get("seeders", ("dock",)))
    seeder = next((name for name in seeders if name == "dock"), "dock")
    return ReturnToDock(seeder, docks=docks, recharge_config=recharge)


@SEEDERS.register("battery-support")
def _build_battery_request_seeder(params: dict[str, Any], rng: Random) -> BatteryRequestSeeder:
    """Build the peer battery-request field seeder."""
    return BatteryRequestSeeder()


@SEEDERS.register("dock")
def _build_dock_seeder(params: dict[str, Any], rng: Random) -> DockSeeder:
    """Build fixed attraction seeds for the docks configured in the scenario."""
    return DockSeeder(tuple(params.get("docks", ())))


@BEHAVIORS.register(ExploreBehaviour.name)
def _build_explore(params: dict[str, Any], rng: Random) -> ExploreBehaviour:
    """Build the exploration behaviour, with the ``ExploreConfig`` under ``"explore"``, if given."""
    config: ExploreConfig | None = params.get("explore")
    return ExploreBehaviour(ExploreSeeder.name, config)


@SEEDERS.register("explore")
def _build_explore_seeder(params: dict[str, Any], rng: Random) -> ExploreSeeder:
    """Build the EXPLORE seeder from the ``ExploreConfig`` under ``"explore"``, if given."""
    config: ExploreConfig | None = params.get("explore")
    return ExploreSeeder(config)


@BEHAVIORS.register("regroup")
def _build_regroup(params: dict[str, Any], rng: Random) -> Regroup:
    """Build the regroup behaviour, with the silence timeout under ``"regroup-silence"`` and the
    rally point under ``"rally-point"``, if given."""
    silence: Time = params.get("regroup-silence", 2 * TimeUnit.MINUTE)
    rally_point: CellId | None = params.get("rally-point")
    return Regroup(RegroupSeeder.name, silence, rally_point)


@SEEDERS.register("regroup")
def _build_regroup_seeder(params: dict[str, Any], rng: Random) -> RegroupSeeder:
    """Build the REGROUP seeder, with the rally point under ``"rally-point"``, if given."""
    rally_point: CellId | None = params.get("rally-point")
    return RegroupSeeder(rally_point=rally_point)


@dataclass(frozen=True)
class BrainSpec:
    """How to assemble one robot's brain, by name.

    This is the data-driven assembly of architecture §A.4. Every brain gets a
    ``UtilityArbiter`` with ``arbiter``, and shares its memory as ``sharing`` says; neither
    has a registry entry.

    Attributes:
        behaviors: Registry names of the behaviors, combined at every decision point. Order
            does not matter, the orchestrator scores them all.
        seeders: Registry names of the seeders and objective fields to register in the
            navigator, which is how a behaviour reaches a field: it names one of them. A
            behaviour that follows a single field takes the first name.
        navigator: Registry name of the movement layer. Only ``"potential-field"`` exists.
        params: Per-part parameters, passed through to the named factory, e.g. an
            ``ExploreConfig`` under ``"explore"`` for the EXPLORE seeder and behaviour.
        arbiter: The overrides and hysteresis of the robot's ``UtilityArbiter``.
        sharing: What the robot broadcasts of its memory, and how often.
        beacons: How beacons are kept, relayed and weighed, or ``None`` for a robot that
            neither keeps nor broadcasts them.
        keeps_map: Whether the robot's memory stores terrain and infestation facts. A brain of
            the hop method alone needs none; without a map it has no quadtree, so a seeder in
            ``seeders`` is refused.
    """

    behaviors: tuple[str, ...] = ("treat-infestation",)
    seeders: tuple[str, ...] = ("treat",)
    navigator: str = "potential-field"
    params: dict[str, Any] = field(default_factory=dict)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    sharing: SharingConfig = field(default_factory=SharingConfig)
    beacons: BeaconConfig | None = None
    keeps_map: bool = True

    def describe(self) -> str:
        """The parts of this brain by name, e.g. ``behaviors=explore seeders=explore``.

        Parameters are named but not spelled out, so a summary table stays readable. A brain
        with beacons, or without a map, says so.
        """
        parts = [
            "behaviors=" + "+".join(self.behaviors),
            "seeders=" + "+".join(self.seeders),
            "navigator=" + self.navigator,
            "sharing=" + self.sharing.describe(),
        ]
        if self.params:
            parts.append("params=" + "+".join(sorted(self.params)))
        if self.beacons is not None:
            parts.append("beacons")
        if not self.keeps_map:
            parts.append("no-map")
        return " ".join(parts)


@dataclass(frozen=True)
class ExperimentConfig:
    """One point of a sweep: a scenario, a brain, and how long to run it.

    Attributes:
        label: Name recorded on the resulting summary, so a sweep's results are traceable.
        scenario: The scenario to run, carrying its own seed.
        scenario_name: Which preset ``scenario`` came from, for the summary's provenance. Empty
            for a scenario assembled by hand rather than by ``build_experiment``.
        brain: How to assemble each robot's brain.
        brains_by_body: A different brain for the robots of some body types; the others get
            ``brain``.
        brains_by_robot: A brain for single robots, by id, e.g. for robots of one body that
            play different roles; it wins over ``brains_by_body`` and ``brain``.
        robots_know_terrain: Whether every robot starts with the field's terrain map in its
            memory. Otherwise it learns an obstacle only once it stands next to it.
        stagger_decisions: Whether each robot decides first at a random time within one
            ``decision_interval``, drawn from the seed, instead of all robots at time zero. Runs
            with different seeds then interleave the robots' decisions differently, also when
            nothing else in the run is random.
        until: How far to run, in simulation time.
        metrics: Sampling cadence and target threshold.
        decision_interval: Time between every robot's consecutive decisions.
    """

    label: str
    scenario: ScenarioConfig
    scenario_name: str = ""
    brain: BrainSpec = field(default_factory=BrainSpec)
    brains_by_body: Mapping[BodyType, BrainSpec] = field(default_factory=dict)
    brains_by_robot: Mapping[RobotId, BrainSpec] = field(default_factory=dict)
    robots_know_terrain: bool = False
    stagger_decisions: bool = False
    until: Time = TimeUnit.HOUR
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    decision_interval: Time = 5_000

    def with_sharing(self, sharing: SharingConfig) -> ExperimentConfig:
        """This experiment with every robot, whatever its body, sharing as ``sharing`` says."""
        by_body: dict[BodyType, BrainSpec] = {}
        for body, brain in self.brains_by_body.items():
            by_body[body] = replace(brain, sharing=sharing)
        by_robot: dict[RobotId, BrainSpec] = {}
        for robot, brain in self.brains_by_robot.items():
            by_robot[robot] = replace(brain, sharing=sharing)
        return replace(
            self,
            brain=replace(self.brain, sharing=sharing),
            brains_by_body=by_body,
            brains_by_robot=by_robot,
        )

    def brain_of(self, robot: RobotId, body: BodyType) -> BrainSpec:
        """The brain of one robot: its own, else its body's, else ``brain``."""
        own = self.brains_by_robot.get(robot)
        if own is not None:
            return own
        return self.brains_by_body.get(body, self.brain)


def run_experiment(
    config: ExperimentConfig,
    renderer: Renderer | None = None,
) -> tuple[MetricsSample, ...]:
    """Assemble one seeded experiment, run it to its horizon, and return its metrics series.

    A study that measures more than the series steps a ``LiveRun`` instead, and a batch of runs
    goes through ``swarm_field.experiments.Experiment``.

    Args:
        config: The scenario, brain and duration to run.
        renderer: Draws the world on every metric tick. Because a render only reads ground truth,
            passing one changes nothing about the run or the series it returns.
    """
    ctx, collector = build_run(config, renderer)
    SimulationLoop().run(ctx, config.until)
    return collector.series()


def build_run(
    config: ExperimentConfig,
    renderer: Renderer | None = None,
) -> tuple[SimulationContext, MetricsCollector]:
    """Assemble one run's object graph and seed its first events.

    Every random draw happens here, in a fixed order, so the run is a function of the seed and
    the configuration. The context's ``rng`` is then left to the pest dynamics, which draw from
    it in event order; the metrics collector holds no random source of its own, so recording
    metrics cannot change the world a run produces.

    Args:
        config: The experiment to assemble.
        renderer: Draws the world on every metric tick. The run is unaffected either way, so a
            sweep can be watched without changing what it measures.
    """
    scenario = config.scenario
    rng = Random(scenario.seed)
    world = build_world(scenario)
    known: KnowledgeSnapshot | None = None
    if config.robots_know_terrain:
        known = terrain_knowledge(world)
    agents: dict[RobotId, SwarmAgent] = {}
    for spec in scenario.robots:
        brain = config.brain_of(spec.robot_id, spec.body)
        agents[spec.robot_id] = build_agent(
            brain,
            spec.robot_id,
            scenario.field,
            rng,
            known,
            flies=capabilities_for(spec.body).can_fly,
            docks=scenario.docks,
        )

    collector = MetricsCollector(config=config.metrics)
    collector.bind(world)

    ctx = SimulationContext(
        now=0,
        rng=rng,
        agents=agents,
        world=world,
        metrics=collector,
        renderer=renderer,
        position_error=scenario.position_error,
        noise=_noise_source(scenario.seed),
    )
    seed_initial_events(ctx, config)
    return ctx, collector


def _noise_source(seed: int) -> Random:
    """The fault-injection random source, seeded apart from the world's.

    A robustness run has to be comparable with its baseline: the pests must evolve exactly as
    they would have without the faults. Deriving the seed from the scenario seed keeps the pair
    reproducible, while keeping the streams apart keeps the comparison honest.
    """
    return Random(seed ^ _NOISE_SEED_OFFSET)


def seed_initial_events(ctx: SimulationContext, config: ExperimentConfig) -> None:
    """Schedule a run's first events.

    The order is fixed and load-bearing: ``EventQueue`` breaks equal timestamps by insertion
    order, so this is what makes a run with several events at time zero reproducible.
    """
    for robot_id in sorted(ctx.agents):
        first: Time = 0
        if config.stagger_decisions:
            first = ctx.rng.randrange(config.decision_interval)
        ctx.schedule(
            RobotDecision(time=first, robot=robot_id, decision_interval=config.decision_interval)
        )
    ctx.schedule(
        WorldTick(time=0, config=PestDynamicsConfig.from_spec(config.scenario.pest_dynamics))
    )
    if config.scenario.pest_sources is not None:
        ctx.schedule(PestSourceBroadcast(time=0, spec=config.scenario.pest_sources))
    if config.scenario.failures.probability > 0.0:
        ctx.schedule(FailureCheck(time=0, config=FailureConfig.from_spec(config.scenario.failures)))
    ctx.schedule(MetricSample(time=0, config=config.metrics))


def build_agent(
    spec: BrainSpec,
    robot_id: RobotId,
    field: FieldGeometry,
    rng: Random,
    known: KnowledgeSnapshot | None = None,
    *,
    flies: bool = False,
    docks: tuple[CellId, ...] = (),
) -> ComposedAgent:
    """Resolve a ``BrainSpec`` into one assembled brain for one robot.

    The robot's ``BodyType`` is not a parameter: the capability gate of ``ComposedAgent`` takes
    it from each ``Perception``, and a body or a robot gets its own brain through
    ``ExperimentConfig`` (architecture §A.3). Only whether the body ``flies`` reaches the brain:
    a flying robot plans over obstacles, which it can enter. The robot's memory starts with what
    ``known`` holds, and its dock parts lead to ``docks``. A brain with beacons gets a
    ``BeaconLink``, whose beacons its hop parts read, and steers headings from what it senses
    around itself at each decision only, not from a map.
    """
    memory = AgentMemory(robot_id, field, keeps_map=spec.keeps_map)
    if known is not None:
        memory.merge_snapshot(known)
    link: BeaconLink | None = None
    if spec.beacons is not None:
        link = BeaconLink(robot_id, spec.beacons, field)
    params = _params_for(spec, link, docks=docks)
    behaviors: list[Behavior] = [BEHAVIORS.build(name, params, rng) for name in spec.behaviors]
    fields = build_field_provider(spec, memory, rng, link, flies=flies, docks=docks)
    heading_sensing: SensedSurroundings | None = None
    if link is not None:
        heading_sensing = link.surroundings
    navigator = Navigator(fields.quadtree, heading_sensing=heading_sensing)
    return ComposedAgent(
        robot_id=robot_id,
        behaviors=behaviors,
        memory=memory,
        fields=fields,
        navigator=navigator,
        arbiter=UtilityArbiter(spec.arbiter),
        sharing=spec.sharing,
        beacon_link=link,
    )


def terrain_knowledge(world: WorldState) -> KnowledgeSnapshot:
    """The terrain of every cell as a snapshot to start each robot's memory with.

    It is merged once here, and every robot merges the snapshot, rather than each robot merging
    the terrain fact by fact.
    """
    memory = AgentMemory(0, world.field)
    memory.merge(terrain_map(world))
    return memory.knowledge()


def terrain_map(world: WorldState) -> list[TerrainObservation]:
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


def _params_for(
    spec: BrainSpec, link: BeaconLink | None = None, *, docks: tuple[CellId, ...] = ()
) -> dict[str, Any]:
    """The spec's shared parameters, with the seeders, beacons, robot's beacon view and the
    scenario's docks folded in.

    The seeders and beacons are fields of their own rather than parameters so that swapping one
    does not mean restating it, and so that the spec gives each in one place. The beacon view is
    the robot's own, for the hop parts built for that robot.
    """
    beacon_view: BeaconView | None = None
    if link is not None:
        beacon_view = link.memory
    return {
        **spec.params,
        "seeders": spec.seeders,
        "beacons": spec.beacons,
        "beacon_view": beacon_view,
        "docks": docks,
    }


def build_field_provider(
    spec: BrainSpec,
    memory: AgentMemory,
    rng: Random,
    link: BeaconLink | None = None,
    *,
    flies: bool = False,
    docks: tuple[CellId, ...] = (),
) -> FieldProvider:
    """Build the objective fields of the navigator named by a spec: its seeders and its fields.

    A name is looked up in ``FIELDS`` first, then in ``SEEDERS``. The potential field is planned
    over the cells the robot believes it can enter: a ground robot avoids obstacles, a robot that
    ``flies`` crosses them.

    Raises:
        ValueError: If the spec names a navigator that is not registered yet.
        KeyError: If the spec names a field or seeder that is not registered.
    """
    if spec.navigator != "potential-field":
        raise ValueError(f"unknown navigator: {spec.navigator}")
    params = _params_for(spec, link, docks=docks)
    seeders: list[FieldSeeder] = []
    fields: list[ObjectiveField] = []
    for name in spec.seeders:
        if name in FIELDS:
            fields.append(FIELDS.build(name, params, rng))
        else:
            seeders.append(SEEDERS.build(name, params, rng))
    return FieldProvider(memory, flies=flies, seeders=seeders, fields=fields)


def build_experiment(
    label: str,
    scenario: ScenarioId,
    seed: int,
    brain: BrainSpec | None = None,
    until: Time = TimeUnit.HOUR,
    metrics: MetricsConfig | None = None,
    decision_interval: Time = 5_000,
) -> ExperimentConfig:
    """Assemble one experiment from a scenario preset, for callers that do not build configs.

    Args:
        label: Name recorded on the resulting summary.
        scenario: Which built-in scenario preset to run.
        seed: The seed making the run reproducible.
        brain: How to assemble each robot's brain; the default spec when omitted.
        until: How far to run, in simulation time.
        metrics: Sampling cadence and target threshold.
        decision_interval: Time between every robot's consecutive decisions.
    """
    return ExperimentConfig(
        label=label,
        scenario=build_scenario(scenario, seed=seed),
        scenario_name=scenario.name,
        brain=brain if brain is not None else BrainSpec(),
        until=until,
        metrics=metrics if metrics is not None else MetricsConfig(),
        decision_interval=decision_interval,
    )
