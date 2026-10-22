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

import csv
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from random import Random
from typing import Any, TypeVar

from swarm_field.contracts import (
    BodyType,
    CellId,
    FailureMode,
    FailureSpec,
    FieldGeometry,
    MetricsSample,
    PositionErrorSpec,
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
    RobotDecision,
    SimulationContext,
    SimulationLoop,
    WorldState,
    WorldTick,
    build_scenario,
    build_world,
    write_metrics_csv,
    write_metrics_json,
)
from swarm_field.swarm import (
    AgentMemory,
    ArbiterConfig,
    Behavior,
    ComposedAgent,
    ExploreConfig,
    ExploreField,
    ExploreSeeder,
    FieldProvider,
    FieldSeeder,
    Navigator,
    Regroup,
    RegroupSeeder,
    TreatInfestation,
    TreatSeeder,
    UtilityArbiter,
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

    def names(self) -> tuple[str, ...]:
        """Every registered name, sorted, for error messages and documentation."""
        return tuple(sorted(self._factories))


BEHAVIORS: Registry[Behavior] = Registry()
SEEDERS: Registry[FieldSeeder] = Registry()

_NOISE_SEED_OFFSET = 0x5EED
"""Keeps the fault-injection random stream distinct from the world's for a given seed."""


@BEHAVIORS.register("treat-infestation")
def _build_treat_infestation(params: dict[str, Any], rng: Random) -> TreatInfestation:
    """Build the pest-response behaviour, following the TREAT seeder the spec names first.

    Raises:
        ValueError: If the spec names no seeder, so there is no field to follow.
    """
    seeders: tuple[str, ...] = tuple(params.get("seeders", (TreatSeeder.name,)))
    if not seeders:
        raise ValueError("treat-infestation needs a seeder to follow")
    return TreatInfestation(seeders[0])


@SEEDERS.register("treat")
def _build_treat_seeder(params: dict[str, Any], rng: Random) -> TreatSeeder:
    """Build the TREAT seeder. It has no parameters yet, so it takes none."""
    return TreatSeeder()


@BEHAVIORS.register("explore")
def _build_explore(params: dict[str, Any], rng: Random) -> ExploreField:
    """Build the memory-scored exploration behaviour for the EXPLORE seeder."""
    return ExploreField(ExploreSeeder.name)


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
    ``UtilityArbiter`` with ``arbiter``; the communication strategy is fixed in
    ``ComposedAgent``, so it has no registry entry.

    Attributes:
        behaviors: Registry names of the behaviors, combined at every decision point. Order
            does not matter, the orchestrator scores them all.
        seeders: Registry names of the seeders to register in the navigator, which is how a
            behaviour reaches a field: it names one of them. A behaviour that follows a single
            seeder takes the first name.
        navigator: Registry name of the movement layer. Only ``"potential-field"`` exists.
        params: Per-part parameters, passed through to the named factory, e.g. an
            ``ExploreConfig`` under ``"explore"`` for the EXPLORE seeder.
        arbiter: The overrides and hysteresis of the robot's ``UtilityArbiter``.
    """

    behaviors: tuple[str, ...] = ("treat-infestation",)
    seeders: tuple[str, ...] = ("treat",)
    navigator: str = "potential-field"
    params: dict[str, Any] = field(default_factory=dict)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)

    def describe(self) -> str:
        """The parts of this brain by name, e.g. ``behaviors=explore seeders=explore``.

        Parameters are named but not spelled out, so a summary table stays readable.
        """
        parts = [
            "behaviors=" + "+".join(self.behaviors),
            "seeders=" + "+".join(self.seeders),
            "navigator=" + self.navigator,
        ]
        if self.params:
            parts.append("params=" + "+".join(sorted(self.params)))
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
    robots_know_terrain: bool = False
    stagger_decisions: bool = False
    until: Time = TimeUnit.HOUR
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    decision_interval: Time = 5_000


@dataclass(frozen=True)
class RunMetadata:
    """How one run was set up: what tells two rows of a results table apart.

    Every field is a plain value, so the metadata fits next to the results in one CSV row.

    Attributes:
        horizon_ms: How far the run went, in simulation time.
        decision_interval_ms: Time between a robot's consecutive decisions.
        metrics_period_ms: Time between two metrics samples.
        field_width: Field width in cells.
        field_height: Field height in cells.
        cell_size_m: The side of a cell in metres.
        robots: How many robots the run started with.
        bodies: How many robots of each body type, e.g. ``KILLER=3 SCOUT=5``.
        radio_ranges_m: The robots' broadcast ranges in metres, each distinct range once, e.g.
            ``300`` or ``100 300``.
        brain: The brain of every robot whose body has no brain of its own.
        brains_by_body: The brain of each body type that has one of its own, e.g.
            ``KILLER: behaviors=...``; empty if every robot has ``brain``.
        robots_know_terrain: Whether the robots started with the terrain map.
        stagger_decisions: Whether the robots' first decisions were spread over one interval.
        pest_spawn_probability: Chance per cell and pest update that a pest spawns.
        pest_growth_rate: How much a pest's severity grows per pest update.
        pest_spread_probability: Chance per pest update that a pest infests a neighbour.
        failure_probability: Chance per check that a live robot fails.
        failure_mode: What a failure is, by name.
        failure_period_ms: Time between two failure checks.
        position_error_probability: Chance per decision that a robot mislocates itself.
        position_error_max_offset: Largest such displacement, in cells.
    """

    horizon_ms: Time
    decision_interval_ms: Time
    metrics_period_ms: Time
    field_width: int
    field_height: int
    cell_size_m: float
    robots: int
    bodies: str
    radio_ranges_m: str
    brain: str
    brains_by_body: str
    robots_know_terrain: bool
    stagger_decisions: bool
    pest_spawn_probability: float
    pest_growth_rate: float
    pest_spread_probability: float
    failure_probability: float
    failure_mode: str
    failure_period_ms: Time
    position_error_probability: float
    position_error_max_offset: int

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> RunMetadata:
        """The setup of the run that ``config`` describes."""
        scenario = config.scenario
        bodies: Counter[str] = Counter()
        for robot in scenario.robots:
            bodies[robot.body.name] += 1
        counts: list[str] = []
        for body, count in sorted(bodies.items()):
            counts.append(f"{body}={count}")
        ranges: list[str] = []
        for radio_range in sorted({robot.radio_range for robot in scenario.robots}):
            ranges.append(f"{radio_range:g}")
        own_brains: list[str] = []
        for body in sorted(config.brains_by_body, key=lambda body: body.name):
            own_brains.append(f"{body.name}: {config.brains_by_body[body].describe()}")
        return cls(
            horizon_ms=config.until,
            decision_interval_ms=config.decision_interval,
            metrics_period_ms=config.metrics.period,
            field_width=scenario.field.width,
            field_height=scenario.field.height,
            cell_size_m=scenario.field.cell_size,
            robots=len(scenario.robots),
            bodies=" ".join(counts),
            radio_ranges_m=" ".join(ranges),
            brain=config.brain.describe(),
            brains_by_body="; ".join(own_brains),
            robots_know_terrain=config.robots_know_terrain,
            stagger_decisions=config.stagger_decisions,
            pest_spawn_probability=scenario.pest_dynamics.spawn_probability,
            pest_growth_rate=scenario.pest_dynamics.growth_rate,
            pest_spread_probability=scenario.pest_dynamics.spread_probability,
            failure_probability=scenario.failures.probability,
            failure_mode=scenario.failures.mode.name,
            failure_period_ms=scenario.failures.period,
            position_error_probability=scenario.position_error.probability,
            position_error_max_offset=scenario.position_error.max_offset,
        )

    def to_row(self) -> dict[str, int | float | str | bool]:
        """The metadata as a flat ``dict`` of field name to value, for a summary row."""
        return asdict(self)


@dataclass(frozen=True)
class RunSummary:
    """What one run produced: its full metrics series and the figures read off it.

    Attributes:
        label: Name of the configuration this run used.
        scenario: Name of the scenario preset this run used.
        seed: The scenario seed the run was reproducible from.
        metadata: How the run was set up, from its ``ExperimentConfig``.
        series: Every metrics sample, oldest first.
        targets_remaining: Active targets left at the end of the run.
        coverage: Fraction of the crop cells sensed by the end of the run.
        mean_detection_latency_ms: Mean delay from infestation to first sighting, averaged over the
            run's samples, or ``None`` if nothing was detected.
        energy_per_area: Battery fraction spent per square metre over the run.
        messages_sent: Messages handed to the channel over the run.
        messages_received: Times a message reached a robot in range over the run.
        active_robots: Robots still taking part when the run ended.
    """

    label: str
    scenario: str
    seed: int
    metadata: RunMetadata
    series: tuple[MetricsSample, ...]
    targets_remaining: int
    coverage: float
    mean_detection_latency_ms: float | None
    energy_per_area: float
    messages_sent: int
    messages_received: int
    active_robots: int

    @classmethod
    def from_run(cls, config: ExperimentConfig, series: Iterable[MetricsSample]) -> RunSummary:
        """Summarise the run of ``config`` from its metrics series.

        The whole-run figures are taken from the last sample, and the metadata from ``config``.

        Raises:
            ValueError: If the series holds no sample.
        """
        taken = tuple(series)
        if not taken:
            raise ValueError("a run summary needs at least one metrics sample")
        final = taken[-1]
        return cls(
            label=config.label,
            scenario=config.scenario_name,
            seed=config.scenario.seed,
            metadata=RunMetadata.from_config(config),
            series=taken,
            targets_remaining=final.targets_remaining,
            coverage=final.coverage,
            mean_detection_latency_ms=_mean(sample.detection_latency_ms for sample in taken),
            energy_per_area=final.energy_per_area,
            messages_sent=final.messages_sent,
            messages_received=final.messages_received,
            active_robots=final.active_robots,
        )

    def to_row(self) -> dict[str, int | float | str | bool | None]:
        """This summary as one flat record, the shape a sweep's table is written in.

        The results come first and the metadata after them, so the result columns keep their
        place when the metadata grows.
        """
        row: dict[str, int | float | str | bool | None] = {
            "label": self.label,
            "scenario": self.scenario,
            "seed": self.seed,
            "samples": len(self.series),
            "targets_remaining": self.targets_remaining,
            "coverage": self.coverage,
            "mean_detection_latency_ms": self.mean_detection_latency_ms,
            "energy_per_area": self.energy_per_area,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "active_robots": self.active_robots,
        }
        row.update(self.metadata.to_row())
        return row

    def export(self, directory: Path) -> tuple[Path, Path]:
        """Write this run's series as CSV and JSON under ``directory``, and return both paths.

        The label is turned into a single file name: a sweep labels runs ``SCENARIO/SEED``, and
        a label is display data, so it must not be allowed to become a directory path.
        """
        stem = _file_stem(self.label)
        csv_path = directory / f"{stem}.csv"
        json_path = directory / f"{stem}.json"
        write_metrics_csv(self.series, csv_path)
        write_metrics_json(self.series, json_path)
        return csv_path, json_path


def run_experiment(
    config: ExperimentConfig,
    renderer: Renderer | None = None,
) -> RunSummary:
    """Assemble, run and summarise one seeded experiment.

    Args:
        config: The scenario, brain and duration to run.
        renderer: Draws the world on every metric tick. Because a render only reads ground truth,
            passing one changes nothing about the run or the summary it returns.
    """
    ctx, collector = build_run(config, renderer)
    SimulationLoop().run(ctx, config.until)
    return RunSummary.from_run(config, collector.series())


def sweep(
    configs: Iterable[ExperimentConfig],
    renderer: Renderer | None = None,
) -> tuple[RunSummary, ...]:
    """Run several configurations in order and return one summary per configuration.

    Each configuration carries its own seed, so a sweep is reproducible end to end and its
    results can be compared row by row.
    """
    return tuple(run_experiment(config, renderer) for config in configs)


@dataclass(frozen=True)
class RobustnessCase:
    """One condition a robustness sweep runs a scenario under.

    A case is a label plus the two robustness knobs, so a sweep row can be traced back to the
    condition that produced it. A case with both probabilities at zero is the baseline.

    Attributes:
        label: Name of the condition, used to label the run.
        failures: How often robots fail, and how.
        position_error: How often robots mislocate themselves.
    """

    label: str
    failures: FailureSpec = field(default_factory=FailureSpec)
    position_error: PositionErrorSpec = field(default_factory=PositionErrorSpec)


def robustness_case_config(base: ExperimentConfig, case: RobustnessCase) -> ExperimentConfig:
    """The configuration of ``base`` under one robustness condition, labelled ``NAME/CASE``.

    Separated from ``robustness_sweep`` so a caller can reproduce a single condition exactly as
    the sweep built it, which is what makes rendering that condition again trustworthy: the same
    base and case produce the same run whether it was swept or drawn. ``base`` is left as it is.
    """
    name = base.scenario_name
    if not name:
        name = base.label
    scenario = replace(base.scenario, failures=case.failures, position_error=case.position_error)
    return replace(base, label=f"{name}/{case.label}", scenario=scenario)


def robustness_sweep(
    base: ExperimentConfig,
    cases: Iterable[RobustnessCase],
    renderer: Renderer | None = None,
) -> tuple[RunSummary, ...]:
    """Run one experiment under each robustness condition, from the seed of ``base``.

    Every case is run against the same seed, so a run and its baseline differ only in the
    faults: the pest dynamics see the same random stream either way. That is what makes a
    difference in the metrics attributable to the condition rather than to the weather.

    Args:
        base: The experiment every condition is applied to, e.g. from
            ``experiments.build_named_experiment``; its seed is shared by every case.
        cases: The conditions to run, including the baseline.
        renderer: Draws every case's world on every metric tick. One renderer sees all of them,
            so it cannot tell the conditions apart; render a single condition through
            ``robustness_case_config`` when the frames need to be attributable.
    """
    runs: list[RunSummary] = []
    for case in cases:
        runs.append(run_experiment(robustness_case_config(base, case), renderer))
    return tuple(runs)


def default_robustness_cases(
    failure_probabilities: Iterable[float] = (0.001, 0.005),
    position_error_probabilities: Iterable[float] = (0.1, 0.3),
    mode: FailureMode = FailureMode.DEAD_BATTERY,
) -> tuple[RobustnessCase, ...]:
    """The conditions the robustness note reports: a baseline, then one axis at a time.

    Only one axis is perturbed at a time, so each row answers one question. The modes are swept
    separately, since a battery death and a radio loss cost the swarm different things.
    """
    cases = [RobustnessCase(label="baseline")]
    for probability in failure_probabilities:
        cases.append(
            RobustnessCase(
                label=f"failure-{mode.name.lower()}-{probability:g}",
                failures=FailureSpec(probability=probability, mode=mode),
            )
        )
    for probability in position_error_probabilities:
        cases.append(
            RobustnessCase(
                label=f"position-error-{probability:g}",
                position_error=PositionErrorSpec(probability=probability),
            )
        )
    return tuple(cases)


def export_summary_csv(runs: Iterable[RunSummary], path: Path) -> Path:
    """Write one row per run to ``path``, so a sweep can be compared in a table."""
    rows = [run.to_row() for run in runs]
    if not rows:
        raise ValueError("there is nothing to export")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def export_summary_json(runs: Iterable[RunSummary], path: Path) -> Path:
    """Write one record per run to ``path`` as JSON."""
    rows = [run.to_row() for run in runs]
    if not rows:
        raise ValueError("there is nothing to export")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


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
    terrain: list[TerrainObservation] = []
    if config.robots_know_terrain:
        terrain = terrain_map(world)
    agents: dict[RobotId, SwarmAgent] = {}
    for spec in scenario.robots:
        brain = config.brains_by_body.get(spec.body, config.brain)
        agents[spec.robot_id] = build_agent(brain, spec.robot_id, scenario.field, rng, terrain)

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
    if config.scenario.failures.probability > 0.0:
        ctx.schedule(FailureCheck(time=0, config=FailureConfig.from_spec(config.scenario.failures)))
    ctx.schedule(MetricSample(time=0, config=config.metrics))


def build_agent(
    spec: BrainSpec,
    robot_id: RobotId,
    field: FieldGeometry,
    rng: Random,
    terrain: Sequence[TerrainObservation] = (),
) -> ComposedAgent:
    """Resolve a ``BrainSpec`` into one assembled brain for one robot.

    The robot's ``BodyType`` is not a parameter: ``ComposedAgent`` has no capability gate to
    apply it in, so a body gets its own brain through ``ExperimentConfig.brains_by_body``
    instead (architecture §A.3). The robot's memory starts with ``terrain``.
    """
    memory = AgentMemory(robot_id, field)
    memory.merge(terrain)
    behaviors: list[Behavior] = [
        BEHAVIORS.build(name, _params_for(spec), rng) for name in spec.behaviors
    ]
    fields = build_field_provider(spec, memory, rng)
    return ComposedAgent(
        robot_id=robot_id,
        behaviors=behaviors,
        memory=memory,
        fields=fields,
        navigator=Navigator(fields.quadtree),
        arbiter=UtilityArbiter(spec.arbiter),
    )


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


def _params_for(spec: BrainSpec) -> dict[str, Any]:
    """The spec's shared parameters, with the seeders the spec names folded in.

    The seeders are a field of their own rather than parameters so that swapping one does not
    mean restating it, and so that ``BrainSpec.seeders`` is the only place the names are given.
    """
    return {**spec.params, "seeders": spec.seeders}


def build_field_provider(spec: BrainSpec, memory: AgentMemory, rng: Random) -> FieldProvider:
    """Build the potential fields of the navigator named by a spec, over the seeders it names.

    Every v1 body is a ground robot, so the potential field is planned over land the robot
    believes it can enter. An aerial body would pass ``flies=True`` and ignore crop cells.

    Raises:
        ValueError: If the spec names a navigator that is not registered yet.
        KeyError: If the spec names a seeder that is not registered.
    """
    if spec.navigator != "potential-field":
        raise ValueError(f"unknown navigator: {spec.navigator}")
    params = _params_for(spec)
    seeders: list[FieldSeeder] = [SEEDERS.build(name, params, rng) for name in spec.seeders]
    return FieldProvider(memory, flies=False, seeders=seeders)


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


def _mean(values: Iterable[float | None]) -> float | None:
    """Mean of the values that are present, or ``None`` if none of them are."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _file_stem(label: str) -> str:
    """Turn a run label into one safe file name, collapsing separators and odd characters.

    Sweep labels read best as ``SCENARIO/SEED``, so the slash is what needs handling most; the
    rest is there so a hand-written label cannot produce a path that escapes the directory.
    """
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_" for character in label
    )
    return cleaned.strip("_") or "run"
