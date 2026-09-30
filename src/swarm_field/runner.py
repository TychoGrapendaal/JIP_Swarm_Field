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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, TypeVar

from swarm_field.contracts import (
    FieldGeometry,
    MetricsSample,
    RobotId,
    ScenarioConfig,
    ScenarioId,
    SwarmAgent,
    Time,
)
from swarm_field.simulation import (
    MetricSample,
    MetricsCollector,
    MetricsConfig,
    PestDynamicsConfig,
    RobotDecision,
    SimulationContext,
    SimulationLoop,
    WorldTick,
    build_scenario,
    build_world,
    write_metrics_csv,
    write_metrics_json,
)
from swarm_field.swarm import (
    AgentMemory,
    Behavior,
    ComposedAgent,
    Navigator,
    Objective,
    TreatInfestation,
    TreatObjective,
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
OBJECTIVES: Registry[Objective] = Registry()


@BEHAVIORS.register("treat-infestation")
def _build_treat_infestation(params: dict[str, Any], rng: Random) -> TreatInfestation:
    """Build the pest-response behaviour over the objective the spec names."""
    objective = OBJECTIVES.build(params.get("objective", "treat"), params, rng)
    return TreatInfestation(objective)


@OBJECTIVES.register("treat")
def _build_treat_objective(params: dict[str, Any], rng: Random) -> TreatObjective:
    """Build the TREAT objective. It has no parameters yet, so it takes none."""
    return TreatObjective()


@dataclass(frozen=True)
class BrainSpec:
    """How to assemble one robot's brain, by name.

    This is the data-driven assembly of architecture §A.4. The arbiter and communication
    strategy have no registry entries because ``ComposedAgent`` does not accept them yet, and
    ``explore`` has none because it is not a ``Behavior`` yet: it builds force maps but has no
    ``propose``. Each is added when the part behind it exists.

    Attributes:
        behaviors: Registry names of the behaviors, combined at every decision point. Order
            does not matter, the orchestrator scores them all.
        objective: Registry name of the objective those behaviors follow.
        navigator: Registry name of the movement layer. Only ``"potential-field"`` exists.
        params: Per-part parameters, passed through to the named factory.
    """

    behaviors: tuple[str, ...] = ("treat-infestation",)
    objective: str = "treat"
    navigator: str = "potential-field"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """One point of a sweep: a scenario, a brain, and how long to run it.

    Attributes:
        label: Name recorded on the resulting summary, so a sweep's results are traceable.
        scenario: The scenario to run, carrying its own seed.
        scenario_name: Which preset ``scenario`` came from, for the summary's provenance. Empty
            for a scenario assembled by hand rather than by ``build_experiment``.
        brain: How to assemble each robot's brain.
        until: How far to run, in simulation time.
        metrics: Sampling cadence and target threshold.
        dwell: Stationary decision interval used by every robot.
    """

    label: str
    scenario: ScenarioConfig
    scenario_name: str = ""
    brain: BrainSpec = field(default_factory=BrainSpec)
    until: Time = 3_600_000
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    dwell: Time = 5_000


@dataclass(frozen=True)
class RunSummary:
    """What one run produced: its full metrics series and the figures read off it.

    Attributes:
        label: Name of the configuration this run used.
        scenario: Name of the scenario preset this run used.
        seed: The scenario seed the run was reproducible from.
        series: Every metrics sample, oldest first.
        targets_remaining: Active targets left at the end of the run.
        coverage: Fraction of the crop cells sensed by the end of the run.
        mean_detection_latency_ms: Mean delay from infestation to first sighting, averaged over the
            run's samples, or ``None`` if nothing was detected.
        energy_per_area: Battery fraction spent per square metre over the run.
        messages_sent: Messages handed to the channel over the run.
    """

    label: str
    scenario: str
    seed: int
    series: tuple[MetricsSample, ...]
    targets_remaining: int
    coverage: float
    mean_detection_latency_ms: float | None
    energy_per_area: float
    messages_sent: int

    @classmethod
    def from_series(
        cls,
        label: str,
        scenario: str,
        seed: int,
        series: Iterable[MetricsSample],
    ) -> RunSummary:
        """Summarise a metrics series, taking the last sample for the whole-run figures."""
        taken = tuple(series)
        if not taken:
            raise ValueError("a run summary needs at least one metrics sample")
        final = taken[-1]
        return cls(
            label=label,
            scenario=scenario,
            seed=seed,
            series=taken,
            targets_remaining=final.targets_remaining,
            coverage=final.coverage,
            mean_detection_latency_ms=_mean(sample.detection_latency_ms for sample in taken),
            energy_per_area=final.energy_per_area,
            messages_sent=final.messages_sent,
        )

    def to_row(self) -> dict[str, int | float | str | None]:
        """This summary as one flat record, the shape a sweep's table is written in."""
        return {
            "label": self.label,
            "scenario": self.scenario,
            "seed": self.seed,
            "samples": len(self.series),
            "targets_remaining": self.targets_remaining,
            "coverage": self.coverage,
            "mean_detection_latency_ms": self.mean_detection_latency_ms,
            "energy_per_area": self.energy_per_area,
            "messages_sent": self.messages_sent,
        }

    def export(self, directory: Path) -> tuple[Path, Path]:
        """Write this run's series as CSV and JSON under ``directory``, and return both paths."""
        csv_path = directory / f"{self.label}.csv"
        json_path = directory / f"{self.label}.json"
        write_metrics_csv(self.series, csv_path)
        write_metrics_json(self.series, json_path)
        return csv_path, json_path


def run_experiment(config: ExperimentConfig) -> RunSummary:
    """Assemble, run and summarise one seeded experiment.

    Args:
        config: The scenario, brain and duration to run.
    """
    ctx, collector = build_run(config)
    SimulationLoop().run(ctx, config.until)
    return RunSummary.from_series(
        config.label, config.scenario_name, config.scenario.seed, collector.series()
    )


def sweep(configs: Iterable[ExperimentConfig]) -> tuple[RunSummary, ...]:
    """Run several configurations in order and return one summary per configuration.

    Each configuration carries its own seed, so a sweep is reproducible end to end and its
    results can be compared row by row.
    """
    return tuple(run_experiment(config) for config in configs)


def sweep_scenarios(
    scenarios: Iterable[ScenarioId],
    seeds: Iterable[int],
    brain: BrainSpec | None = None,
    until: Time = 3_600_000,
    metrics: MetricsConfig | None = None,
) -> tuple[RunSummary, ...]:
    """Run the cross product of ``scenarios`` and ``seeds``, labelling each run ``SCENARIO/SEED``.

    The label is derived rather than passed in, so a sweep's rows stay in the same order as the
    configuration that produced them and cannot be mislabelled by hand.
    """
    configs: list[ExperimentConfig] = []
    for scenario in scenarios:
        for seed in seeds:
            label = f"{scenario.name}/{seed}"
            configs.append(
                build_experiment(label, scenario, seed, brain=brain, until=until, metrics=metrics)
            )
    return sweep(configs)


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


def build_run(config: ExperimentConfig) -> tuple[SimulationContext, MetricsCollector]:
    """Assemble one run's object graph and seed its first events.

    Every random draw happens here, in a fixed order, so the run is a function of the seed and
    the configuration. The context's ``rng`` is then left to the pest dynamics, which draw from
    it in event order; the metrics collector holds no random source of its own, so recording
    metrics cannot change the world a run produces.

    Args:
        config: The experiment to assemble.
    """
    scenario = config.scenario
    rng = Random(scenario.seed)
    world = build_world(scenario)
    agents: dict[RobotId, SwarmAgent] = {
        spec.robot_id: build_agent(config.brain, spec.robot_id, scenario.field, rng)
        for spec in scenario.robots
    }

    collector = MetricsCollector(config=config.metrics)
    collector.bind(world)

    ctx = SimulationContext(
        now=0,
        rng=rng,
        agents=agents,
        world=world,
        metrics=collector,
    )
    seed_initial_events(ctx, config)
    return ctx, collector


def seed_initial_events(ctx: SimulationContext, config: ExperimentConfig) -> None:
    """Schedule a run's first events.

    The order is fixed and load-bearing: ``EventQueue`` breaks equal timestamps by insertion
    order, so this is what makes a run with several events at time zero reproducible.
    """
    for robot_id in sorted(ctx.agents):
        ctx.schedule(RobotDecision(time=0, robot=robot_id, dwell=config.dwell))
    ctx.schedule(
        WorldTick(time=0, config=PestDynamicsConfig.from_spec(config.scenario.pest_dynamics))
    )
    ctx.schedule(MetricSample(time=0, config=config.metrics))


def build_agent(
    spec: BrainSpec,
    robot_id: RobotId,
    field: FieldGeometry,
    rng: Random,
) -> ComposedAgent:
    """Resolve a ``BrainSpec`` into one assembled brain for one robot.

    The robot's ``BodyType`` is not a parameter yet: ``ComposedAgent`` has no capability gate
    to apply it in, so every v1 body builds the same brain. It is added when the gate does
    (architecture §A.3).
    """
    memory = AgentMemory(robot_id, field)
    behaviors: list[Behavior] = [
        BEHAVIORS.build(name, _params_for(spec), rng) for name in spec.behaviors
    ]
    return ComposedAgent(
        robot_id=robot_id,
        behaviors=behaviors,
        memory=memory,
        navigator=build_navigator(spec, memory),
    )


def _params_for(spec: BrainSpec) -> dict[str, Any]:
    """The spec's shared parameters, with the objective the spec names folded in.

    The objective is a field of its own rather than a parameter so that swapping it does not
    mean restating it, and so that ``BrainSpec.objective`` is the only place it is named.
    """
    return {**spec.params, "objective": spec.objective}


def build_navigator(spec: BrainSpec, memory: AgentMemory) -> Navigator:
    """Build the movement layer named by a spec.

    Every v1 body is a ground robot, so the potential field is planned over land the robot
    believes it can enter. An aerial body would pass ``flies=True`` and ignore crop cells.

    Raises:
        ValueError: If the spec names a navigator that is not registered yet.
    """
    if spec.navigator != "potential-field":
        raise ValueError(f"unknown navigator: {spec.navigator}")
    return Navigator(memory, flies=False)


def build_experiment(
    label: str,
    scenario: ScenarioId,
    seed: int,
    brain: BrainSpec | None = None,
    until: Time = 3_600_000,
    metrics: MetricsConfig | None = None,
    dwell: Time = 5_000,
) -> ExperimentConfig:
    """Assemble one experiment from a scenario preset, for callers that do not build configs.

    Args:
        label: Name recorded on the resulting summary.
        scenario: Which built-in scenario preset to run.
        seed: The seed making the run reproducible.
        brain: How to assemble each robot's brain; the default spec when omitted.
        until: How far to run, in simulation time.
        metrics: Sampling cadence and target threshold.
        dwell: Stationary decision interval used by every robot.
    """
    return ExperimentConfig(
        label=label,
        scenario=build_scenario(scenario, seed=seed),
        scenario_name=scenario.name,
        brain=brain if brain is not None else BrainSpec(),
        until=until,
        metrics=metrics if metrics is not None else MetricsConfig(),
        dwell=dwell,
    )


def _mean(values: Iterable[float | None]) -> float | None:
    """Mean of the values that are present, or ``None`` if none of them are."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)
