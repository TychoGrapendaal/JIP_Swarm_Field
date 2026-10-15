"""A pest experiment: scouts explore, killers explore and treat, pests keep appearing.

``PestScenario.into_experiment`` describes the run as an ``ExperimentConfig`` for
``swarm_field.runner``, which assembles and runs it. Every robot is a ``ComposedAgent`` with a
``UtilityArbiter``. Scouts carry the mock explore behaviour on the EXPLORE field
(``ExploreSeeder``), which always bids 0.5. Killers carry ``TreatInfestation`` on the TREAT field
(``TreatSeeder``) and, with ``killers_explore``, the same explore behaviour, so their arbiter
weighs exploring against treating. Without it a killer waits until it hears of a pest.

``WorldTick`` spawns pests on random free cells at ``pests_per_hour``, and grows and spreads
them as ``PestDynamicsSpec`` says. Robots share their memory and status only through the
channel, within ``radio_range_m``, so a killer learns of a pest when a robot that knows of it
comes in range. Robots start with the field's terrain map, since they only sense the cell they
stand on and would never see an obstacle they cannot enter. ``MetricsCollector`` measures the
run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from swarm_field.contracts import (
    BodyType,
    CellId,
    FieldGeometry,
    PestDynamicsSpec,
    RobotSpec,
    ScenarioConfig,
    Time,
    TimeUnit,
)
from swarm_field.experiments.explore import cropped_cells, river_and_rocks
from swarm_field.runner import BrainSpec, ExperimentConfig
from swarm_field.simulation import MetricsConfig
from swarm_field.swarm import ArbiterConfig, ExploreConfig


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
        speed_mps_drone: The scouts' speed; a straight move takes ``cell_size / speed`` seconds.
        speed_mps_ground: The killers' speed.
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
        sample_period: The time between two metrics samples.
        until: How long the run lasts.
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
    until: Time = 3 * TimeUnit.HOUR
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
        if body is BodyType.SCOUT:
            return self.speed_mps_drone
        return self.speed_mps_ground

    def starts(self) -> list[CellId]:
        """The robots' starting cells, side by side from ``(0, 0)``, skipping obstacles."""
        blocked = set(self.blocked_cells())
        free = [(x, 0) for x in range(self.size) if (x, 0) not in blocked]
        robots = self.scouts + self.killers
        if len(free) < robots:
            raise ValueError(f"the bottom row has room for {len(free)} robots, not {robots}")
        return free[:robots]

    def pest_dynamics(self) -> PestDynamicsSpec:
        """``pests_per_hour`` as the chance per free cell and pest update that a pest spawns."""
        free_cells = self.size * self.size - len(set(self.blocked_cells()))
        updates_per_hour = TimeUnit.HOUR / self.pest_period
        return PestDynamicsSpec(
            period=self.pest_period,
            spawn_probability=self.pests_per_hour / updates_per_hour / free_cells,
            growth_rate=self.growth_rate,
            spread_probability=self.spread_probability,
            spawn_severity=self.spawn_severity,
        )

    def into_experiment(self, label: str = "pests") -> ExperimentConfig:
        """The run as an experiment for ``swarm_field.runner``.

        Scouts get the explore brain, killers the treat brain, with the explore behaviour too
        when ``killers_explore``.
        """
        robots: list[RobotSpec] = []
        for index, (body, start) in enumerate(zip(self.bodies(), self.starts(), strict=True)):
            robots.append(
                RobotSpec(
                    robot_id=index + 1,
                    body=body,
                    start=start,
                    v_max=self.body_speed(body),
                    battery=1.0,
                    radio_range=self.radio_range_m,
                )
            )
        obstacles = self.blocked_cells()
        scenario = ScenarioConfig(
            field=self.field_geometry(),
            crops=cropped_cells(self.field_geometry(), obstacles),
            robots=robots,
            seed=self.seed,
            obstacles=obstacles,
            pest_dynamics=self.pest_dynamics(),
        )
        params = {"explore": self.explore}
        explore = BrainSpec(
            behaviors=("explore",), seeders=("explore",), params=params, arbiter=self.arbiter
        )
        treat = BrainSpec(
            behaviors=("treat-infestation",), seeders=("treat",), params={}, arbiter=self.arbiter
        )
        if self.killers_explore:
            treat = BrainSpec(
                behaviors=("treat-infestation", "explore"),
                seeders=("treat", "explore"),
                params=params,
                arbiter=self.arbiter,
            )
        return ExperimentConfig(
            label=label,
            scenario=scenario,
            brain=explore,
            brains_by_body={BodyType.KILLER: treat},
            robots_know_terrain=True,
            stagger_decisions=True,
            until=self.until,
            metrics=MetricsConfig(period=self.sample_period, fresh_age=self.explore.revisit_age),
            decision_interval=self.decision_interval,
        )
