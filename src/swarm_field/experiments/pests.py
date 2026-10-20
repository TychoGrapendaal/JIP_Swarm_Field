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
comes in range. Robots start with the field's terrain map, so they plan around obstacles they
have not seen yet. ``MetricsCollector`` measures the run.

``RELAYED_KILLERS`` is a variant in which the scouts only relay: they stand still on a lattice,
and a few killers hear of pests through them from a mock source on every pest
(``PestSourceSpec``), which stands in for an explorer hovering above it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from random import Random

from swarm_field.contracts import (
    BodyType,
    CellId,
    FieldGeometry,
    PestDynamicsSpec,
    PestSourceSpec,
    RobotSpec,
    ScenarioConfig,
    Time,
    TimeUnit,
)
from swarm_field.experiments.explore import cropped_cells, river_and_rocks
from swarm_field.runner import BrainSpec, ExperimentConfig
from swarm_field.simulation import MetricsConfig
from swarm_field.swarm import ArbiterConfig, ExploreConfig, KnowledgeSharing


@dataclass(frozen=True)
class PestScenario:
    """The field, the swarm, the pests and the communication of one pest run.

    Attributes:
        size: The field's side in cells.
        obstacles: The blocked cells; by default the river with one bridge and two rocks of
            ``river_and_rocks``.
        scouts: How many scouts start, side by side from cell ``(0, 0)`` along the bottom row.
        killers: How many killers start, along the bottom row after the scouts.
        lattice_starts: Whether the scouts and the killers each start on a jittered lattice
            over the whole field instead of on the bottom row.
        lattice_jitter: How far a lattice start may move from its slot's centre, as a fraction
            of the lattice pitch.
        killers_explore: Whether killers also carry the explore behaviour.
        scouts_relay: Whether scouts only relay: they carry no behaviour, so they stand still
            and only broadcast.
        cell_size: The side of a cell in metres.
        speed_mps_drone: The scouts' speed; a straight move takes ``cell_size / speed`` seconds.
        speed_mps_ground: The killers' speed.
        decision_interval: The time between two decisions of a robot.
        radio_range_m: How far a robot's broadcasts reach.
        pest_source_range_m: How far the mock source on every pest broadcasts it, or ``None``
            for no sources. ``math.inf`` tells every robot at once.
        sharing: Which memory facts every robot broadcasts to its peers.
        robots_know_terrain: Whether every robot starts with the field's terrain map.
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
    lattice_starts: bool = False
    lattice_jitter: float = 0.3
    killers_explore: bool = True
    scouts_relay: bool = False
    cell_size: float = 10.0
    speed_mps_drone: float = 1.5
    speed_mps_ground: float = 0.1
    decision_interval: Time = TimeUnit.SECOND * 15
    radio_range_m: float = 300.0
    pest_source_range_m: float | None = None
    sharing: KnowledgeSharing = KnowledgeSharing.EVERYTHING
    robots_know_terrain: bool = True
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
        """The robots' starting cells, side by side from ``(0, 0)``, skipping obstacles.

        With ``lattice_starts`` they are the ``lattice_start_cells`` instead.
        """
        if self.lattice_starts:
            return self.lattice_start_cells()
        blocked = set(self.blocked_cells())
        free = [(x, 0) for x in range(self.size) if (x, 0) not in blocked]
        robots = self.scouts + self.killers
        if len(free) < robots:
            raise ValueError(f"the bottom row has room for {len(free)} robots, not {robots}")
        return free[:robots]

    def lattice_start_cells(self) -> list[CellId]:
        """The robots' starting cells: the scouts on one jittered lattice, the killers on another.

        A lattice has as many columns as rows, or one more. Each robot starts in its slot's
        centre, moved by up to ``lattice_jitter`` of the pitch along each axis. The jitter is
        drawn from its own ``Random(seed)``, so the run's random streams are untouched.
        """
        jitter = Random(self.seed)
        starts = self._jittered_lattice(self.scouts, jitter)
        starts.extend(self._jittered_lattice(self.killers, jitter))
        blocked = set(self.blocked_cells())
        for cell in starts:
            if cell in blocked:
                raise ValueError(f"the lattice start {cell} is an obstacle")
        return starts

    def _jittered_lattice(self, robots: int, jitter: Random) -> list[CellId]:
        """``robots`` cells on a lattice over the field, each moved at random within its slot."""
        if robots == 0:
            return []
        columns = math.ceil(math.sqrt(robots))
        rows = math.ceil(robots / columns)
        pitch_x = self.size / columns
        pitch_y = self.size / rows
        cells: list[CellId] = []
        for index in range(robots):
            row, column = divmod(index, columns)
            x = (column + 0.5 + jitter.uniform(-self.lattice_jitter, self.lattice_jitter)) * pitch_x
            y = (row + 0.5 + jitter.uniform(-self.lattice_jitter, self.lattice_jitter)) * pitch_y
            cells.append((min(int(x), self.size - 1), min(int(y), self.size - 1)))
        return cells

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

        Scouts get the explore brain, or none with ``scouts_relay``. Killers get the treat brain,
        with the explore behaviour too when ``killers_explore``.
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
        pest_sources = None
        if self.pest_source_range_m is not None:
            pest_sources = PestSourceSpec(
                period=self.decision_interval, radio_range_m=self.pest_source_range_m
            )
        scenario = ScenarioConfig(
            field=self.field_geometry(),
            crops=cropped_cells(self.field_geometry(), obstacles),
            robots=robots,
            seed=self.seed,
            obstacles=obstacles,
            pest_dynamics=self.pest_dynamics(),
            pest_sources=pest_sources,
        )
        params = {"explore": self.explore}
        explore = BrainSpec(
            behaviors=("explore",),
            seeders=("explore",),
            params=params,
            arbiter=self.arbiter,
            sharing=self.sharing,
        )
        treat = BrainSpec(
            behaviors=("treat-infestation",),
            seeders=("treat",),
            params={},
            arbiter=self.arbiter,
            sharing=self.sharing,
        )
        if self.scouts_relay:
            explore = BrainSpec(
                behaviors=(), seeders=(), params={}, arbiter=self.arbiter, sharing=self.sharing
            )
        if self.killers_explore:
            treat = BrainSpec(
                behaviors=("treat-infestation", "explore"),
                seeders=("treat", "explore"),
                params=params,
                arbiter=self.arbiter,
                sharing=self.sharing,
            )
        return ExperimentConfig(
            label=label,
            scenario=scenario,
            brain=explore,
            brains_by_body={BodyType.KILLER: treat},
            robots_know_terrain=self.robots_know_terrain,
            stagger_decisions=True,
            until=self.until,
            metrics=MetricsConfig(period=self.sample_period, fresh_age=self.explore.revisit_age),
            decision_interval=self.decision_interval,
        )


RELAYED_KILLERS = PestScenario(
    obstacles=(),
    scouts=400,
    killers=8,
    lattice_starts=True,
    scouts_relay=True,
    killers_explore=False,
    radio_range_m=20.0,
    pest_source_range_m=20.0,
    sharing=KnowledgeSharing.INFESTATIONS,
    robots_know_terrain=False,
)
"""The seeded arm of the pest-response benchmark (#70): relays on a lattice, a few killers.

The scouts only relay, and a mock source on every pest stands in for the explorer that found
it. The radio reaches 1.25 relay pitches, so a pest reaches the nearest killer through several
relays. The oracle arm is ``replace(RELAYED_KILLERS, pest_source_range_m=math.inf,
sharing=KnowledgeSharing.NOTHING)``: every killer hears every pest at once.
"""
