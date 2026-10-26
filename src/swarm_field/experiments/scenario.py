"""A swarm experiment: a field, pests, settings shared by every robot, and the swarm itself.

A ``SwarmScenario`` is built rather than spelled out. It holds a ``MapConfig`` (the field), a
``PestConfig`` (the pests and their treatment), a ``RobotConfig`` (speeds, radio, decisions) and
the swarm: robots added preset by preset, each preset with a count::

    scenario = RIVER_SMALL.without_robots().add_robots(EXPLORER, 10).add_robots(HOP_KILLER, 4)

``into_experiment`` describes the run as an ``ExperimentConfig`` for ``swarm_field.runner``,
which assembles and runs it. Every robot is a ``ComposedAgent`` with a ``UtilityArbiter`` and the
brain of its preset (``swarm_field.experiments.presets``). Who broadcasts what follows from who
listens: the scenario merges what every preset needs relayed (``Relaying``), e.g. hop beacons
for hop killers and pest facts for seeded killers, and every preset builds its brain for that.

``WorldTick`` spawns pests on random free cells at ``PestConfig.pests_per_hour``, and grows and
spreads them. Robots share what they know only through the channel, within the radio range. A
mock source on every pest (``PestConfig.source_range_m``) can stand in for the explorer that
found it.

The ready scenarios are ``RIVER_SMALL``, ``RIVERS_LARGE`` and ``RELAYED_KILLERS``, the fields of
the pest-response benchmarks, ``EXPLORE``, explorers alone on an open field, and ``PESTS``,
explorers and seeded killers on the river field. ``pest_response_configurations`` turns a
scenario into the compared pest responses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from random import Random

from swarm_field.contracts import (
    BodyType,
    CellId,
    EnergySpec,
    MessageKind,
    PestDynamicsSpec,
    PestSourceSpec,
    RobotId,
    RobotSpec,
    ScenarioConfig,
    Time,
    TimeUnit,
)
from swarm_field.experiments.maps import (
    MapConfig,
    cropped_cells,
    river_and_rocks,
    two_rivers_and_rocks,
)
from swarm_field.experiments.presets import (
    EXPLORER,
    RELAY,
    SEEDED_KILLER,
    ExplorerPreset,
    HopKillerPreset,
    Relaying,
    RobotPreset,
    SeededKillerPreset,
    SwarmContext,
    hop_beacon_variants,
)
from swarm_field.runner import BrainSpec, ExperimentConfig
from swarm_field.simulation import MetricsConfig
from swarm_field.swarm import ArbiterConfig, KnowledgeSharing


@dataclass(frozen=True)
class PestConfig:
    """The pests of a run, how they are treated, and the mock sources that report them.

    Attributes:
        pests_per_hour: How many pests spawn per hour, on average, on the free healthy cells.
        period: The time between two pest updates (``WorldTick``).
        growth_rate: How much a pest's severity grows per pest update.
        spread_probability: The chance per pest update that a pest of severity 1 infests one
            neighbour; 0 keeps every pest where it spawned.
        spawn_severity: The severity of a new pest.
        treatment_rate: How much infestation severity a treating killer removes per hour. It
            must beat the growth, ``growth_rate`` per ``period``, or no pest ever clears.
        treatment_doses: How many treatments a killer carries.
        source_range_m: How far the mock source on every pest broadcasts it, once per decision
            interval, or ``None`` for no sources. ``math.inf`` tells every robot at once. A
            source sends what the swarm's killers read: the pest's fact for seeded killers, a
            beacon of 0 for hop killers.
    """

    pests_per_hour: float = 15.0
    period: Time = TimeUnit.MINUTE
    growth_rate: float = 0.05
    spread_probability: float = 0.0
    spawn_severity: float = 0.1
    treatment_rate: float = 1.0
    treatment_doses: int = 10
    source_range_m: float | None = None

    def dynamics(self, free_cells: int) -> PestDynamicsSpec:
        """``pests_per_hour`` as the chance per free cell and pest update that a pest spawns."""
        updates_per_hour = TimeUnit.HOUR / self.period
        return PestDynamicsSpec(
            period=self.period,
            spawn_probability=self.pests_per_hour / updates_per_hour / free_cells,
            growth_rate=self.growth_rate,
            spread_probability=self.spread_probability,
            spawn_severity=self.spawn_severity,
        )


@dataclass(frozen=True)
class RobotConfig:
    """What every robot of a run shares: its speed by body, its radio, and how it starts.

    Attributes:
        speed_mps_drone: The scouts' speed; a straight move takes ``cell_size / speed`` seconds.
        speed_mps_ground: The killers' speed.
        radio_range_m: How far a robot's broadcasts reach.
        decision_interval: The time between two decisions of a robot.
        knows_terrain: Whether every robot starts with the field's terrain map; a robot that
            keeps no map drops it.
        lattice_starts: Whether the robots of every preset start on a jittered lattice of their
            own over the whole field, instead of all side by side along the bottom row.
        lattice_jitter: How far a lattice start may move from its slot's centre, as a fraction
            of the lattice pitch.
        battery_capacity: How many standard charges every robot's battery holds; a standard
            charge lasts a body's endurance in travel, 30 minutes for a scout. A large capacity,
            e.g. ``1000``, keeps robots working without trips to a dock, and keeps hop killers,
            which cannot return to one yet, from running empty.
    """

    speed_mps_drone: float = 2.0
    speed_mps_ground: float = 0.5
    radio_range_m: float = 300.0
    decision_interval: Time = TimeUnit.SECOND * 15
    knows_terrain: bool = True
    lattice_starts: bool = False
    lattice_jitter: float = 0.3
    battery_capacity: float = 1.0

    def __post_init__(self) -> None:
        """Validate the supported parameter ranges."""
        if not 0.0 <= self.lattice_jitter < 0.5:
            raise ValueError("lattice jitter must be at least 0 and below half a pitch")
        if self.battery_capacity <= 0.0:
            raise ValueError("battery capacity must be positive")

    def speed_of(self, body: BodyType) -> float:
        """The top speed of a robot of ``body``: the drone speed for scouts, else ground."""
        if body is BodyType.SCOUT:
            return self.speed_mps_drone
        return self.speed_mps_ground


@dataclass(frozen=True)
class RobotGroup:
    """Robots of one preset, added together.

    Attributes:
        preset: Their kind, with its brain.
        count: How many there are.
    """

    preset: RobotPreset
    count: int


def _metrics_every_second() -> MetricsConfig:
    return MetricsConfig(period=TimeUnit.SECOND)


@dataclass(frozen=True)
class SwarmScenario:
    """The field, the pests and the swarm of one run.

    Build one from a ready scenario or from ``SwarmScenario()``, an empty swarm on the river
    field, with ``add_robots`` and the ``with_`` methods.

    Attributes:
        map: The field.
        pests: The pests and their treatment.
        robots: What every robot shares.
        swarm: The robots, a group per ``add_robots``, in the order of their ids.
        arbiter: The overrides and hysteresis of the presets' ``UtilityArbiter``.
        metrics: How ``MetricsCollector`` samples the run.
        until: How long the run lasts.
        seed: Seeds the pests, the lattice jitter and when each robot decides first.
    """

    map: MapConfig = field(default_factory=MapConfig)
    pests: PestConfig = field(default_factory=PestConfig)
    robots: RobotConfig = field(default_factory=RobotConfig)
    swarm: tuple[RobotGroup, ...] = ()
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    metrics: MetricsConfig = field(default_factory=_metrics_every_second)
    until: Time = 3 * TimeUnit.HOUR
    seed: int = 0

    def add_robots(self, preset: RobotPreset, count: int) -> SwarmScenario:
        """This scenario with ``count`` more robots of ``preset``, after the others.

        Raises:
            ValueError: If ``count`` is negative.
        """
        if count < 0:
            raise ValueError(f"cannot add {count} robots")
        return replace(self, swarm=(*self.swarm, RobotGroup(preset, count)))

    def without_robots(self, body: BodyType | None = None) -> SwarmScenario:
        """This scenario without its robots of ``body``, or without any robots."""
        kept: list[RobotGroup] = []
        for group in self.swarm:
            if body is not None and group.preset.body is not body:
                kept.append(group)
        return replace(self, swarm=tuple(kept))

    def with_map(self, map_config: MapConfig) -> SwarmScenario:
        return replace(self, map=map_config)

    def with_pests(self, pests: PestConfig) -> SwarmScenario:
        return replace(self, pests=pests)

    def with_robot_config(self, robots: RobotConfig) -> SwarmScenario:
        return replace(self, robots=robots)

    def with_metrics(self, metrics: MetricsConfig) -> SwarmScenario:
        return replace(self, metrics=metrics)

    def count(self, name: str | None = None) -> int:
        """How many robots of the preset called ``name`` there are, or how many in all."""
        found = 0
        for group in self.swarm:
            if name is None or group.preset.name == name:
                found += group.count
        return found

    @property
    def killers(self) -> int:
        """How many killers there are, of every preset."""
        found = 0
        for group in self.swarm:
            if group.preset.body is BodyType.KILLER:
                found += group.count
        return found

    def presets(self) -> list[RobotPreset]:
        """The preset of every robot, in the order of their ids."""
        found: list[RobotPreset] = []
        for group in self.swarm:
            found.extend([group.preset] * group.count)
        return found

    def relaying(self) -> Relaying:
        """What the swarm's robots relay: what all its presets need, merged."""
        hop_cells = self.robots.radio_range_m / self.map.cell_size
        merged = Relaying()
        for group in self.swarm:
            if group.count > 0:
                merged = merged.merge(group.preset.needs(hop_cells))
        return merged

    def sweep_time(self) -> Time:
        """How long the explorers need to visit every free cell once, if they split it evenly.

        A robot moves one cell per decision at most, so a step takes the move rounded up to
        whole decision intervals. The explorers are the robots of an ``ExplorerPreset``; without
        any, the sweep time is 0.
        """
        explorers = 0
        for group in self.swarm:
            if isinstance(group.preset, ExplorerPreset):
                explorers += group.count
        if explorers == 0:
            return 0
        interval = self.robots.decision_interval
        move = math.ceil(self.map.cell_size / self.robots.speed_mps_drone * TimeUnit.SECOND)
        step = math.ceil(move / interval) * interval
        return self.map.free_cells() * step // explorers

    def context(self) -> SwarmContext:
        """What this swarm tells each of its presets when the preset builds a brain."""
        return SwarmContext(self.relaying(), self.arbiter, self.sweep_time())

    def brain(self, preset: RobotPreset) -> BrainSpec:
        """The brain of a robot of ``preset`` in this swarm."""
        return preset.brain(self.context())

    def starts(self) -> list[CellId]:
        """The robots' starting cells, side by side from ``(0, 0)``, skipping obstacles.

        A swarm larger than the bottom row starts over at ``(0, 0)`` once the row is full, so
        robots share start cells, as they may share any cell, rather than start elsewhere on the
        field. With ``RobotConfig.lattice_starts`` they are on lattices instead
        (``_lattice_start_cells``).

        Raises:
            ValueError: If the bottom row has no free cell at all.
        """
        if self.robots.lattice_starts:
            return self._lattice_start_cells()
        blocked = set(self.map.blocked_cells())
        free = [(x, 0) for x in range(self.map.size) if (x, 0) not in blocked]
        if not free:
            raise ValueError("the bottom row has no free cell to start on")
        starts: list[CellId] = []
        for index in range(self.count()):
            starts.append(free[index % len(free)])
        return starts

    def _lattice_start_cells(self) -> list[CellId]:
        """The robots' starting cells: every group on a jittered lattice of its own.

        A lattice has as many columns as rows, or one more. Each robot starts in its slot's
        centre, moved by up to ``lattice_jitter`` of the pitch along each axis. The jitter is
        drawn from its own ``Random(seed)``, so the run's random streams are untouched. A start
        on an obstacle moves to the nearest free cell.
        """
        jitter = Random(self.seed)
        lattice: list[CellId] = []
        for group in self.swarm:
            lattice.extend(self._jittered_lattice(group.count, jitter))
        blocked = set(self.map.blocked_cells())
        starts: list[CellId] = []
        for cell in lattice:
            starts.append(self._nearest_free_cell(cell, blocked))
        return starts

    def _nearest_free_cell(self, cell: CellId, blocked: set[CellId]) -> CellId:
        """``cell`` if it is free, else the nearest free cell, searched ring by ring."""
        for radius in range(self.map.size):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    candidate = (cell[0] + dx, cell[1] + dy)
                    if self.map.is_inside(candidate) and candidate not in blocked:
                        return candidate
        raise ValueError("the field has no free cell")

    def _jittered_lattice(self, robots: int, jitter: Random) -> list[CellId]:
        """``robots`` cells on a lattice over the field, each moved at random within its slot."""
        if robots == 0:
            return []
        size = self.map.size
        spread = self.robots.lattice_jitter
        columns = math.ceil(math.sqrt(robots))
        rows = math.ceil(robots / columns)
        pitch_x = size / columns
        pitch_y = size / rows
        cells: list[CellId] = []
        for index in range(robots):
            row, column = divmod(index, columns)
            x = (column + 0.5 + jitter.uniform(-spread, spread)) * pitch_x
            y = (row + 0.5 + jitter.uniform(-spread, spread)) * pitch_y
            cells.append((min(int(x), size - 1), min(int(y), size - 1)))
        return cells

    def pest_sources(self) -> PestSourceSpec | None:
        """The mock pest sources: what the swarm's killers read, or ``None`` if none is used.

        Without sources, and in a swarm where no robot reads what they send, there are none.
        """
        if self.pests.source_range_m is None:
            return None
        relaying = self.relaying()
        kinds: set[MessageKind] = set()
        if relaying.pest_facts is not KnowledgeSharing.NOTHING:
            kinds.add(MessageKind.KNOWLEDGE)
        if relaying.beacons is not None:
            kinds.add(MessageKind.BEACON)
        if not kinds:
            return None
        return PestSourceSpec(
            period=self.robots.decision_interval,
            radio_range_m=self.pests.source_range_m,
            kinds=frozenset(kinds),
        )

    def into_experiment(self, label: str = "swarm") -> ExperimentConfig:
        """The run as an experiment for ``swarm_field.runner``.

        The first robot's brain is the default brain, and every robot with another brain gets
        its own.
        """
        presets = self.presets()
        robots: list[RobotSpec] = []
        for index, (preset, start) in enumerate(zip(presets, self.starts(), strict=True)):
            robots.append(
                RobotSpec(
                    robot_id=index + 1,
                    body=preset.body,
                    start=start,
                    v_max=self.robots.speed_of(preset.body),
                    battery=1.0,
                    radio_range=self.robots.radio_range_m,
                )
            )
        obstacles = self.map.blocked_cells()
        geometry = self.map.field_geometry()
        scenario = ScenarioConfig(
            field=geometry,
            crops=cropped_cells(geometry, obstacles),
            robots=robots,
            seed=self.seed,
            obstacles=obstacles,
            pest_dynamics=self.pests.dynamics(self.map.free_cells()),
            pest_sources=self.pest_sources(),
            treatment_rate=self.pests.treatment_rate,
            treatment_doses=self.pests.treatment_doses,
            energy=EnergySpec(capacity=self.robots.battery_capacity),
            docks=self.map.docks,
        )
        context = self.context()
        brains: list[BrainSpec] = []
        for group in self.swarm:
            # A group without robots has no brain to build, and like ``relaying`` it must not
            # count: a hop killer's brain refuses a swarm whose other robots relay no beacons.
            if group.count == 0:
                continue
            brain = group.preset.brain(context)
            brains.extend([brain] * group.count)
        default = BrainSpec()
        by_robot: dict[RobotId, BrainSpec] = {}
        if brains:
            default = brains[0]
            for robot, brain in zip(robots, brains, strict=True):
                if brain != default:
                    by_robot[robot.robot_id] = brain
        return ExperimentConfig(
            label=label,
            scenario=scenario,
            brain=default,
            brains_by_robot=by_robot,
            robots_know_terrain=self.robots.knows_terrain,
            stagger_decisions=True,
            until=self.until,
            metrics=self.metrics,
            decision_interval=self.robots.decision_interval,
        )


FAST_TREATMENT = 60.0
"""A treatment rate that clears a pest of full severity in one minute, far above its growth.

The ready pest scenarios treat this fast, so that a pest's time to clear is the swarm's response
rather than the treatment itself.
"""

RELAYED_KILLERS = (
    SwarmScenario(
        map=MapConfig(obstacles=()),
        pests=PestConfig(treatment_rate=FAST_TREATMENT, source_range_m=20.0),
        robots=RobotConfig(radio_range_m=20.0, knows_terrain=False, lattice_starts=True),
    )
    .add_robots(RELAY, 400)
    .add_robots(SEEDED_KILLER, 8)
)
"""The open-field pest-response benchmark (#70): 400 relays and 8 killers on lattices.

A mock source on every pest stands in for the explorer that found it. The radio reaches 1.25
relay pitches, so a pest reaches the nearest killer through several relays.
``pest_response_configurations`` turns it into the compared pest responses.
"""

RIVER_SMALL = (
    SwarmScenario(
        map=MapConfig(obstacles=river_and_rocks(32)),
        pests=PestConfig(pests_per_hour=18.0, treatment_rate=FAST_TREATMENT, source_range_m=50.0),
        robots=RobotConfig(radio_range_m=50.0, knows_terrain=False, lattice_starts=True),
    )
    .add_robots(RELAY, 64)
    .add_robots(SEEDED_KILLER, 6)
)
"""The obstacle comparison's small field (#70): one river with a bridge, and two rocks.

Relays stand on a lattice 4 cells apart, as sparse as it gets while the radio, which reaches
1.25 pitches, still joins almost all of them into one network and every cell is in range of a
relay. No robot knows the terrain; killers learn obstacles by sensing.
"""

RIVERS_LARGE = (
    RIVER_SMALL.without_robots()
    .with_map(MapConfig(size=64, obstacles=two_rivers_and_rocks(64)))
    .with_pests(replace(RIVER_SMALL.pests, pests_per_hour=48.0))
    .add_robots(RELAY, 256)
    .add_robots(SEEDED_KILLER, 18)
)
"""``RIVER_SMALL`` on four times the area: two rivers with three bridges, and eight rocks."""

EXPLORE = SwarmScenario(
    map=MapConfig(obstacles=()),
    pests=PestConfig(pests_per_hour=0.0),
    robots=RobotConfig(speed_mps_drone=1.0, decision_interval=2_500, radio_range_m=math.inf),
    metrics=MetricsConfig(period=TimeUnit.MINUTE),
).add_robots(EXPLORER, 5)
"""Five explorers alone on an open field, without pests, measured by coverage and idleness."""

PESTS = (
    SwarmScenario(pests=PestConfig(treatment_rate=FAST_TREATMENT))
    .add_robots(EXPLORER, 5)
    .add_robots(SEEDED_KILLER, 3)
)
"""Five explorers and three seeded killers on the river field, while pests keep appearing.

Killers treat with ``FAST_TREATMENT``: at the default rate of one severity per hour a pest grows
faster than it is treated and never clears.
"""

SMOKE = (
    replace(
        PESTS,
        metrics=MetricsConfig(period=TimeUnit.MINUTE, measure_from=30 * TimeUnit.MINUTE),
    )
    .without_robots()
    .add_robots(EXPLORER, 5)
    .add_robots(SEEDED_KILLER, 5)
)
"""The base demo and smoke test (#67): five explorers that only explore, five seeded killers
that only treat, so every robot has one objective and the arbiter never has to choose.

On ``PESTS``' field and pests, with its ground robots at 0.5 m/s and its 300 m radio, so a pest
across the field is minutes away and news of it reaches the killers. Sampled every minute, with
idleness measured after a warm-up of half an hour.
"""


def pest_response_configurations(
    scenario: SwarmScenario, seeded_no_yield: bool = False
) -> dict[str, SwarmScenario]:
    """The pest responses compared on ``scenario``, by name, each with its killers.

    Each is ``scenario`` with its killers, as many as it has of every preset, replaced by
    killers of one response, added after its other robots: ``seeded`` with claims, and the hop
    configurations of ``hop_beacon_variants`` (``hop``, ``hop-repel`` and ``hop-tuned``).

    Args:
        scenario: The scenario every configuration is a variant of.
        seeded_no_yield: Whether to add ``seeded-no-yield``: the seeded response without the
            claims, whose relays pass on only the pests, as before the claims existed.
    """
    killers = scenario.killers
    others = scenario.without_robots(BodyType.KILLER)
    configurations = {"seeded": others.add_robots(SEEDED_KILLER, killers)}
    if seeded_no_yield:
        no_claims = SeededKillerPreset(claims=False)
        configurations["seeded-no-yield"] = others.add_robots(no_claims, killers)
    hop_cells = scenario.robots.radio_range_m / scenario.map.cell_size
    for name, beacons in hop_beacon_variants(hop_cells).items():
        configurations[name] = others.add_robots(HopKillerPreset(beacons=beacons), killers)
    return configurations
