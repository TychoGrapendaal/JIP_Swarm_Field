"""A swarm experiment: a field, pests, settings shared by every robot, and the swarm itself.

A ``SwarmScenario`` is built rather than spelled out. It holds a ``MapConfig`` (the field), a
``PestConfig`` (the pests and their treatment), a ``RobotConfig`` (speeds, radio, decisions) and
the swarm: robots added preset by preset, each preset with a count. A scenario is composed from
ready parts, a map (``swarm_field.experiments.maps``), pests and robot settings::

    scenario = (
        SwarmScenario()
        .with_map(SMALL_RIVER_MAP)
        .with_pests(SMALL_FIELD_PESTS)
        .with_robot_config(DEFAULT_ROBOTS)
        .add_robots(EXPLORER, 10)
        .add_robots(HOP_BEST, 4)
    )

``into_experiment`` describes the run as an ``ExperimentConfig`` for ``swarm_field.runner``,
which assembles and runs it. Every robot is a ``ComposedAgent`` with a ``UtilityArbiter`` and the
brain of its preset (``swarm_field.experiments.presets``). Who broadcasts what follows from who
listens: the scenario merges what every preset needs relayed (``Relaying``), e.g. hop beacons
for hop killers and pest facts for seeded killers, and every preset builds its brain for that.

``WorldTick`` spawns pests on random free cells at ``PestConfig.pests_per_hour``, and grows and
spreads them. Robots share what they know only through the channel, within the radio range. A
mock source on every pest (``PestConfig.source_range_m``) can stand in for the explorer that
found it.

The experiments' defaults are the pests ``STATIC_PESTS``, ``SMALL_FIELD_PESTS`` and
``LARGE_FIELD_PESTS``, and the robot settings ``DEFAULT_ROBOTS``; the default scenarios
``RIVER_SMALL``, ``RIVERS_LARGE``, ``OPEN_SMALL`` and ``OPEN_LARGE`` put the default swarm on
each standard field. The earlier parts are the pests ``FAST_PESTS`` and ``NO_PESTS``, and the
robot settings ``LATTICE_ROBOTS`` and ``EXPLORE_ROBOTS``. The earlier scenarios are composed
from them: ``RELAYED_RIVER_SMALL``, ``RELAYED_RIVERS_LARGE`` and ``RELAYED_KILLERS``, the swarms
of the relay benchmarks, ``EXPLORE``, explorers alone on an open field, and ``PESTS``, explorers
and seeded killers on the river field. ``moving_explorer_scenario`` builds the scenario of the
earlier comparison experiments, moving explorers and killers without relays or mock sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from random import Random

from swarm_field.contracts import (
    BodyType,
    CellId,
    DockBeaconSpec,
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
    LARGE_OPEN_MAP,
    LARGE_RIVERS_MAP,
    SMALL_OPEN_MAP,
    SMALL_RIVER_MAP,
    MapConfig,
    cropped_cells,
)
from swarm_field.experiments.presets import (
    EXPLORER,
    RELAY,
    SEEDED_BEST,
    SEEDED_KILLER,
    ExplorerPreset,
    Relaying,
    RobotPreset,
    SwarmContext,
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
        initial_pests: How many pests are on the field when the run starts, each on a free cell
            of its own drawn from the run's seed, with ``spawn_severity``.
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
    initial_pests: int = 0
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
        decision_interval: The time between two decisions of a robot that stands still; a
            moving robot decides again when its move arrives.
        knows_terrain: Whether every robot starts with the field's terrain map; a robot that
            keeps no map drops it.
        lattice_starts: Whether the robots of every preset start on a jittered lattice of their
            own over the whole field, instead of all side by side along the bottom row.
        lattice_jitter: How far a lattice start may move from its slot's centre, as a fraction
            of the lattice pitch.
        battery_capacity: How many standard charges every robot's battery holds; a standard
            charge lasts a body's endurance in travel, 30 minutes for a scout. A large capacity,
            e.g. ``1000``, keeps robots working without trips to a dock.
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


@dataclass(frozen=True)
class SwarmScenario:
    """The field, the pests and the swarm of one run.

    Compose one from ``SwarmScenario()``, an empty swarm on ``MapConfig()``, the river field,
    with every setting at its default, with the ``with_`` methods and the ready parts, and
    ``add_robots``; or change a ready scenario the same way.

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
    metrics: MetricsConfig = field(default_factory=lambda: MetricsConfig(period=TimeUnit.SECOND))
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

    def with_pests_per_hour(self, pests_per_hour: float) -> SwarmScenario:
        return replace(self, pests=replace(self.pests, pests_per_hour=pests_per_hour))

    def with_hours(self, hours: int, warmup_hours: int) -> SwarmScenario:
        """This scenario run for ``hours`` and measured after ``warmup_hours``.

        Its metrics are sampled once a minute and averaged from the end of the warm-up on, which
        ``measure_pest_response`` also leaves out of ``infested``.

        Raises:
            ValueError: If the warm-up lasts as long as the run, or longer.
        """
        if warmup_hours >= hours:
            raise ValueError(f"a warm-up of {warmup_hours} h leaves nothing of a {hours} h run")
        metrics = replace(
            self.metrics, period=TimeUnit.MINUTE, measure_from=warmup_hours * TimeUnit.HOUR
        )
        return replace(self, until=hours * TimeUnit.HOUR, metrics=metrics)

    def with_seed(self, seed: int) -> SwarmScenario:
        return replace(self, seed=seed)

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
        sweep_time = self.sweep_time()
        merged = Relaying()
        for group in self.swarm:
            if group.count > 0:
                merged = merged.merge(group.preset.needs(hop_cells, sweep_time))
        return merged

    def run_cost(self) -> float:
        """A rough relative cost of a run: the field's cells times the hours, five times when
        relays gossip pests; to start the costliest runs of a batch first."""
        cost = self.map.size**2 * self.until / TimeUnit.HOUR
        if self.relaying().pest_facts is not KnowledgeSharing.NOTHING:
            cost *= 5
        return cost

    def sweep_time(self) -> Time:
        """How long the explorers need to visit every free cell once, if they split it evenly.

        A robot decides again as soon as a move arrives, so a step takes one straight move. The
        explorers are the robots of an ``ExplorerPreset``; without any, the sweep time is 0.
        """
        explorers = 0
        for group in self.swarm:
            if isinstance(group.preset, ExplorerPreset):
                explorers += group.count
        if explorers == 0:
            return 0
        step = math.ceil(self.map.cell_size / self.robots.speed_mps_drone * TimeUnit.SECOND)
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

    def initial_infestation(self) -> dict[CellId, float]:
        """The pests on the field at the start: ``PestConfig.initial_pests`` free cells.

        The cells are drawn from a ``Random`` of their own, seeded by the run's seed, so the
        run's random streams and the lattice starts are untouched.

        Raises:
            ValueError: If the field has fewer free cells than initial pests.
        """
        if self.pests.initial_pests == 0:
            return {}
        blocked = set(self.map.blocked_cells())
        free: list[CellId] = []
        for x in range(self.map.size):
            for y in range(self.map.size):
                if (x, y) not in blocked:
                    free.append((x, y))
        if self.pests.initial_pests > len(free):
            raise ValueError(f"{self.pests.initial_pests} initial pests, {len(free)} free cells")
        cells = Random(f"initial-pests/{self.seed}").sample(free, self.pests.initial_pests)
        infestation: dict[CellId, float] = {}
        for cell in cells:
            infestation[cell] = self.pests.spawn_severity
        return infestation

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

    def dock_beacons(self) -> DockBeaconSpec | None:
        """The docks' own radio: only when the swarm relays DOCK beacons and the field has docks.

        A dock broadcasts once per decision interval, as far as ``MapConfig.dock_range_m``, or
        the robots' radio range without one.
        """
        beacons = self.relaying().beacons
        if beacons is None or beacons.dock is None or not self.map.docks:
            return None
        reach = self.map.dock_range_m
        if reach is None:
            reach = self.robots.radio_range_m
        return DockBeaconSpec(period=self.robots.decision_interval, radio_range_m=reach)

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
            initial_infestation=self.initial_infestation(),
            obstacles=obstacles,
            pest_dynamics=self.pests.dynamics(self.map.free_cells()),
            pest_sources=self.pest_sources(),
            dock_beacons=self.dock_beacons(),
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

UNLIMITED_BATTERY_CAPACITY = 1000.0
"""Charges in every robot's battery: enough that no robot runs empty during a run."""

UNLIMITED_DOSES = 1000
"""Doses every killer carries: enough that no killer runs out during a run."""

DEFAULT_ROBOTS = RobotConfig(
    speed_mps_drone=8.0,
    speed_mps_ground=1.0,
    radio_range_m=100.0,
    knows_terrain=False,
    lattice_starts=True,
    battery_capacity=UNLIMITED_BATTERY_CAPACITY,
)
"""The robots of the experiments by default: explorers fly at 8 m/s, killers drive at 1 m/s,
the radio reaches 100 m, ten cells of 10 m, every preset starts on a jittered lattice of its own,
no robot knows the terrain, so robots learn obstacles by sensing, and no battery runs empty."""

STATIC_PESTS = PestConfig(
    growth_rate=0.0,
    spread_probability=0.0,
    treatment_rate=FAST_TREATMENT,
    treatment_doses=UNLIMITED_DOSES,
)
"""The pests of the experiments by default: static, they neither grow nor spread, a killer
treats one in a few seconds (``FAST_TREATMENT``) and never runs out of doses. The field's pest
parts (``SMALL_FIELD_PESTS``, ``LARGE_FIELD_PESTS``) set how many there are."""

SMALL_FIELD_PESTS = replace(STATIC_PESTS, pests_per_hour=20.0, initial_pests=20)
"""``STATIC_PESTS`` for a small field and the default swarm: 20 pests at the start and 20 per
hour. A starting point; an experiment sets the load its question needs."""

LARGE_FIELD_PESTS = replace(STATIC_PESTS, pests_per_hour=8.0, initial_pests=8)
"""``STATIC_PESTS`` for a large field and the default swarm: 8 pests at the start and 8 per
hour. A starting point; an experiment sets the load its question needs."""

DEFAULT_EXPLORERS = 10
"""Explorers in a swarm of the experiments, by default."""

DEFAULT_KILLERS = 4
"""Killers in a swarm of the experiments, by default."""

SMALL_FIELD_HOURS = 2
"""How long a run on a small field lasts, by default."""

LARGE_FIELD_HOURS = 4
"""How long a run on a large field lasts, by default."""

FAST_PESTS = PestConfig(treatment_rate=FAST_TREATMENT)
"""The pests of the earlier pest experiments: 15 per hour, growing and treated with
``FAST_TREATMENT``, 10 doses per killer, and no mock sources. At the default rate of one
severity per hour a pest grows faster than it is treated and never clears. A scenario with mock
sources sets ``source_range_m`` to its radio range."""

NO_PESTS = PestConfig(pests_per_hour=0.0)
"""No pests at all, for explorers measured by coverage and idleness alone."""

LATTICE_ROBOTS = RobotConfig(radio_range_m=50.0, knows_terrain=False, lattice_starts=True)
"""Robots of the earlier pest-response experiments: as ``DEFAULT_ROBOTS``, but scouts fly at
2 m/s and killers drive at 0.5 m/s, the radio reaches 50 m, and batteries hold one charge."""

EXPLORE_ROBOTS = RobotConfig(speed_mps_drone=1.0, decision_interval=2_500, radio_range_m=math.inf)
"""Robots of the earlier exploration experiments: scouts at 1 m/s that decide every 2.5 s while
they stand still, and hear each other across the whole field."""


def _default_scenario(field: MapConfig, pests: PestConfig, hours: int) -> SwarmScenario:
    """The experiments' default swarm on ``field``: ``DEFAULT_EXPLORERS`` explorers and
    ``DEFAULT_KILLERS`` killers of ``SEEDED_BEST``, with ``DEFAULT_ROBOTS``, for ``hours`` and
    measured from the start: the initial pests load the killers at once."""
    scenario = (
        SwarmScenario()
        .with_map(field)
        .with_pests(pests)
        .with_robot_config(DEFAULT_ROBOTS)
        .add_robots(EXPLORER, DEFAULT_EXPLORERS)
        .add_robots(SEEDED_BEST, DEFAULT_KILLERS)
    )
    return scenario.with_hours(hours, warmup_hours=0)


RIVER_SMALL = _default_scenario(SMALL_RIVER_MAP, SMALL_FIELD_PESTS, SMALL_FIELD_HOURS)
"""The default experiment on the small river field (``SMALL_RIVER_MAP``): 10 explorers and 4
seeded killers (``SEEDED_BEST``), ``DEFAULT_ROBOTS`` and ``SMALL_FIELD_PESTS``, for 2 hours.
The hop arm is the same scenario with its killers swapped for ``HOP_BEST``."""

RIVERS_LARGE = _default_scenario(LARGE_RIVERS_MAP, LARGE_FIELD_PESTS, LARGE_FIELD_HOURS)
"""The default experiment on the large river field (``LARGE_RIVERS_MAP``), as ``RIVER_SMALL``,
with ``LARGE_FIELD_PESTS``, for 4 hours."""

OPEN_SMALL = _default_scenario(SMALL_OPEN_MAP, SMALL_FIELD_PESTS, SMALL_FIELD_HOURS)
"""The default experiment on the small open field (``SMALL_OPEN_MAP``), as ``RIVER_SMALL``."""

OPEN_LARGE = _default_scenario(LARGE_OPEN_MAP, LARGE_FIELD_PESTS, LARGE_FIELD_HOURS)
"""The default experiment on the large open field (``LARGE_OPEN_MAP``), as ``RIVERS_LARGE``."""

RELAYED_KILLERS = (
    SwarmScenario()
    .with_map(SMALL_OPEN_MAP)
    .with_pests(replace(FAST_PESTS, source_range_m=20.0))
    .with_robot_config(replace(LATTICE_ROBOTS, radio_range_m=20.0))
    .add_robots(RELAY, 400)
    .add_robots(SEEDED_KILLER, 8)
)
"""The open-field pest-response benchmark (#70): 400 relays and 8 killers on lattices.

A mock source on every pest stands in for the explorer that found it. The radio reaches 1.25
relay pitches, so a pest reaches the nearest killer through several relays.
"""

RELAYED_RIVER_SMALL = (
    SwarmScenario()
    .with_map(SMALL_RIVER_MAP)
    .with_pests(replace(FAST_PESTS, pests_per_hour=18.0, source_range_m=50.0))
    .with_robot_config(LATTICE_ROBOTS)
    .add_robots(RELAY, 64)
    .add_robots(SEEDED_KILLER, 6)
)
"""The obstacle benchmark's small field (#70): standing relays and seeded killers on
``SMALL_RIVER_MAP``, with a mock source on every pest.

Relays stand on a lattice 4 cells apart, as sparse as it gets while the radio, which reaches
1.25 pitches, still joins almost all of them into one network and every cell is in range of a
relay. No robot knows the terrain; killers learn obstacles by sensing.
"""

RELAYED_RIVERS_LARGE = (
    SwarmScenario()
    .with_map(LARGE_RIVERS_MAP)
    .with_pests(replace(FAST_PESTS, pests_per_hour=48.0, source_range_m=50.0))
    .with_robot_config(LATTICE_ROBOTS)
    .add_robots(RELAY, 256)
    .add_robots(SEEDED_KILLER, 18)
)
"""``RELAYED_RIVER_SMALL`` on four times the area (``LARGE_RIVERS_MAP``)."""

EXPLORE = (
    SwarmScenario()
    .with_map(SMALL_OPEN_MAP)
    .with_pests(NO_PESTS)
    .with_robot_config(EXPLORE_ROBOTS)
    .with_metrics(MetricsConfig(period=TimeUnit.MINUTE))
    .add_robots(EXPLORER, 5)
)
"""Five explorers alone on an open field, without pests, measured by coverage and idleness."""

PESTS = (
    SwarmScenario()
    .with_map(MapConfig())
    .with_pests(FAST_PESTS)
    .add_robots(EXPLORER, 5)
    .add_robots(SEEDED_KILLER, 3)
)
"""Five explorers and three seeded killers on the river field with one dock, while pests keep
appearing, with every robot's settings at their defaults (``RobotConfig()``)."""

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


PESTS_PER_KILLER_HOUR = 3.0
"""The pests per killer and hour of the comparison experiments, a load found by experiment:
multiplied by the killers, the pest rate of a ``moving_explorer_scenario``."""


def moving_explorer_scenario(
    killer: RobotPreset,
    explorers: int,
    killers: int,
    radio_range_m: float,
    explorer: RobotPreset = EXPLORER,
    field: MapConfig = SMALL_RIVER_MAP,
) -> SwarmScenario:
    """The scenario of the comparison experiments: moving explorers and killers on ``field``.

    Explorers of one preset find the pests and killers of another treat them, with no relays and
    no mock sources. Every robot has the settings of ``LATTICE_ROBOTS`` but the radio, and
    battery and doses are unlimited (``UNLIMITED_BATTERY_CAPACITY``, ``UNLIMITED_DOSES``), so that
    a run measures the pest response alone. The pests are ``FAST_PESTS`` at
    ``PESTS_PER_KILLER_HOUR`` per killer.

    Args:
        killer: The killers' preset.
        explorers: How many explorers there are.
        killers: How many killers there are.
        radio_range_m: Every robot's radio range.
        explorer: The explorers' preset.
        field: The field: ``SMALL_RIVER_MAP``, the small field with one river, by default,
            or e.g. ``LARGE_RIVERS_MAP``, the large one with two rivers.
    """
    robots = replace(
        LATTICE_ROBOTS, radio_range_m=radio_range_m, battery_capacity=UNLIMITED_BATTERY_CAPACITY
    )
    pests = replace(
        FAST_PESTS,
        pests_per_hour=PESTS_PER_KILLER_HOUR * killers,
        treatment_doses=UNLIMITED_DOSES,
    )
    return (
        SwarmScenario()
        .with_map(field)
        .with_pests(pests)
        .with_robot_config(robots)
        .add_robots(explorer, explorers)
        .add_robots(killer, killers)
    )
