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
from dataclasses import dataclass, field, replace
from random import Random

from swarm_field.contracts import (
    BodyType,
    CellId,
    FieldGeometry,
    MessageKind,
    PestDynamicsSpec,
    PestSourceSpec,
    RobotSpec,
    ScenarioConfig,
    Time,
    TimeUnit,
)
from swarm_field.experiments.explore import cropped_cells, river_and_rocks, two_rivers_and_rocks
from swarm_field.runner import BrainSpec, ExperimentConfig
from swarm_field.simulation import MetricsConfig
from swarm_field.swarm import (
    ArbiterConfig,
    BeaconConfig,
    EscapeConfig,
    ExploreConfig,
    HopField,
    KnowledgeSharing,
    SharingConfig,
    TreatByBeacon,
    TreatInfestation,
    TreatSeeder,
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
        lattice_starts: Whether the scouts and the killers each start on a jittered lattice
            over the whole field instead of on the bottom row.
        lattice_jitter: How far a lattice start may move from its slot's centre, as a fraction
            of the lattice pitch.
        killers_explore: Whether killers also carry the explore behaviour.
        hop_treat: Whether killers respond to pests on hop beacons (``TreatByBeacon`` on the
            hop field) instead of on remembered pest cells; needs ``beacons``.
        scouts_relay: Whether scouts only relay: they carry no behaviour, so they stand still
            and only broadcast.
        cell_size: The side of a cell in metres.
        speed_mps_drone: The scouts' speed; a straight move takes ``cell_size / speed`` seconds.
        speed_mps_ground: The killers' speed.
        decision_interval: The time between two decisions of a robot.
        radio_range_m: How far a robot's broadcasts reach.
        pest_source_range_m: How far the mock source on every pest broadcasts it, once per
            ``decision_interval``, or ``None`` for no sources. ``math.inf`` tells every robot
            at once.
        sharing: Which memory facts every robot broadcasts to its peers.
        beacons: How every robot keeps, relays and weighs beacons, or ``None`` for robots
            without beacons.
        robots_know_terrain: Whether every robot starts with the field's terrain map.
        treatment_rate: How much infestation severity a treating killer removes per hour. It
            must beat the growth, ``growth_rate`` per ``pest_period``, or no pest ever clears.
        treatment_doses: How many treatments a killer carries.
        pests_per_hour: How many pests spawn per hour, on average, on the free healthy cells.
        pest_period: The time between two pest updates (``WorldTick``).
        growth_rate: How much a pest's severity grows per pest update.
        spread_probability: The chance per pest update that a pest of severity 1 infests one
            neighbour; 0 keeps every pest where it spawned.
        spawn_severity: The severity of a new pest.
        explore: The parameters of every robot's EXPLORE field.
        treat_window_blocks: How many blocks of the quadtree around a killer's own its TREAT
            field is searched cell by cell; beyond them, leaf by leaf.
        treat_yield: Whether a killer's TREAT field leaves out the pests a closer killer is
            heading for (``TreatSeeder.yield_to_closer``), and a killer on a pest another
            killer already treats waits (``TreatInfestation.yield_to_treating_peer``).
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
    hop_treat: bool = False
    scouts_relay: bool = False
    cell_size: float = 10.0
    speed_mps_drone: float = 2.0
    speed_mps_ground: float = 0.5
    decision_interval: Time = TimeUnit.SECOND * 15
    radio_range_m: float = 300.0
    pest_source_range_m: float | None = None
    sharing: KnowledgeSharing = KnowledgeSharing.EVERYTHING
    beacons: BeaconConfig | None = None
    robots_know_terrain: bool = True
    treatment_rate: float = 1.0
    treatment_doses: int = 10
    pests_per_hour: float = 15.0
    pest_period: Time = TimeUnit.MINUTE
    growth_rate: float = 0.05
    spread_probability: float = 0.0
    spawn_severity: float = 0.1
    explore: ExploreConfig = field(default_factory=ExploreConfig)
    treat_window_blocks: int = 2
    treat_yield: bool = False
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    sample_period: Time = TimeUnit.SECOND
    until: Time = 3 * TimeUnit.HOUR
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the supported parameter ranges."""
        if not 0.0 <= self.lattice_jitter < 0.5:
            raise ValueError("lattice jitter must be at least 0 and below half a pitch")

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

        With ``lattice_starts`` they are on lattices instead (``_lattice_start_cells``).
        """
        if self.lattice_starts:
            return self._lattice_start_cells()
        blocked = set(self.blocked_cells())
        free = [(x, 0) for x in range(self.size) if (x, 0) not in blocked]
        robots = self.scouts + self.killers
        if len(free) < robots:
            raise ValueError(f"the bottom row has room for {len(free)} robots, not {robots}")
        return free[:robots]

    def _lattice_start_cells(self) -> list[CellId]:
        """The robots' starting cells: the scouts on one jittered lattice, the killers on another.

        A lattice has as many columns as rows, or one more. Each robot starts in its slot's
        centre, moved by up to ``lattice_jitter`` of the pitch along each axis. The jitter is
        drawn from its own ``Random(seed)``, so the run's random streams are untouched. A start
        on an obstacle moves to the nearest free cell.
        """
        jitter = Random(self.seed)
        lattice = self._jittered_lattice(self.scouts, jitter)
        lattice.extend(self._jittered_lattice(self.killers, jitter))
        blocked = set(self.blocked_cells())
        starts: list[CellId] = []
        for cell in lattice:
            starts.append(self._nearest_free_cell(cell, blocked))
        return starts

    def _nearest_free_cell(self, cell: CellId, blocked: set[CellId]) -> CellId:
        """``cell`` if it is free, else the nearest free cell, searched ring by ring."""
        for radius in range(self.size):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    candidate = (cell[0] + dx, cell[1] + dy)
                    inside = 0 <= candidate[0] < self.size and 0 <= candidate[1] < self.size
                    if inside and candidate not in blocked:
                        return candidate
        raise ValueError("the field has no free cell")

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
        on hop beacons with ``hop_treat``, with the explore behaviour too when
        ``killers_explore``.
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
                period=self.decision_interval,
                radio_range_m=self.pest_source_range_m,
                kinds=self._pest_source_kinds(),
            )
        scenario = ScenarioConfig(
            field=self.field_geometry(),
            crops=cropped_cells(self.field_geometry(), obstacles),
            robots=robots,
            seed=self.seed,
            obstacles=obstacles,
            pest_dynamics=self.pest_dynamics(),
            pest_sources=pest_sources,
            treatment_rate=self.treatment_rate,
            treatment_doses=self.treatment_doses,
        )
        params: dict[str, object] = {"explore": self.explore}
        explore = self._brain(("explore",), ("explore",), params)
        if self.scouts_relay:
            explore = self._brain((), (), {})
        behavior, field = TreatInfestation.name, TreatSeeder.name
        if self.hop_treat:
            behavior, field = TreatByBeacon.name, HopField.name
        treat_params: dict[str, object] = {
            "treat_window_blocks": self.treat_window_blocks,
            "treat_yield": self.treat_yield,
        }
        treat = self._brain((behavior,), (field,), treat_params)
        if self.killers_explore:
            treat_params.update(params)
            treat = self._brain((behavior, "explore"), (field, "explore"), treat_params)
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

    def _brain(
        self, behaviors: tuple[str, ...], seeders: tuple[str, ...], params: dict[str, object]
    ) -> BrainSpec:
        """A brain with the scenario's arbiter, sharing and beacons, and a map if it needs one."""
        return BrainSpec(
            behaviors=behaviors,
            seeders=seeders,
            params=params,
            arbiter=self.arbiter,
            sharing=SharingConfig(facts=self.sharing),
            beacons=self.beacons,
            keeps_map=self._keeps_map(),
        )

    def _keeps_map(self) -> bool:
        """Whether the robots keep a map: all but those of the hop method alone.

        The seeded pest response and exploration plan over the map, and relays of the seeded
        response gossip the pests they know; with ``hop_treat`` and no exploration, no robot
        reads a map.
        """
        return not self.hop_treat or self.killers_explore

    def _pest_source_kinds(self) -> frozenset[MessageKind]:
        """What the mock sources send: pest facts for robots with a map, beacons for relays."""
        kinds: set[MessageKind] = set()
        if self._keeps_map():
            kinds.add(MessageKind.KNOWLEDGE)
        if self.beacons is not None:
            kinds.add(MessageKind.BEACON)
        return frozenset(kinds)


FAST_TREATMENT = 60.0
"""A treatment rate that clears a pest of full severity in one minute, far above its growth.

The #70 pest-response results were measured with treatment that cleared a pest at once, so their
scenarios treat this fast to stay comparable.
"""

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
    treatment_rate=FAST_TREATMENT,
)
"""The scenario of the open-field pest-response benchmark (#70): relays on a lattice, killers.

The scouts only relay, and a mock source on every pest stands in for the explorer that found
it. The radio reaches 1.25 relay pitches, so a pest reaches the nearest killer through several
relays. As it stands it is the seeded response without the killers' claims; every arm of the
benchmark is a variant of it, in ``tests/benchmark_pests.py``.
"""


RIVER_SMALL = PestScenario(
    obstacles=river_and_rocks(32),
    scouts=64,
    killers=6,
    lattice_starts=True,
    scouts_relay=True,
    killers_explore=False,
    radio_range_m=50.0,
    pest_source_range_m=50.0,
    sharing=KnowledgeSharing.INFESTATIONS,
    robots_know_terrain=False,
    pests_per_hour=18.0,
    treatment_rate=FAST_TREATMENT,
)
"""The obstacle comparison's small field (#70): one river with a bridge, and two rocks.

Relays stand on a lattice 4 cells apart, as sparse as it gets while the radio, which reaches
1.25 pitches, still joins almost all of them into one network and every cell is in range of a
relay. No robot knows the terrain; killers learn obstacles by sensing.
``pest_response_configurations`` turns it into the compared configurations.
"""

RIVERS_LARGE = replace(
    RIVER_SMALL,
    size=64,
    obstacles=two_rivers_and_rocks(64),
    scouts=256,
    killers=18,
    pests_per_hour=48.0,
)
"""``RIVER_SMALL`` on four times the area: two rivers with three bridges, and eight rocks."""


def pest_response_configurations(
    scenario: PestScenario, seeded_no_yield: bool = False
) -> dict[str, PestScenario]:
    """The four pest responses compared on ``scenario``, by name.

    ``seeded`` treats remembered pests on the TREAT field. Its killers leave a pest to a closer
    killer that claimed it, and do not join one another killer already treats (``treat_yield``);
    its relays gossip the pests and pass on the killers' claims
    (``KnowledgeSharing.INFESTATIONS_AND_TARGETS``). The hop configurations treat on hop
    beacons of one radio range each, capped where a beacon's strength falls below 0.05: ``hop``
    alone, ``hop-repel`` with killers repelling each other, and ``hop-tuned`` with pulls
    weighted by the square root of the distance and the escape.

    Args:
        scenario: The scenario every configuration is a variant of.
        seeded_no_yield: Whether to add ``seeded-no-yield``: the seeded response without the
            claims, whose relays gossip only the pests, as before the claims existed.
    """
    hop_beacons = BeaconConfig(
        hop_cells=scenario.radio_range_m / scenario.cell_size,
        max_value=math.log(0.05) / math.log(0.5),
    )
    repel = replace(hop_beacons, killer_repulsion=2.0)
    tuned = replace(repel, distance_power=0.5, escape=EscapeConfig())
    hop = replace(scenario, hop_treat=True, beacons=hop_beacons, sharing=KnowledgeSharing.NOTHING)
    configurations = {
        "seeded": replace(
            scenario, treat_yield=True, sharing=KnowledgeSharing.INFESTATIONS_AND_TARGETS
        ),
    }
    if seeded_no_yield:
        configurations["seeded-no-yield"] = replace(
            scenario, treat_yield=False, sharing=KnowledgeSharing.INFESTATIONS
        )
    configurations["hop"] = hop
    configurations["hop-repel"] = replace(hop, beacons=repel)
    configurations["hop-tuned"] = replace(hop, beacons=tuned)
    return configurations
