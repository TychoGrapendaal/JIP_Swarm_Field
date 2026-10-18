"""An arbiter experiment: a mixed swarm whose robots weigh exploring, treating and recharging.

``ArbiterScenario`` is built so that every rule of ``UtilityArbiter`` has something to decide:

- scouts carry the explore behaviour and ``ReturnToDock``;
- killers carry ``TreatInfestation``, the explore behaviour and ``ReturnToDock``;
- pests keep spawning, so treating starts, finishes (rule 2) and moves on to the next pest
  after a treatment (rule 3), and treating the pest underfoot overrides (rule 1);
- batteries drain, so ``ReturnToDock`` ramps up against the other objectives (rules 4 and 5),
  overrides when critical (rule 1), and finishes once the robot has charged (rule 2).

``ArbiterLog.attach`` wraps every robot's arbiter in a ``RecordingArbiter``, which counts the
rule behind each choice without changing it; ``read_rule`` works the rule out from outside
``UtilityArbiter``, so the arbiter itself is untouched.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from math import isnan

from swarm_field.contracts import (
    BodyType,
    CellId,
    EnergySpec,
    FieldGeometry,
    PestDynamicsSpec,
    RobotId,
    RobotSpec,
    RoleSwitchSpec,
    ScenarioConfig,
    Time,
    TimeUnit,
)
from swarm_field.experiments.explore import cropped_cells, river_and_rocks
from swarm_field.experiments.run import LiveRun
from swarm_field.runner import BrainSpec, ExperimentConfig, dock_seeders
from swarm_field.simulation import MetricsConfig
from swarm_field.swarm import (
    ActionCandidate,
    ArbiterChoice,
    ArbiterConfig,
    Commitment,
    ExploreConfig,
    RebalanceConfig,
    RechargeConfig,
    UtilityArbiter,
)

logger = logging.getLogger(__name__)

EXPLORE_BEHAVIOURS: tuple[str, ...] = ("explore", "explore-utility")
"""The registry names of the explore behaviours a scenario can use: mock (0.5) or scored."""


@dataclass(frozen=True)
class ArbiterScenario:
    """The field, the swarm, the pests, the docks and the batteries of one arbiter run.

    Attributes:
        size: The field's side in cells.
        obstacles: The blocked cells; ``river_and_rocks`` when ``None``.
        scouts: How many scouts start along the bottom row.
        killers: How many killers start along the bottom row after the scouts.
        docks: The cells robots recharge on; one in each corner region when ``None``.
        explore_behaviour: The registry name of the explore behaviour, one of
            ``EXPLORE_BEHAVIOURS``.
        cell_size: The side of a cell in metres.
        speed_mps_scout: The scouts' speed.
        speed_mps_killer: The killers' speed.
        decision_interval: The time between two decisions of a robot.
        radio_range_m: How far a robot's broadcasts reach.
        pests_per_hour: How many pests spawn per hour, on average.
        growth_rate: How much a pest's severity grows per minute.
        spawn_severity: The severity of a new pest.
        energy: The battery drain.
        min_start_battery: Robots start with batteries spread evenly from this to full, so they
            do not all need a dock at once.
        recharge: When and how much robots want a dock.
        explore: The parameters of the EXPLORE field.
        arbiter: The overrides and hysteresis of every robot's ``UtilityArbiter``.
        sample_period: The time between two metrics samples.
        until: How long the run lasts.
        seed: Seeds the pests and when each robot decides first.
        rebalance: Lets robots switch between monitor and killer (#64) when given: every robot
            then carries both roles' behaviours, and treating is gated to killers.
        switch_duration: How long a role switch keeps a robot from acting.
    """

    size: int = 32
    obstacles: tuple[CellId, ...] | None = None
    scouts: int = 8
    killers: int = 6
    docks: tuple[CellId, ...] | None = None
    explore_behaviour: str = "explore"
    cell_size: float = 10.0
    speed_mps_scout: float = 1.5
    speed_mps_killer: float = 0.5
    decision_interval: Time = 10 * TimeUnit.SECOND
    radio_range_m: float = 300.0
    pests_per_hour: float = 20.0
    growth_rate: float = 0.05
    spawn_severity: float = 0.1
    energy: EnergySpec = field(
        default_factory=lambda: EnergySpec(
            per_metre=1 / 2000, per_decision=1e-4, treat_per_action=0.01
        )
    )
    min_start_battery: float = 0.5
    recharge: RechargeConfig = field(default_factory=RechargeConfig)
    explore: ExploreConfig = field(default_factory=ExploreConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    sample_period: Time = 10 * TimeUnit.SECOND
    until: Time = 3 * TimeUnit.HOUR
    seed: int = 0
    rebalance: RebalanceConfig | None = None
    switch_duration: Time = TimeUnit.MINUTE

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

    def dock_cells(self) -> tuple[CellId, ...]:
        """The docks, or one a tenth of the way in from each corner when none are given.

        Raises:
            ValueError: If a dock is outside the field or on an obstacle.
        """
        docks = self.docks
        if docks is None:
            near = max(1, self.size // 10)
            far = self.size - 1 - near
            docks = ((near, near), (far, near), (near, far), (far, far))
        blocked = set(self.blocked_cells())
        for dock in docks:
            inside = 0 <= dock[0] < self.size and 0 <= dock[1] < self.size
            if not inside or dock in blocked:
                raise ValueError(f"dock {dock} is outside the field or on an obstacle")
        return docks

    def bodies(self) -> list[BodyType]:
        """The body of every robot, in the order of their ids: scouts first, then killers."""
        return [BodyType.SCOUT] * self.scouts + [BodyType.KILLER] * self.killers

    def body_speed(self, body: BodyType) -> float:
        if body is BodyType.SCOUT:
            return self.speed_mps_scout
        return self.speed_mps_killer

    def starts(self) -> list[CellId]:
        """The robots' starting cells along the bottom rows, skipping obstacles."""
        blocked = set(self.blocked_cells())
        free: list[CellId] = []
        for y in range(2):
            free.extend((x, y) for x in range(self.size) if (x, y) not in blocked)
        robots = self.scouts + self.killers
        if len(free) < robots:
            raise ValueError(f"the bottom rows have room for {len(free)} robots, not {robots}")
        return free[:robots]

    def start_batteries(self) -> list[float]:
        """Each robot's starting battery, spread from ``min_start_battery`` to full.

        The levels are interleaved over the ids so scouts and killers both get high and low
        ones.
        """
        robots = self.scouts + self.killers
        levels: list[float] = []
        for index in range(robots):
            rank = (index * 7) % robots
            fraction = rank / max(1, robots - 1)
            levels.append(self.min_start_battery + (1.0 - self.min_start_battery) * fraction)
        return levels

    def pest_dynamics(self) -> PestDynamicsSpec:
        """``pests_per_hour`` as the chance per free cell and minute that a pest spawns."""
        free_cells = self.size * self.size - len(set(self.blocked_cells()))
        updates_per_hour = 60
        return PestDynamicsSpec(
            period=TimeUnit.MINUTE,
            spawn_probability=self.pests_per_hour / updates_per_hour / free_cells,
            growth_rate=self.growth_rate,
            spread_probability=0.0,
            spawn_severity=self.spawn_severity,
        )

    def scout_brain(self) -> BrainSpec:
        """Explore, and recharge when the battery runs low."""
        if self.explore_behaviour not in EXPLORE_BEHAVIOURS:
            raise ValueError(f"unknown explore behaviour: {self.explore_behaviour}")
        return BrainSpec(
            behaviors=(self.explore_behaviour, "return-to-dock"),
            seeders=("explore", *dock_seeders(self.dock_cells())),
            params=self._params(),
            arbiter=self.arbiter,
        )

    def killer_brain(self) -> BrainSpec:
        """Treat known pests, explore otherwise, and recharge when the battery runs low.

        The TREAT seeder comes first, since ``TreatInfestation`` follows the first seeder named.
        """
        return BrainSpec(
            behaviors=("treat-infestation", self.explore_behaviour, "return-to-dock"),
            seeders=("treat", "explore", *dock_seeders(self.dock_cells())),
            params=self._params(),
            arbiter=self.arbiter,
        )

    def rebalancing_brain(self, rebalance: RebalanceConfig) -> BrainSpec:
        """Every role's behaviours plus ``RebalanceRoles``; treating bids only for killers."""
        params = self._params()
        params["rebalance"] = rebalance
        params["body_gates"] = {"treat-infestation": (BodyType.KILLER,)}
        return BrainSpec(
            behaviors=(
                "treat-infestation",
                self.explore_behaviour,
                "return-to-dock",
                "rebalance-roles",
            ),
            seeders=("treat", "explore", *dock_seeders(self.dock_cells())),
            params=params,
            arbiter=self.arbiter,
        )

    def monitors_needed(self) -> int:
        """Monitors that keep every free cell fresher than the revisit age, at best.

        A monitor senses at most one new cell per decision, and no faster than it can drive
        across one, so this ignores overlap and is a lower bound.
        """
        free_cells = self.size * self.size - len(set(self.blocked_cells()))
        per_decision = self.decision_interval / TimeUnit.SECOND
        per_cell = self.cell_size / self.speed_mps_scout
        cells_per_minute = 60.0 / max(per_decision, per_cell)
        revisit_minutes = self.explore.revisit_age / TimeUnit.MINUTE
        return math.ceil(free_cells / (revisit_minutes * cells_per_minute))

    def into_experiment(self, label: str = "arbiter") -> ExperimentConfig:
        """The run as an experiment for ``swarm_field.runner``."""
        robots: list[RobotSpec] = []
        placed = zip(self.bodies(), self.starts(), self.start_batteries(), strict=True)
        for index, (body, start, battery) in enumerate(placed):
            robots.append(
                RobotSpec(
                    robot_id=index + 1,
                    body=body,
                    start=start,
                    v_max=self.body_speed(body),
                    battery=battery,
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
            energy=self.energy,
            roles=RoleSwitchSpec(
                speeds={
                    BodyType.SCOUT: self.speed_mps_scout,
                    BodyType.KILLER: self.speed_mps_killer,
                },
                duration=self.switch_duration,
            ),
        )
        brain = self.scout_brain()
        brains_by_body: dict[BodyType, BrainSpec] = {BodyType.KILLER: self.killer_brain()}
        if self.rebalance is not None:
            brain = self.rebalancing_brain(self.rebalance)
            brains_by_body = {}
        return ExperimentConfig(
            label=label,
            scenario=scenario,
            brain=brain,
            brains_by_body=brains_by_body,
            robots_know_terrain=True,
            stagger_decisions=True,
            until=self.until,
            metrics=MetricsConfig(period=self.sample_period, fresh_age=self.explore.revisit_age),
            decision_interval=self.decision_interval,
        )

    def _params(self) -> dict[str, object]:
        return {"explore": self.explore, "docks": self.dock_cells(), "recharge": self.recharge}


class ArbiterRule(Enum):
    """Which of ``UtilityArbiter``'s rules settled a choice, so a run can count them."""

    OVERRIDE = auto()
    """Rule 1: a bid at or above the override threshold won outright."""

    FIRST_COMMITMENT = auto()
    """The robot had no commitment, so the highest bid won."""

    OBJECTIVE_FINISHED = auto()
    """Rule 2: nothing bid for the committed objective any more, so the highest bid won."""

    UNCONTESTED = auto()
    """The committed bid was also the highest, so there was nothing to arbitrate."""

    DWELL_HOLD = auto()
    """Rule 4: a better rival was ignored because the objective was still inside ``min_dwell``."""

    MARGIN_HOLD = auto()
    """Rule 5: a better rival did not beat the commitment by the effort margin."""

    MARGIN_SWITCH = auto()
    """Rule 5: a rival beat the commitment by the effort margin, and the robot switched."""


@dataclass(frozen=True)
class RuleReading:
    """Which rule settled one choice, read from outside the arbiter.

    Attributes:
        rule: The rule that settled the choice.
        retargeted: Whether rule 3 fired first: the committed target was no longer bid for, so
            the robot moved on to its objective's best remaining target.
    """

    rule: ArbiterRule
    retargeted: bool = False


def read_rule(
    before: Commitment | None,
    candidates: Sequence[ActionCandidate],
    choice: ArbiterChoice,
    now: Time,
    config: ArbiterConfig,
) -> RuleReading:
    """Which of ``UtilityArbiter``'s rules produced ``choice``, from what it was given.

    ``before`` is the arbiter's commitment before the choice. The arbiter is left as it is: the
    rule follows from the bids, the commitment, the clock and ``choice.switched``, applied in
    the arbiter's own order.
    """
    bids: list[ActionCandidate] = []
    for candidate in candidates:
        if not isnan(candidate.score):
            bids.append(candidate)
    for bid in bids:
        if _clamped(bid.score) >= config.override_threshold:
            return RuleReading(ArbiterRule.OVERRIDE)
    if before is None:
        return RuleReading(ArbiterRule.FIRST_COMMITMENT)
    on_objective = [bid for bid in bids if bid.source == before.source]
    if not on_objective:
        return RuleReading(ArbiterRule.OBJECTIVE_FINISHED)
    retargeted = not any(before.is_for(bid) for bid in bids)
    if choice.switched:
        return RuleReading(ArbiterRule.MARGIN_SWITCH, retargeted)
    best = max(bids, key=lambda bid: _clamped(bid.score))
    if choice.candidate is best or choice.candidate == best:
        return RuleReading(ArbiterRule.UNCONTESTED, retargeted)
    if now - before.objective_since < config.min_dwell:
        return RuleReading(ArbiterRule.DWELL_HOLD, retargeted)
    return RuleReading(ArbiterRule.MARGIN_HOLD, retargeted)


def _clamped(score: float) -> float:
    """A score on the arbiter's ``[0, 1]`` scale, as ``UtilityArbiter`` clamps it."""
    return min(1.0, max(0.0, score))


@dataclass(frozen=True)
class ArbiterDecision:
    """One choice a robot's arbiter made, with the rule that settled it.

    Attributes:
        time: When the choice was made.
        robot: The robot that chose.
        rule: The rule that settled it.
        switched: Whether the robot abandoned a commitment still on offer.
        retargeted: Whether rule 3 moved the robot on to its objective's next target first.
        source: The winning behaviour.
        target: The winning bid's target, if it had one.
        score: The winning bid's score.
        rival: The best losing bid's behaviour and score, if there was one.
        previous: The behaviour that won this robot's previous decision, if any.
        action: The kind of the winning bid's action, e.g. ``FollowField`` or ``Charge``.
    """

    time: Time
    robot: RobotId
    rule: ArbiterRule
    switched: bool
    retargeted: bool
    source: str
    target: CellId | None
    score: float
    rival: tuple[str, float] | None
    previous: str | None = None
    action: str = ""


@dataclass
class ArbiterLog:
    """Every robot's arbiter decisions over a run, counted by rule.

    Attributes:
        rules: How often each rule settled a choice, over all robots.
        rules_by_robot: The same, per robot.
        retargets: How often rule 3 moved a robot on to its objective's next target.
        switches: How often a robot abandoned a commitment still on offer.
        idle: How often a robot had no bid at all.
        winners: How often each behaviour won.
        last: Each robot's last decision.
        notable: The most recent decisions other than an uncontested or first choice.
    """

    rules: Counter[ArbiterRule] = field(default_factory=Counter)
    rules_by_robot: dict[RobotId, Counter[ArbiterRule]] = field(default_factory=dict)
    retargets: int = 0
    switches: int = 0
    idle: int = 0
    winners: Counter[str] = field(default_factory=Counter)
    last: dict[RobotId, ArbiterDecision] = field(default_factory=dict)
    notable: deque[ArbiterDecision] = field(default_factory=lambda: deque(maxlen=8))

    def attach(self, run: LiveRun) -> None:
        """Wrap every robot's arbiter in a ``RecordingArbiter`` that reports to this log."""
        for robot_id, agent in run.agents().items():
            inner = agent.arbiter
            if not isinstance(inner, UtilityArbiter):
                logger.warning("Robot %s has a %s, not recorded", robot_id, type(inner).__name__)
                continue
            agent.arbiter = RecordingArbiter(inner, robot_id, self)

    def record(self, decision: ArbiterDecision) -> None:
        self.rules[decision.rule] += 1
        self.rules_by_robot.setdefault(decision.robot, Counter())[decision.rule] += 1
        self.winners[decision.source] += 1
        if decision.retargeted:
            self.retargets += 1
        if decision.switched:
            self.switches += 1
        self.last[decision.robot] = decision
        quiet = (ArbiterRule.UNCONTESTED, ArbiterRule.FIRST_COMMITMENT)
        if decision.rule not in quiet or decision.retargeted:
            self.notable.append(decision)

    def completed_jobs(self) -> int:
        """Targets and objectives finished: what ``switches_per_job`` divides by."""
        return self.retargets + self.rules[ArbiterRule.OBJECTIVE_FINISHED]

    def switches_per_job(self) -> float | None:
        """Abandoned commitments per completed job; above about 0.5 the margins are too low."""
        jobs = self.completed_jobs()
        if jobs == 0:
            return None
        return self.switches / jobs

    def decisions(self) -> int:
        return sum(self.rules.values()) + self.idle


class RecordingArbiter:
    """Delegates every choice to ``inner`` and records which rule settled it.

    Attributes:
        inner: The arbiter that actually chooses.
        robot: The robot it chooses for.
        log: Where the choices are recorded.
    """

    def __init__(self, inner: UtilityArbiter, robot: RobotId, log: ArbiterLog) -> None:
        self.inner: UtilityArbiter = inner
        self.robot: RobotId = robot
        self.log: ArbiterLog = log

    def choose(self, candidates: Sequence[ActionCandidate], now: Time) -> ArbiterChoice | None:
        before = self.inner.commitment
        choice = self.inner.choose(candidates, now)
        if choice is None:
            self.log.idle += 1
            return None
        reading = read_rule(before, candidates, choice, now, self.inner.config)
        winner = choice.candidate
        previous: str | None = None
        last = self.log.last.get(self.robot)
        if last is not None:
            previous = last.source
        rival: tuple[str, float] | None = None
        for candidate in candidates:
            if candidate is winner:
                continue
            if rival is None or candidate.score > rival[1]:
                rival = (candidate.source, candidate.score)
        self.log.record(
            ArbiterDecision(
                time=now,
                robot=self.robot,
                rule=reading.rule,
                switched=choice.switched,
                retargeted=reading.retargeted,
                source=winner.source,
                target=winner.target,
                score=winner.score,
                rival=rival,
                previous=previous,
                action=type(winner.action).__name__,
            )
        )
        return choice
