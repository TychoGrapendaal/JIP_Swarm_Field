"""Scenario contracts shared by the simulation and future frontends."""

from dataclasses import dataclass, field
from enum import Enum, auto

from swarm_field.contracts.ids import BodyType, CellId, CropType, RobotId, TimeUnit
from swarm_field.contracts.world import FieldGeometry


class ScenarioId(Enum):
    """Named scenario presets available from the scenario builder."""

    S1 = auto()
    S2 = auto()
    S3 = auto()
    S4 = auto()
    S5 = auto()


@dataclass(frozen=True)
class PestDynamicsSpec:
    """Configuration for the coarse-grained pest population model.

    Attributes:
        period: Time between world ticks in simulation milliseconds.
        spawn_probability: Probability of a new infestation on each healthy cell per tick.
        growth_rate: Additive infestation growth per tick.
        spread_probability: Maximum spread probability from one infested cell per neighbour.
        spawn_severity: Initial severity assigned to new infestations.
    """

    period: int = TimeUnit.MINUTE
    spawn_probability: float = 0.01
    growth_rate: float = 0.05
    spread_probability: float = 0.20
    spawn_severity: float = 0.10

    def __post_init__(self) -> None:
        """Validate the supported parameter ranges."""
        if self.period <= 0:
            raise ValueError("pest dynamics period must be positive")
        for name, value in (
            ("spawn_probability", self.spawn_probability),
            ("growth_rate", self.growth_rate),
            ("spread_probability", self.spread_probability),
            ("spawn_severity", self.spawn_severity),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class EnergySpec:
    """Configuration for the coarse-grained battery model.

    A robot's battery is stored as a fraction of its full charge, so every term below is
    expressed in that fraction rather than in joules. Travel is charged by the distance
    actually travelled, and every decision point costs a fixed amount for sensing, deciding
    and transmitting.

    Attributes:
        per_metre: Battery fraction consumed per metre travelled.
        per_decision: Battery fraction consumed per decision point.
        treat_per_action: Battery fraction consumed by a ``Treat`` action.
    """

    per_metre: float = 0.0
    per_decision: float = 0.0
    treat_per_action: float = 0.0

    def __post_init__(self) -> None:
        """Validate that no term is negative."""
        for name, value in (
            ("per_metre", self.per_metre),
            ("per_decision", self.per_decision),
            ("treat_per_action", self.treat_per_action),
        ):
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")


class FailureMode(Enum):
    """What stops a robot from taking part in a run.

    A failed robot is out for good: it keeps its last position and stops deciding, so the rest
    of the swarm carries on without it. The modes differ in what the robot still does for the
    others, which is what makes them worth sweeping separately.
    """

    DEAD_BATTERY = auto()
    """Battery exhausted. The robot stops, and is dark to the rest of the swarm."""

    DISABLED = auto()
    """Actuator or motor failure. The robot stops moving but stays observable."""

    OFFLINE = auto()
    """Radio failure. The robot still decides and moves, but nobody hears it."""


@dataclass(frozen=True)
class FailureSpec:
    """Recurring chance that a robot fails, for testing how the swarm degrades.

    The draw is per robot per check, from the simulation's own fault-injection randomness
    rather than the world's, so switching failures on does not change the pest dynamics a run
    produces. A run with ``probability`` at zero never fails a robot.

    Attributes:
        period: Time between failure checks in simulation milliseconds.
        probability: Chance that a given live robot fails on one check.
        mode: What the failure is when it happens.
    """

    period: int = TimeUnit.MINUTE
    probability: float = 0.0
    mode: FailureMode = FailureMode.DEAD_BATTERY

    def __post_init__(self) -> None:
        """Validate the supported parameter ranges."""
        if self.period <= 0:
            raise ValueError("failure period must be positive")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("failure probability must be between 0 and 1")


@dataclass(frozen=True)
class PositionErrorSpec:
    """Recurring chance that a robot mislocates itself, for testing recovery from drift.

    When a robot draws localisation error its perceived position is displaced by up to
    ``max_offset`` cells instead of its true one, so it decides as if it were somewhere else
    until the next clean reading. The displacement is clamped to the field, and the error is
    redrawn every decision, so a robot does not stay lost for a fixed number of decisions.

    Attributes:
        probability: Chance that a robot mislocates itself on one decision.
        max_offset: Largest displacement in cells, in each direction.
    """

    probability: float = 0.0
    max_offset: int = 1

    def __post_init__(self) -> None:
        """Validate the supported parameter ranges."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("position error probability must be between 0 and 1")
        if self.max_offset < 0:
            raise ValueError("position error offset cannot be negative")


@dataclass
class RobotSpec:
    """Fixed parameters of one robot at the start of a run.

    Attributes:
        robot_id: The robot's identifier.
        body: Its body type.
        start: Starting cell.
        v_max: Top speed in metres per second.
        battery: Starting / full energy, from 0 to 1.
        radio_range: Broadcast range in metres.
        spare_batteries: Charged replacement batteries carried at the start, if applicable.
    """

    robot_id: RobotId
    body: BodyType
    start: CellId
    v_max: float
    battery: float
    radio_range: float
    spare_batteries: int = 0


@dataclass
class ScenarioConfig:
    """A complete, runnable scenario definition.

    Attributes:
        field: The field geometry.
        crops: Crop growing in each defined cell; cells not present remain bare soil.
        robots: The robots to place at the start.
        seed: RNG seed for reproducible runs.
        initial_infestation: Initial true infestation severity by cell.
        obstacles: Cells that are blocked at time zero.
        pest_dynamics: Parameters for the recurring pest dynamics event.
        docks: Field cells where robots can recharge; battery robots replenish their spare stock.
        energy: Parameters for the battery model the simulation drains.
        failures: Recurring chance that a robot fails, for robustness runs.
        position_error: Recurring chance that a robot mislocates itself, for robustness runs.
    """

    field: FieldGeometry
    crops: dict[CellId, CropType]
    robots: list[RobotSpec]
    seed: int
    initial_infestation: dict[CellId, float] = field(default_factory=dict)
    obstacles: tuple[CellId, ...] = ()
    pest_dynamics: PestDynamicsSpec = PestDynamicsSpec()
    docks: tuple[CellId, ...] = ()
    energy: EnergySpec = EnergySpec()
    failures: FailureSpec = FailureSpec()
    position_error: PositionErrorSpec = PositionErrorSpec()
