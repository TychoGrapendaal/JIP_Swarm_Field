"""Scenario contracts shared by the simulation and future frontends."""

from dataclasses import dataclass, field
from enum import Enum, auto

from swarm_field.contracts.ids import BodyType, CellId, CropType, RobotId
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

    period: int = 60_000
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
    """

    robot_id: RobotId
    body: BodyType
    start: CellId
    v_max: float
    battery: float
    radio_range: float


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
    """

    field: FieldGeometry
    crops: dict[CellId, CropType]
    robots: list[RobotSpec]
    seed: int
    initial_infestation: dict[CellId, float] = field(default_factory=dict)
    obstacles: tuple[CellId, ...] = ()
    pest_dynamics: PestDynamicsSpec = PestDynamicsSpec()
