"""Scenario definition — the input port that constructs a runnable world (B.4).

The current pest model supports one pest kind. Pest dynamics are configured on the simulation
world event for now; scenario-level pest configuration can be added when multiple pest models
are introduced.
"""

from dataclasses import dataclass

from swarm_field.contracts.ids import BodyType, CellId, CropType, RobotId
from swarm_field.contracts.world import FieldGeometry


@dataclass
class RobotSpec:
    """Fixed parameters of one robot at the start of a run.

    Attributes:
        robot_id: The robot's identifier.
        body: Its body type.
        start: Starting cell.
        v_max: Top speed.
        battery: Starting / full energy.
        radio_range: Broadcast range.
    """

    robot_id: RobotId
    body: BodyType
    start: CellId
    v_max: float
    battery: float
    radio_range: float
    # further params (mass, sensor suite, ...) filled in by the implementers


@dataclass
class ScenarioConfig:
    """A complete, runnable scenario: the field, its crops, the robots and run settings.

    Attributes:
        field: The field geometry.
        crops: Crop growing in each cell; supports multiple species (S3).
        robots: The robots to place at the start.
        seed: RNG seed, for reproducible runs.
    """

    field: FieldGeometry
    crops: dict[CellId, CropType]
    robots: list[RobotSpec]
    seed: int
