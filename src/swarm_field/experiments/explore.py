"""An exploration experiment: a swarm with one objective on an open or obstructed field.

``ExploreScenario.into_experiment`` describes the run as an ``ExperimentConfig`` for
``swarm_field.runner``, which assembles and runs it. Every robot is a scout with one behaviour,
the mock explore behaviour on the EXPLORE field (``ExploreSeeder``). There are no pests. Robots
start with the field's terrain map, so they plan around obstacles they have not seen yet, and
share what they know over the channel within ``radio_range_m``. The idleness and redundancy of
``MetricsCollector`` measure the swarm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from swarm_field.contracts import (
    BodyType,
    CellId,
    CropType,
    FieldGeometry,
    PestDynamicsSpec,
    RobotSpec,
    ScenarioConfig,
    Time,
    TimeUnit,
)
from swarm_field.runner import BrainSpec, ExperimentConfig
from swarm_field.simulation import MetricsConfig
from swarm_field.swarm import ExploreConfig


@dataclass(frozen=True)
class ExploreScenario:
    """The field, the swarm and the communication of one exploration run.

    Attributes:
        width: Field width in cells.
        height: Field height in cells.
        robots: How many robots start, side by side from cell ``(0, 0)`` along the bottom row.
        obstacles: The blocked cells.
        cell_size: The side of a cell in metres.
        speed_mps: The robots' speed; a straight move takes ``cell_size / speed_mps`` seconds.
        decision_interval: The time between two decisions of a robot.
        radio_range_m: How far a robot's broadcasts reach.
        explore: The parameters of the EXPLORE field.
        sample_period: The time between two metrics samples.
        fresh_age: A visit to a cell sensed less than this before is redundant.
        measure_from: Idleness is averaged from this time on.
        until: How long the run lasts.
        seed: Draws when each robot decides first, within one decision interval, so runs with
            different seeds interleave the robots' decisions differently.
    """

    width: int = 32
    height: int = 32
    robots: int = 5
    obstacles: tuple[CellId, ...] = ()
    cell_size: float = 10.0
    speed_mps: float = 1.0
    decision_interval: Time = 2_500
    radio_range_m: float = float("inf")
    explore: ExploreConfig = field(default_factory=ExploreConfig)
    sample_period: Time = TimeUnit.MINUTE
    fresh_age: Time = 30 * TimeUnit.MINUTE
    measure_from: Time = 0
    until: Time = 3 * TimeUnit.HOUR
    seed: int = 0

    def field_geometry(self) -> FieldGeometry:
        return FieldGeometry(
            width=self.width,
            height=self.height,
            cell_size=self.cell_size,
            row_sp=0.0,
            in_row_sp=0.0,
            headland=0.0,
        )

    def starts(self) -> list[CellId]:
        """The robots' starting cells, side by side from ``(0, 0)``, skipping obstacles."""
        free = [(x, 0) for x in range(self.width) if (x, 0) not in self.obstacles]
        if len(free) < self.robots:
            raise ValueError(f"the bottom row has room for {len(free)} robots, not {self.robots}")
        return free[: self.robots]

    def into_experiment(self, label: str = "explore") -> ExperimentConfig:
        """The run as an experiment for ``swarm_field.runner``."""
        robots: list[RobotSpec] = []
        for index, start in enumerate(self.starts()):
            robots.append(
                RobotSpec(
                    robot_id=index + 1,
                    body=BodyType.SCOUT,
                    start=start,
                    v_max=self.speed_mps,
                    battery=1.0,
                    radio_range=self.radio_range_m,
                )
            )
        scenario = ScenarioConfig(
            field=self.field_geometry(),
            crops=cropped_cells(self.field_geometry(), self.obstacles),
            robots=robots,
            seed=self.seed,
            obstacles=self.obstacles,
            pest_dynamics=PestDynamicsSpec(
                spawn_probability=0.0, growth_rate=0.0, spread_probability=0.0
            ),
            # This experiment measures exploration, so provide a dock at each robot's start.
            # The physical battery model can now recycle scouts instead of leaving them stranded.
            docks=tuple(self.starts()),
        )
        brain = BrainSpec(
            behaviors=("explore", "return-to-dock"),
            seeders=("explore", "dock"),
            params={"explore": self.explore},
        )
        return ExperimentConfig(
            label=label,
            scenario=scenario,
            brain=brain,
            robots_know_terrain=True,
            stagger_decisions=True,
            until=self.until,
            metrics=MetricsConfig(
                period=self.sample_period,
                fresh_age=self.fresh_age,
                measure_from=self.measure_from,
            ),
            decision_interval=self.decision_interval,
        )


def cropped_cells(field: FieldGeometry, obstacles: tuple[CellId, ...]) -> dict[CellId, CropType]:
    """A crop on every free cell, so that ``MetricsCollector`` measures the whole field."""
    blocked = set(obstacles)
    crops: dict[CellId, CropType] = {}
    for y in range(field.height):
        for x in range(field.width):
            if (x, y) not in blocked:
                crops[(x, y)] = CropType.MAIZE
    return crops


def river_and_rocks(size: int) -> tuple[CellId, ...]:
    """A river across the middle of the field with one bridge, and a few square rocks."""
    river_x = size // 2
    bridge_y = 3 * size // 4
    cells: list[CellId] = []
    for y in range(size):
        if abs(y - bridge_y) > 1:
            cells.extend([(river_x, y), (river_x + 1, y)])
    for corner_x, corner_y in ((size // 4, size // 3), (3 * size // 4, size // 5)):
        for dx in range(3):
            cells.extend((corner_x + dx, corner_y + dy) for dy in range(3))
    return tuple(cells)
