"""Ground-truth world model for the high-level swarm simulation."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import ceil, hypot
from typing import assert_never

from swarm_field.contracts import (
    Action,
    BodyType,
    CellId,
    CellState,
    Charge,
    CropObservation,
    EnergySpec,
    FailureMode,
    FieldGeometry,
    Idle,
    InfestationObservation,
    MetricsSample,
    Move,
    Observation,
    PeerObservation,
    RobotId,
    RobotPublicState,
    RobotSelfState,
    Terrain,
    TerrainObservation,
    Time,
    TimeUnit,
    Treat,
    WorldSnapshot,
)


@dataclass(frozen=True)
class EnergyConfig:
    """The battery model the world drains as robots act.

    Every term is a fraction of a full charge, matching the ``battery`` of a ``RobotState``.

    Attributes:
        per_metre: Battery fraction consumed per metre travelled.
        per_decision: Battery fraction consumed per decision point, for sensing, deciding and
            transmitting.
        treat_per_action: Battery fraction consumed by a ``Treat`` action.
    """

    per_metre: float = 0.0
    per_decision: float = 0.0
    treat_per_action: float = 0.0

    @classmethod
    def from_spec(cls, spec: EnergySpec) -> "EnergyConfig":
        """Build the world's battery model from the shared scenario contract."""
        return cls(
            per_metre=spec.per_metre,
            per_decision=spec.per_decision,
            treat_per_action=spec.treat_per_action,
        )


@dataclass
class RobotState:
    """Mutable ground-truth state for one simulated robot.

    Attributes:
        robot_id: Robot identifier.
        position: Current discrete field cell.
        body: Robot body class.
        speed_mps: Maximum travel speed in metres per second.
        radio_range_m: Broadcast radius in metres; robots within it hear this robot.
        battery: Remaining battery level, 0..1.
        consumable: Remaining treatment consumable, 0..1.
        health: Robot health, 0..1.
        move_target: Cell the robot is travelling toward, if any.
        move_started_at: Simulation time when the current move started.
        move_arrival_at: Simulation time when the current move completes.
        current_action: Action currently being executed or most recently applied.
        energy_spent: Battery fraction this robot has consumed since the run began, whatever
            recharged it in between.
        failed: Whether the robot has stopped taking part in the run.
    """

    robot_id: RobotId
    position: CellId
    body: BodyType
    speed_mps: float
    radio_range_m: float = 0.0
    battery: float = 1.0
    consumable: float = 1.0
    health: float = 1.0
    move_target: CellId | None = None
    move_started_at: Time | None = None
    move_arrival_at: Time | None = None
    current_action: Action | None = None
    energy_spent: float = 0.0
    failed: bool = False


class WorldState:
    """Ground truth for a coarse field-cell simulation.

    Cells represent patches of field, not individual plants. Obstacles are represented by
    marking one or more cells as blocked, so a river can occupy several adjacent cells.
    Robot occupancy is intentionally unrestricted: many robots may share one cell.
    """

    def __init__(
        self,
        field: FieldGeometry,
        robots: Iterable[RobotState] = (),
        energy: EnergyConfig | None = None,
    ) -> None:
        """Create the field and place any initial robots."""
        if field.width <= 0 or field.height <= 0:
            raise ValueError("field dimensions must be positive")
        if field.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if field.connectivity not in (4, 8):
            raise ValueError("field connectivity must be 4 or 8")

        self.field: FieldGeometry = field
        self.energy: EnergyConfig = energy if energy is not None else EnergyConfig()
        self.cells: dict[CellId, CellState] = {
            (x, y): CellState(
                crop=None,
                growth_stage=0.0,
                infestation=0.0,
                obstacle=False,
            )
            for y in range(field.height)
            for x in range(field.width)
        }
        self.robots: dict[RobotId, RobotState] = {}

        for robot in robots:
            self.add_robot(robot)

    def is_inside(self, cell: CellId) -> bool:
        """Return whether ``cell`` lies inside the field."""
        x, y = cell
        return 0 <= x < self.field.width and 0 <= y < self.field.height

    def cell(self, cell: CellId) -> CellState:
        """Return mutable ground truth for one cell."""
        if not self.is_inside(cell):
            raise ValueError(f"cell outside field: {cell}")
        return self.cells[cell]

    def neighbours(self, cell: CellId) -> list[CellId]:
        """Return valid candidate neighbours according to field connectivity."""
        if not self.is_inside(cell):
            raise ValueError(f"cell outside field: {cell}")

        offsets = ((0, 1), (1, 0), (0, -1), (-1, 0))
        if self.field.connectivity == 8:
            offsets = (
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            )

        x, y = cell
        result: list[CellId] = []
        for dx, dy in offsets:
            neighbour = (x + dx, y + dy)
            if self.is_inside(neighbour):
                result.append(neighbour)
        return result

    def is_walkable(self, cell: CellId) -> bool:
        """Return whether a target cell is physically traversable."""
        return self.is_inside(cell) and not self.cells[cell].obstacle

    def add_robot(self, robot: RobotState) -> None:
        """Add a robot without imposing any per-cell occupancy limit."""
        if robot.robot_id in self.robots:
            raise ValueError(f"robot already exists: {robot.robot_id}")
        if not self.is_inside(robot.position):
            raise ValueError(f"robot position outside field: {robot.position}")
        if not self.is_walkable(robot.position):
            raise ValueError(f"robot cannot occupy obstacle cell: {robot.position}")
        if robot.speed_mps <= 0:
            raise ValueError("robot speed_mps must be positive")
        self.robots[robot.robot_id] = robot

    def robots_at(self, cell: CellId) -> tuple[RobotId, ...]:
        """Return every robot currently occupying ``cell``."""
        if not self.is_inside(cell):
            raise ValueError(f"cell outside field: {cell}")
        return tuple(
            robot_id for robot_id, robot in sorted(self.robots.items()) if robot.position == cell
        )

    def positions(self) -> dict[RobotId, CellId]:
        """Return a copy of all ground-truth discrete robot positions."""
        return {robot_id: robot.position for robot_id, robot in sorted(self.robots.items())}

    def perceived_self(self, robot_id: RobotId, position: CellId | None = None) -> RobotSelfState:
        """Return the current self-perception for a robot.

        ``position`` overrides where the robot believes it is, which is how localisation error
        reaches the algorithm: the robot decides from the reported cell while the world keeps
        the true one. The reported cell is clamped into the field, so an error at the edge is a
        smaller error rather than a position outside the world.
        """
        robot = self._robot(robot_id)
        reported = robot.position if position is None else self._clamp_to_field(position)
        return RobotSelfState(
            robot_id=robot.robot_id,
            position=reported,
            battery=robot.battery,
            body=robot.body,
            speed_mps=robot.speed_mps,
            consumable=robot.consumable,
            health=robot.health,
        )

    def fail_robot(self, robot_id: RobotId, mode: FailureMode) -> None:
        """Stop one robot from taking part in the run, in the way ``mode`` describes.

        A robot that has already failed stays failed: a later check cannot bring it back, so a
        sweep's failure count is monotone and the swarm only ever loses capacity.
        """
        robot = self._robot(robot_id)
        if robot.failed:
            return
        robot.failed = True
        match mode:
            case FailureMode.DEAD_BATTERY:
                robot.battery = 0.0
            case FailureMode.DISABLED:
                robot.health = 0.0
            case FailureMode.OFFLINE:
                robot.radio_range_m = 0.0
            case _:
                assert_never(mode)

    def live_robots(self) -> tuple[RobotId, ...]:
        """Every robot still taking part in the run, in id order."""
        return tuple(
            robot_id for robot_id, robot in sorted(self.robots.items()) if not robot.failed
        )

    def observe_for(self, robot_id: RobotId, time: Time) -> list[Observation]:
        """Generate the observations of the one cell the robot stands on.

        A cell is a patch of field the robot monitors as a whole, so every fact kind (crop,
        infestation, terrain, peers sharing the cell) comes from the robot's own cell only.
        """
        robot = self._robot(robot_id)
        observations: list[Observation] = []

        cell = robot.position
        state = self.cells[cell]
        observations.append(
            CropObservation(
                cell=cell,
                confidence=1.0,
                observed_at=time,
                source=robot_id,
                crop=state.crop,
                growth_stage=state.growth_stage,
            )
        )
        observations.append(
            InfestationObservation(
                cell=cell,
                confidence=1.0,
                observed_at=time,
                source=robot_id,
                severity=state.infestation,
                pest=None,
            )
        )
        terrain = Terrain.FREE
        if state.obstacle:
            terrain = Terrain.OBSTACLE
        observations.append(
            TerrainObservation(
                cell=cell,
                confidence=1.0,
                observed_at=time,
                source=robot_id,
                terrain=terrain,
            )
        )

        for peer in self.robots_at(cell):
            if peer != robot_id:
                observations.append(
                    PeerObservation(
                        cell=cell,
                        confidence=1.0,
                        observed_at=time,
                        source=robot_id,
                        peer=peer,
                        body=self._robot(peer).body,
                    )
                )

        return observations

    def apply_action(self, robot_id: RobotId, action: Action, time: Time = 0) -> Time | None:
        """Apply an action at a time already advanced by the simulation event."""
        robot = self._robot(robot_id)
        robot.current_action = action
        self._spend(robot, self.energy.per_decision)

        if isinstance(action, Move):
            return self._start_move(robot, action.to, time)
        if isinstance(action, Treat):
            self._apply_treat(robot, action.cell)
        elif isinstance(action, Charge):
            robot.battery = 1.0
        elif isinstance(action, Idle):
            pass
        return None

    def energy_spent(self) -> float:
        """Total battery fraction every robot has consumed since the run began.

        Recharging does not subtract from this, so it measures the energy the swarm spent
        rather than the charge it happens to be holding.
        """
        return sum(robot.energy_spent for robot in self.robots.values())

    def advance_to(self, time: Time) -> None:
        """Advance any robot whose direct-cell movement has completed by ``time``."""
        for robot in self.robots.values():
            if robot.move_arrival_at is None or robot.move_target is None:
                continue
            if robot.move_arrival_at > time:
                continue

            robot.position = robot.move_target
            robot.move_target = None
            robot.move_started_at = None
            robot.move_arrival_at = None
            robot.current_action = Idle()

    def continuous_position(self, robot_id: RobotId, time: Time) -> tuple[float, float]:
        """Return the robot position in cell coordinates, interpolated during a move."""
        robot = self._robot(robot_id)
        if (
            robot.move_target is None
            or robot.move_started_at is None
            or robot.move_arrival_at is None
        ):
            return self._cell_center(robot.position)

        if time <= robot.move_started_at:
            return self._cell_center(robot.position)
        if time >= robot.move_arrival_at:
            return self._cell_center(robot.move_target)

        start_x, start_y = self._cell_center(robot.position)
        end_x, end_y = self._cell_center(robot.move_target)
        elapsed = time - robot.move_started_at
        duration = robot.move_arrival_at - robot.move_started_at
        fraction = elapsed / duration
        return (
            start_x + (end_x - start_x) * fraction,
            start_y + (end_y - start_y) * fraction,
        )

    def movement_duration(self, robot_id: RobotId, target: CellId) -> Time:
        """Return travel time in milliseconds for a direct move to ``target``."""
        robot = self._robot(robot_id)
        if not self.is_inside(target):
            raise ValueError(f"target outside field: {target}")
        if not self.is_walkable(target):
            raise ValueError(f"target is an obstacle: {target}")

        distance_m = self._distance_m(robot.position, target)
        return max(1, ceil(distance_m / robot.speed_mps * TimeUnit.SECOND))

    def public_states(self) -> list[RobotPublicState]:
        """Return ground-truth robot state for visualization."""
        return [
            RobotPublicState(
                robot_id=robot.robot_id,
                position=robot.position,
                body=robot.body,
                speed_mps=robot.speed_mps,
                battery=robot.battery,
                health=robot.health,
                target=robot.move_target,
                current_action=robot.current_action,
                failed=robot.failed,
            )
            for robot in sorted(self.robots.values(), key=lambda item: item.robot_id)
        ]

    def snapshot(self, time: Time, metrics: MetricsSample | None = None) -> WorldSnapshot:
        """Return a read-only-style snapshot of current ground truth.

        Args:
            time: The simulation time to advance to and report.
            metrics: The metrics sample taken at ``time``, passed through so a renderer can show
                the run's figures on the same frame. The world does not measure anything itself.
        """
        self.advance_to(time)
        cells = {cell: replace(state) for cell, state in self.cells.items()}
        return WorldSnapshot(
            time=time,
            field=self.field,
            cells=cells,
            robots=self.public_states(),
            metrics=metrics,
        )

    def _start_move(self, robot: RobotState, target: CellId, time: Time) -> Time | None:
        if robot.move_target is not None:
            return robot.move_arrival_at
        if target == robot.position:
            robot.current_action = Idle()
            return None
        if not self.is_walkable(target):
            robot.current_action = Idle()
            return None

        duration = self.movement_duration(robot.robot_id, target)
        self._spend(robot, self.energy.per_metre * self._distance_m(target, robot.position))
        robot.move_target = target
        robot.move_started_at = time
        robot.move_arrival_at = time + duration
        return robot.move_arrival_at

    def _apply_treat(self, robot: RobotState, target: CellId) -> None:
        if not self.is_inside(target):
            return
        if robot.consumable <= 0:
            return
        self._spend(robot, self.energy.treat_per_action)
        self.cells[target].infestation = 0.0
        robot.consumable = max(0.0, robot.consumable - 0.1)

    @staticmethod
    def _spend(robot: RobotState, amount: float) -> None:
        """Draw ``amount`` of battery, never below empty, and record it as spent."""
        if amount <= 0.0:
            return
        drawn = min(robot.battery, amount)
        robot.battery -= drawn
        robot.energy_spent += drawn

    def _distance_m(self, source: CellId, target: CellId) -> float:
        """Distance in metres between two cells, at the field's cell size."""
        return hypot(target[0] - source[0], target[1] - source[1]) * self.field.cell_size

    @staticmethod
    def _cell_center(cell: CellId) -> tuple[float, float]:
        return (cell[0] + 0.5, cell[1] + 0.5)

    def _robot(self, robot_id: RobotId) -> RobotState:
        try:
            return self.robots[robot_id]
        except KeyError as exc:
            raise ValueError(f"unknown robot: {robot_id}") from exc

    def _clamp_to_field(self, cell: CellId) -> CellId:
        """Pull a cell inside the field, so a reported position is always somewhere real."""
        return (
            min(max(cell[0], 0), self.field.width - 1),
            min(max(cell[1], 0), self.field.height - 1),
        )
