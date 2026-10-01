"""Ground-truth world model for the high-level swarm simulation."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import ceil, hypot

from swarm_field.contracts import (
    Action,
    BodyType,
    CellId,
    CellState,
    Charge,
    CropObservation,
    FieldGeometry,
    Idle,
    InfestationObservation,
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


class WorldState:
    """Ground truth for a coarse field-cell simulation.

    Cells represent patches of field, not individual plants. Obstacles are represented by
    marking one or more cells as blocked, so a river can occupy several adjacent cells.
    Robot occupancy is intentionally unrestricted: many robots may share one cell.
    """

    def __init__(self, field: FieldGeometry, robots: Iterable[RobotState] = ()) -> None:
        """Create the field and place any initial robots."""
        if field.width <= 0 or field.height <= 0:
            raise ValueError("field dimensions must be positive")
        if field.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if field.connectivity not in (4, 8):
            raise ValueError("field connectivity must be 4 or 8")

        self.field: FieldGeometry = field
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

    def perceived_self(self, robot_id: RobotId) -> RobotSelfState:
        """Return the current self-perception for a robot."""
        robot = self._robot(robot_id)
        return RobotSelfState(
            robot_id=robot.robot_id,
            position=robot.position,
            battery=robot.battery,
            body=robot.body,
            speed_mps=robot.speed_mps,
            consumable=robot.consumable,
            health=robot.health,
        )

    def observe_for(self, robot_id: RobotId, time: Time) -> list[Observation]:
        """Generate local observations from the robot's current cell and neighbours."""
        robot = self._robot(robot_id)
        self.advance_to(time)
        visible_cells = [robot.position, *self.neighbours(robot.position)]
        observations: list[Observation] = []

        for cell in visible_cells:
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
        """Apply a robot action and return a movement arrival time when applicable."""
        self.advance_to(time)
        robot = self._robot(robot_id)
        robot.current_action = action

        if isinstance(action, Move):
            return self._start_move(robot, action.to, time)
        if isinstance(action, Treat):
            self._apply_treat(robot, action.cell)
        elif isinstance(action, Charge):
            robot.battery = 1.0
        elif isinstance(action, Idle):
            pass
        return None

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

        distance_m = (
            hypot(target[0] - robot.position[0], target[1] - robot.position[1])
            * self.field.cell_size
        )
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
            )
            for robot in sorted(self.robots.values(), key=lambda item: item.robot_id)
        ]

    def snapshot(self, time: Time) -> WorldSnapshot:
        """Return a read-only-style snapshot of current ground truth."""
        self.advance_to(time)
        cells = {cell: replace(state) for cell, state in self.cells.items()}
        return WorldSnapshot(
            time=time,
            field=self.field,
            cells=cells,
            robots=self.public_states(),
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
        robot.move_target = target
        robot.move_started_at = time
        robot.move_arrival_at = time + duration
        return robot.move_arrival_at

    def _apply_treat(self, robot: RobotState, target: CellId) -> None:
        if not self.is_inside(target):
            return
        if robot.consumable <= 0:
            return
        self.cells[target].infestation = 0.0
        robot.consumable = max(0.0, robot.consumable - 0.1)

    @staticmethod
    def _cell_center(cell: CellId) -> tuple[float, float]:
        return (cell[0] + 0.5, cell[1] + 0.5)

    def _robot(self, robot_id: RobotId) -> RobotState:
        try:
            return self.robots[robot_id]
        except KeyError as exc:
            raise ValueError(f"unknown robot: {robot_id}") from exc
