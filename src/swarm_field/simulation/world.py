"""Ground-truth grid and robot state for the simulation.

``WorldState`` is the owner of simulation truth. The algorithm should only receive data
through the ``Perception`` contract; visualization can consume ``WorldSnapshot`` through
``snapshot`` without reaching into the simulation internals.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace

from swarm_field.contracts import (
    Action,
    BodyType,
    CellId,
    CellState,
    Charge,
    CropObservation,
    Direction,
    FieldGeometry,
    Idle,
    InfestationObservation,
    Move,
    Observation,
    ObstacleObservation,
    PeerObservation,
    RobotId,
    RobotPublicState,
    RobotSelfState,
    Time,
    Treat,
    Turn,
    WorldSnapshot,
)


@dataclass
class RobotState:
    """Mutable ground-truth state for one simulated robot.

    Attributes:
        robot_id: The robot's identifier.
        position: Ground-truth grid cell occupied by the robot.
        body: Robot body type.
        facing: Ground-truth heading.
        battery: Remaining battery level, 0..1 for v1.
        consumable: Remaining treatment consumable, 0..1 for v1.
        health: Robot health, 0..1.
        current_action: Most recently applied physical action.
    """

    robot_id: RobotId
    position: CellId
    body: BodyType
    facing: Direction = Direction.N
    battery: float = 1.0
    consumable: float = 1.0
    health: float = 1.0
    current_action: Action | None = None


class WorldState:
    """Ground-truth grid, robot poses and basic action application for B.1.

    The world owns every cell in the configured rectangular grid. The initial implementation
    intentionally keeps sensing simple: a robot observes its current cell and the cells in the
    configured 4- or 8-neighbourhood, with perfect confidence. Sensor noise, field of view and
    richer robot-body sensing can be added later without changing the world/algorithm boundary.
    """

    def __init__(
        self,
        field: FieldGeometry,
        robots: Iterable[RobotState] = (),
    ) -> None:
        """Create a complete grid and optionally place initial robots."""
        if field.width <= 0 or field.height <= 0:
            raise ValueError("field dimensions must be positive")
        if field.connectivity not in (4, 8):
            raise ValueError("field connectivity must be 4 or 8")

        self.field: FieldGeometry = field
        self.cells: dict[CellId, CellState] = {
            (x, y): CellState(
                crop=None,
                growth_stage=0.0,
                infestation=0.0,
                obstacle=False,
                occupant=None,
            )
            for y in range(field.height)
            for x in range(field.width)
        }
        self.robots: dict[RobotId, RobotState] = {}

        for robot in robots:
            self.add_robot(robot)

    def is_inside(self, cell: CellId) -> bool:
        """Return whether ``cell`` lies inside the grid."""
        x, y = cell
        return 0 <= x < self.field.width and 0 <= y < self.field.height

    def cell(self, cell: CellId) -> CellState:
        """Return the mutable ground-truth state for one valid cell."""
        if not self.is_inside(cell):
            raise ValueError(f"cell outside field: {cell}")
        return self.cells[cell]

    def neighbours(self, cell: CellId) -> list[CellId]:
        """Return valid 4- or 8-connected neighbours in deterministic order."""
        if not self.is_inside(cell):
            raise ValueError(f"cell outside field: {cell}")

        directions = (
            Direction.N,
            Direction.E,
            Direction.S,
            Direction.W,
        )
        if self.field.connectivity == 8:
            directions = tuple(Direction)

        x, y = cell
        result: list[CellId] = []
        for direction in directions:
            dx, dy = direction.offset
            neighbour = (x + dx, y + dy)
            if self.is_inside(neighbour):
                result.append(neighbour)
        return result

    def is_walkable(self, cell: CellId) -> bool:
        """Return whether a robot may occupy ``cell``."""
        if not self.is_inside(cell):
            return False
        state = self.cells[cell]
        return not state.obstacle and state.occupant is None

    def add_robot(self, robot: RobotState) -> None:
        """Add a robot to the world and update cell occupancy."""
        if robot.robot_id in self.robots:
            raise ValueError(f"robot already exists: {robot.robot_id}")
        if not self.is_inside(robot.position):
            raise ValueError(f"robot position outside field: {robot.position}")
        if not self.is_walkable(robot.position):
            raise ValueError(f"robot cannot occupy cell: {robot.position}")

        self.robots[robot.robot_id] = robot
        self.cells[robot.position].occupant = robot.robot_id

    def positions(self) -> dict[RobotId, CellId]:
        """Return a copy of the ground-truth robot positions."""
        return {robot_id: robot.position for robot_id, robot in sorted(self.robots.items())}

    def perceived_self(self, robot_id: RobotId) -> RobotSelfState:
        """Return the robot's current self-perception.

        Position noise is intentionally absent in B.1, so the perceived position equals the
        ground-truth position for now.
        """
        robot = self._robot(robot_id)
        return RobotSelfState(
            robot_id=robot.robot_id,
            position=robot.position,
            facing=robot.facing,
            battery=robot.battery,
            body=robot.body,
            consumable=robot.consumable,
            health=robot.health,
        )

    def observe_for(self, robot_id: RobotId, time: Time) -> list[Observation]:
        """Build local observations visible to a robot at ``time``.

        V1 exposes the robot's current cell plus its configured grid neighbours. Every reading
        has confidence 1.0 because the current world contract has no sensor-noise parameters.
        """
        robot = self._robot(robot_id)
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
            observations.append(
                ObstacleObservation(
                    cell=cell,
                    confidence=1.0,
                    observed_at=time,
                    source=robot_id,
                    is_obstacle=state.obstacle,
                )
            )

            if state.occupant is not None and state.occupant != robot_id:
                observations.append(
                    PeerObservation(
                        cell=cell,
                        confidence=1.0,
                        observed_at=time,
                        source=robot_id,
                        peer=state.occupant,
                    )
                )

        return observations

    def apply_action(self, robot_id: RobotId, action: Action) -> None:
        """Apply one physical action to ground truth.

        ``Move`` advances one cell in the robot's current facing direction. An illegal move is
        ignored rather than allowing the robot to leave the field or overlap another robot.
        ``Treat`` removes infestation from the robot's current cell and ``Charge`` restores its
        battery. These are intentionally simple B.1 behaviours; detailed energy and treatment
        dynamics can be introduced by later simulation events.
        """
        robot = self._robot(robot_id)
        robot.current_action = action

        if isinstance(action, Move):
            self._apply_move(robot)
        elif isinstance(action, Turn):
            robot.facing = action.facing
        elif isinstance(action, Treat):
            self.cells[robot.position].infestation = 0.0
            robot.consumable = 0.0
        elif isinstance(action, Charge):
            robot.battery = 1.0
        elif isinstance(action, Idle):
            pass

    def public_states(self) -> list[RobotPublicState]:
        """Return ground-truth robot state for visualization."""
        return [
            RobotPublicState(
                robot_id=robot.robot_id,
                position=robot.position,
                facing=robot.facing,
                body=robot.body,
                battery=robot.battery,
                health=robot.health,
                current_action=robot.current_action,
            )
            for robot in sorted(self.robots.values(), key=lambda item: item.robot_id)
        ]

    def snapshot(self, time: Time) -> WorldSnapshot:
        """Return a snapshot containing independent cell values and public robot states."""
        cells = {cell: replace(state) for cell, state in self.cells.items()}
        return WorldSnapshot(
            time=time,
            field=self.field,
            cells=cells,
            robots=self.public_states(),
        )

    def _robot(self, robot_id: RobotId) -> RobotState:
        try:
            return self.robots[robot_id]
        except KeyError as exc:
            raise ValueError(f"unknown robot: {robot_id}") from exc

    def _apply_move(self, robot: RobotState) -> None:
        dx, dy = robot.facing.offset
        destination = (robot.position[0] + dx, robot.position[1] + dy)
        if destination not in self.neighbours(robot.position):
            return
        if not self.is_walkable(destination):
            return

        self.cells[robot.position].occupant = None
        robot.position = destination
        self.cells[destination].occupant = robot.robot_id
