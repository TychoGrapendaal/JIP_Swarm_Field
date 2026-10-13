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
    ReplaceBattery,
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
    spare_batteries: int = 0
    treating_cell: CellId | None = None
    treatment_updated_at: Time | None = None
    charging_updated_at: Time | None = None
    replacing_battery_for: RobotId | None = None
    replacement_complete_at: Time | None = None


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
        docks: Iterable[CellId] = (),
    ) -> None:
        """Create the field, fixed charging docks, and any initial robots."""
        if field.width <= 0 or field.height <= 0:
            raise ValueError("field dimensions must be positive")
        if field.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if field.connectivity not in (4, 8):
            raise ValueError("field connectivity must be 4 or 8")

        self.field: FieldGeometry = field
        self.docks: frozenset[CellId] = frozenset(docks)
        if any(not self.is_inside(cell) for cell in self.docks):
            raise ValueError("dock outside field")
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
        """Return whether a ground robot can enter a target cell."""
        return self.is_inside(cell) and not self.cells[cell].obstacle

    def is_traversable_for(self, body: BodyType, cell: CellId) -> bool:
        """Return whether a body can enter a cell (scouts fly over ground obstacles)."""
        if not self.is_inside(cell):
            return False
        return body is BodyType.SCOUT or not self.cells[cell].obstacle

    def add_robot(self, robot: RobotState) -> None:
        """Add a robot without imposing any per-cell occupancy limit."""
        if robot.robot_id in self.robots:
            raise ValueError(f"robot already exists: {robot.robot_id}")
        if not self.is_inside(robot.position):
            raise ValueError(f"robot position outside field: {robot.position}")
        if not self.is_traversable_for(robot.body, robot.position):
            raise ValueError(f"robot cannot occupy this cell: {robot.position}")
        if robot.spare_batteries < 0:
            raise ValueError("spare_batteries cannot be negative")
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
            spare_batteries=robot.spare_batteries,
        )

    def observe_for(self, robot_id: RobotId, time: Time) -> list[Observation]:
        """Generate body-specific sensing, local terrain data and nearby peer status.

        Scouts sense crop and infestation over a 3 x 3 footprint. Killers sense crop and
        infestation in their own cell. Battery robots do not sense infestation. All bodies get
        a one-cell terrain ring for local obstacle avoidance, and peer status within one cell.
        """
        from swarm_field.swarm.capabilities import capabilities_for

        robot = self._robot(robot_id)
        profile = capabilities_for(robot.body)
        observations: list[Observation] = []
        pest_cells = self._cells_in_square(robot.position, profile.pest_sensing_radius)
        terrain_cells = self._cells_in_square(robot.position, profile.terrain_sensing_radius)
        peer_cells = self._cells_in_square(robot.position, profile.peer_sensing_radius)

        for cell in pest_cells:
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
            if profile.can_detect_pests:
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

        for cell in terrain_cells:
            state = self.cells[cell]
            terrain = Terrain.OBSTACLE if state.obstacle else Terrain.FREE
            observations.append(
                TerrainObservation(
                    cell=cell,
                    confidence=1.0,
                    observed_at=time,
                    source=robot_id,
                    terrain=terrain,
                )
            )

        for cell in peer_cells:
            for peer in self.robots_at(cell):
                if peer == robot_id:
                    continue
                peer_state = self._robot(peer)
                observations.append(
                    PeerObservation(
                        cell=cell,
                        confidence=1.0,
                        observed_at=time,
                        source=robot_id,
                        peer=peer,
                        body=peer_state.body,
                        battery=peer_state.battery,
                    )
                )

        return observations

    def _cells_in_square(self, centre: CellId, radius: int) -> list[CellId]:
        """Cells in a bounded square sensing footprint, independent of field connectivity."""
        x, y = centre
        return [
            (cx, cy)
            for cx in range(x - radius, x + radius + 1)
            for cy in range(y - radius, y + radius + 1)
            if self.is_inside((cx, cy))
        ]

    def apply_action(self, robot_id: RobotId, action: Action, time: Time = 0) -> Time | None:
        """Apply an action at a time already advanced by the simulation event."""
        robot = self._robot(robot_id)
        # While a cell-to-cell movement is in progress, new decision ticks may only preserve
        # that movement. Treating, charging and battery exchange require physical co-location.
        if robot.move_target is not None and not isinstance(action, Move):
            return robot.move_arrival_at
        robot.current_action = action

        if isinstance(action, Move):
            self._stop_treatment(robot)
            self._stop_charging(robot)
            self._stop_battery_replacement(robot)
            return self._start_move(robot, action.to, time)
        if isinstance(action, Treat):
            self._stop_charging(robot)
            self._stop_battery_replacement(robot)
            self._apply_treat(robot, action.cell, time)
        elif isinstance(action, ReplaceBattery):
            self._stop_treatment(robot)
            self._stop_charging(robot)
            self._apply_replace_battery(robot, action.robot_id, time)
        elif isinstance(action, Charge):
            self._stop_treatment(robot)
            self._stop_battery_replacement(robot)
            self._start_charging(robot, time)
        elif isinstance(action, Idle):
            self._stop_treatment(robot)
            self._stop_charging(robot)
            self._stop_battery_replacement(robot)
        return None

    def advance_to(self, time: Time) -> None:
        """Advance movement, treatment, recharging, and replacement work through ``time``."""
        from swarm_field.swarm.capabilities import capabilities_for

        for robot in self.robots.values():
            if robot.treating_cell is not None and robot.treatment_updated_at is not None:
                elapsed = max(0, time - robot.treatment_updated_at)
                cell = self.cells.get(robot.treating_cell)
                if cell is not None and elapsed > 0:
                    # A severity of 1.0 takes one simulated hour to treat completely.
                    cell.infestation = max(0.0, cell.infestation - elapsed / TimeUnit.HOUR)
                robot.treatment_updated_at = time
                if cell is None or cell.infestation <= 0.0:
                    self._stop_treatment(robot)

            if robot.charging_updated_at is not None:
                profile = capabilities_for(robot.body)
                elapsed = max(0, time - robot.charging_updated_at)
                recharge_ms = (
                    None
                    if profile.dock_recharge_seconds is None
                    else profile.dock_recharge_seconds * TimeUnit.SECOND
                )
                if recharge_ms is not None and elapsed > 0:
                    robot.battery = min(1.0, robot.battery + elapsed / recharge_ms)
                robot.charging_updated_at = time
                if robot.battery >= 1.0 or recharge_ms is None:
                    self._stop_charging(robot)
                    robot.current_action = Idle()

            if robot.replacement_complete_at is not None and robot.replacement_complete_at <= time:
                self._complete_battery_replacement(robot)

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
        if not self.is_traversable_for(robot.body, target):
            raise ValueError(f"target is not traversable for {robot.body.name}: {target}")

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
        if not self.is_traversable_for(robot.body, target):
            robot.current_action = Idle()
            return None

        if robot.battery <= 0.0:
            robot.current_action = Idle()
            return None
        duration = self.movement_duration(robot.robot_id, target)
        from swarm_field.swarm.capabilities import capabilities_for

        endurance = capabilities_for(robot.body).endurance_seconds * TimeUnit.SECOND
        robot.battery = max(0.0, robot.battery - duration / endurance)
        robot.move_target = target
        robot.move_started_at = time
        robot.move_arrival_at = time + duration
        return robot.move_arrival_at

    def _apply_treat(self, robot: RobotState, target: CellId, time: Time) -> None:
        """Start or continue a treatment; severity decreases with elapsed simulation time."""
        if (
            robot.body is not BodyType.KILLER
            or target != robot.position
            or not self.is_inside(target)
            or self.cells[target].infestation <= 0.0
        ):
            robot.treating_cell = None
            robot.treatment_updated_at = None
            robot.current_action = Idle()
            return
        if robot.treating_cell != target:
            # Consumable stock is charged once when a treatment job starts. The last dose must
            # still be allowed to finish its treatment after stock reaches zero.
            if robot.consumable <= 0.0:
                robot.treating_cell = None
                robot.treatment_updated_at = None
                robot.current_action = Idle()
                return
            robot.consumable = max(0.0, robot.consumable - 0.1)
            robot.treating_cell = target
        robot.treatment_updated_at = time

    def _apply_replace_battery(
        self, support: RobotState, recipient_id: RobotId, time: Time
    ) -> None:
        """Start a battery exchange that completes after the configured service duration."""
        from swarm_field.swarm.capabilities import capabilities_for

        recipient = self.robots.get(recipient_id)
        if (
            support.body is not BodyType.BATTERY
            or support.spare_batteries <= 0
            or recipient is None
            or recipient.body not in (BodyType.SCOUT, BodyType.KILLER)
            or recipient.position != support.position
            or recipient.battery >= 1.0
        ):
            self._stop_battery_replacement(support)
            support.current_action = Idle()
            return

        if support.replacing_battery_for == recipient_id:
            # Repeated decision ticks must not restart an exchange already in progress.
            return
        self._stop_battery_replacement(support)
        duration = capabilities_for(support.body).battery_replacement_seconds * TimeUnit.SECOND
        support.replacing_battery_for = recipient_id
        support.replacement_complete_at = time + max(1, ceil(duration))
        support.current_action = ReplaceBattery(robot_id=recipient_id)

    def _complete_battery_replacement(self, support: RobotState) -> None:
        """Finish an exchange only if both robots are still co-located and stock remains."""
        recipient_id = support.replacing_battery_for
        recipient = self.robots.get(recipient_id) if recipient_id is not None else None
        if (
            support.body is BodyType.BATTERY
            and support.spare_batteries > 0
            and recipient is not None
            and recipient.body in (BodyType.SCOUT, BodyType.KILLER)
            and recipient.position == support.position
            and recipient.battery < 1.0
        ):
            recipient.battery = 1.0
            support.spare_batteries -= 1
        self._stop_battery_replacement(support)
        support.current_action = Idle()

    def _start_charging(self, robot: RobotState, time: Time) -> None:
        """Start charging at a dock; battery replacement stock is restocked there."""
        from swarm_field.swarm.capabilities import capabilities_for

        if robot.position not in self.docks:
            self._stop_charging(robot)
            robot.current_action = Idle()
            return
        if robot.body is BodyType.BATTERY:
            robot.spare_batteries = capabilities_for(robot.body).spare_battery_capacity
        if robot.battery >= 1.0:
            self._stop_charging(robot)
            robot.current_action = Idle()
            return
        if robot.charging_updated_at is None:
            robot.charging_updated_at = time
        robot.current_action = Charge()

    @staticmethod
    def _stop_treatment(robot: RobotState) -> None:
        robot.treating_cell = None
        robot.treatment_updated_at = None

    @staticmethod
    def _stop_charging(robot: RobotState) -> None:
        robot.charging_updated_at = None

    @staticmethod
    def _stop_battery_replacement(robot: RobotState) -> None:
        robot.replacing_battery_for = None
        robot.replacement_complete_at = None

    @staticmethod
    def _cell_center(cell: CellId) -> tuple[float, float]:
        return (cell[0] + 0.5, cell[1] + 0.5)

    def _robot(self, robot_id: RobotId) -> RobotState:
        try:
            return self.robots[robot_id]
        except KeyError as exc:
            raise ValueError(f"unknown robot: {robot_id}") from exc
