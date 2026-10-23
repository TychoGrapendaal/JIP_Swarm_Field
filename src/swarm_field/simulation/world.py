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
    spare_batteries: int = 0
    treating_cell: CellId | None = None
    treatment_updated_at: Time | None = None
    charging_updated_at: Time | None = None
    replacing_battery_for: RobotId | None = None
    replacement_complete_at: Time | None = None
    energy_spent: float = 0.0
    failed: bool = False
    failure_mode: FailureMode | None = None
    charging_requested_at: Time | None = None


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
        energy: EnergyConfig | None = None,
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
        self._charging_slots: dict[CellId, RobotId] = {}
        self._charging_queues: dict[CellId, list[RobotId]] = {dock: [] for dock in self.docks}
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
            spare_batteries=robot.spare_batteries,
        )

    def observe_for(self, robot_id: RobotId, time: Time) -> list[Observation]:
        """Generate body-specific pest, terrain, and peer observations."""
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
            terrain = Terrain.OBSTACLE if self.cells[cell].obstacle else Terrain.FREE
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
            for peer_id in self.robots_at(cell):
                if peer_id == robot_id:
                    continue
                peer = self._robot(peer_id)
                if peer.failed and peer.failure_mode is FailureMode.DEAD_BATTERY:
                    continue
                observations.append(
                    PeerObservation(
                        cell=cell,
                        confidence=1.0,
                        observed_at=time,
                        source=robot_id,
                        peer=peer_id,
                        body=peer.body,
                        battery=peer.battery,
                        target=peer.move_target,
                    )
                )

        return observations

    def fail_robot(self, robot_id: RobotId, mode: FailureMode, time: Time = 0) -> None:
        """Record a failure and stop the capabilities disabled by that failure mode."""
        robot = self._robot(robot_id)
        if robot.failed:
            return
        robot.failed = True
        robot.failure_mode = mode
        self._stop_charging(robot, time)
        match mode:
            case FailureMode.DEAD_BATTERY:
                robot.battery = 0.0
                self._stop_treatment(robot)
                self._stop_battery_replacement(robot)
                robot.move_target = None
                robot.move_started_at = None
                robot.move_arrival_at = None
                robot.current_action = Idle()
            case FailureMode.DISABLED:
                robot.health = 0.0
            case FailureMode.OFFLINE:
                robot.radio_range_m = 0.0
            case _:
                assert_never(mode)
        self._refresh_dock_slots(time)

    def battery_failure_count(self) -> int:
        """Count distinct robots recorded as failed because of battery depletion."""
        return sum(robot.failure_mode is FailureMode.DEAD_BATTERY for robot in self.robots.values())

    def live_robots(self) -> tuple[RobotId, ...]:
        """Every robot still taking part in the run, in id order."""
        return tuple(
            robot_id for robot_id, robot in sorted(self.robots.items()) if not robot.failed
        )

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
        """Apply an action while enforcing battery affordability and physical co-location."""
        robot = self._robot(robot_id)
        if robot.failed:
            return None

        # Every decision incurs its configured cost, including decisions while a committed move
        # is underway. Travel was paid up front, so a move may finish if the decision drains the
        # remaining battery; depletion is settled after arrival.
        can_pay_decision = robot.battery + 1e-12 >= self.energy.per_decision
        self._spend(robot, self.energy.per_decision)
        if robot.move_target is not None:
            return robot.move_arrival_at

        robot.current_action = action
        may_charge_empty_at_dock = isinstance(action, Charge) and robot.position in self.docks
        if not can_pay_decision and not may_charge_empty_at_dock:
            self._stop_treatment(robot)
            self._stop_battery_replacement(robot)
            if robot.position not in self.docks:
                self._settle_battery_failure(robot, time)
            else:
                robot.current_action = (
                    Charge() if robot.charging_requested_at is not None else Idle()
                )
            self._refresh_dock_slots(time)
            return None

        # An empty robot at a dock can only request/wait for charging, not start other work.
        if robot.battery <= 0.0 and robot.position in self.docks and not isinstance(action, Charge):
            robot.current_action = Charge() if robot.charging_requested_at is not None else Idle()
            self._settle_battery_failure(robot, time)
            self._refresh_dock_slots(time)
            return None

        next_event: Time | None = None
        if isinstance(action, Move):
            self._stop_treatment(robot)
            self._stop_charging(robot, time)
            self._stop_battery_replacement(robot)
            next_event = self._start_move(robot, action.to, time)
        elif isinstance(action, Treat):
            self._stop_charging(robot, time)
            self._stop_battery_replacement(robot)
            if robot.battery + 1e-12 < self.energy.treat_per_action:
                self._stop_treatment(robot)
                robot.current_action = Idle()
            else:
                self._spend(robot, self.energy.treat_per_action)
                self._apply_treat(robot, action.cell, time)
        elif isinstance(action, ReplaceBattery):
            self._stop_treatment(robot)
            self._stop_charging(robot, time)
            self._apply_replace_battery(robot, action.robot_id, time)
        elif isinstance(action, Charge):
            self._stop_treatment(robot)
            self._stop_battery_replacement(robot)
            self._start_charging(robot, time)
        elif isinstance(action, Idle):
            self._stop_treatment(robot)
            self._stop_charging(robot, time)
            self._stop_battery_replacement(robot)

        self._settle_battery_failure(robot, time)
        self._refresh_dock_slots(time)
        return next_event

    def energy_spent(self) -> float:
        """Total battery fraction every robot has consumed since the run began.

        Recharging does not subtract from this, so it measures the energy the swarm spent
        rather than the charge it happens to be holding.
        """
        return sum(robot.energy_spent for robot in self.robots.values())

    def advance_to(self, time: Time) -> None:
        """Advance active jobs, charging queues, and battery failure through ``time``."""
        if time < 0:
            raise ValueError("simulation time cannot be negative")
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
            if not robot.failed:
                robot.current_action = Idle()

        self._advance_charging(time)
        self._settle_battery_failures(time)
        self._refresh_dock_slots(time)

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
        if robot.failed or robot.move_target is not None:
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

        if robot.battery <= 0.0:
            robot.current_action = Idle()
            return None
        duration = self.movement_duration(robot.robot_id, target)
        travel_cost = self._travel_cost(robot, self._distance_m(robot.position, target))
        # Do not partially debit a movement and then start an action the robot cannot fund.
        if robot.battery + 1e-12 < travel_cost:
            robot.current_action = Idle()
            return None

        self._spend(robot, travel_cost)
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
            and not support.failed
            and support.spare_batteries > 0
            and recipient is not None
            and not recipient.failed
            and recipient.body in (BodyType.SCOUT, BodyType.KILLER)
            and recipient.position == support.position
            and recipient.battery < 1.0
        ):
            recipient.battery = 1.0
            support.spare_batteries -= 1
        self._stop_battery_replacement(support)
        support.current_action = Idle()

    def _start_charging(self, robot: RobotState, time: Time) -> None:
        """Request charging at a dock; each dock serves one robot in FIFO order."""
        from swarm_field.swarm.capabilities import capabilities_for

        if robot.position not in self.docks:
            self._stop_charging(robot, time)
            robot.current_action = Idle()
            return

        profile = capabilities_for(robot.body)
        if robot.body is BodyType.BATTERY:
            robot.spare_batteries = profile.spare_battery_capacity

        if robot.battery >= 1.0 or profile.dock_recharge_seconds <= 0.0:
            self._stop_charging(robot, time)
            robot.current_action = Idle()
            return

        if robot.charging_updated_at is not None or robot.charging_requested_at is not None:
            robot.current_action = Charge()
            return

        robot.charging_requested_at = time
        self._charging_queues.setdefault(robot.position, []).append(robot.robot_id)
        robot.current_action = Charge()
        self._start_next_charger(robot.position, time)

    def _advance_charging(self, time: Time) -> None:
        """Advance each dock's active charger and perform exact simulated-time handoffs."""
        from swarm_field.swarm.capabilities import capabilities_for

        for dock in sorted(self.docks):
            while True:
                charger_id = self._charging_slots.get(dock)
                if charger_id is None:
                    self._start_next_charger(dock, time)
                    charger_id = self._charging_slots.get(dock)
                    if charger_id is None:
                        break

                charger = self.robots.get(charger_id)
                if (
                    charger is None
                    or charger.failed
                    or charger.position != dock
                    or charger.charging_updated_at is None
                ):
                    self._charging_slots.pop(dock, None)
                    self._start_next_charger(dock, time)
                    continue

                recharge_ms = capabilities_for(charger.body).dock_recharge_seconds * TimeUnit.SECOND
                if recharge_ms <= 0.0:
                    self._stop_charging(charger, time)
                    charger.current_action = Idle()
                    continue

                updated_at = charger.charging_updated_at
                elapsed = max(0, time - updated_at)
                remaining = max(0.0, 1.0 - charger.battery)
                completion_at = updated_at + ceil(remaining * recharge_ms)
                if charger.battery >= 1.0 or completion_at <= time:
                    charger.battery = 1.0
                    self._release_charging_slot(dock, charger, completion_at)
                    charger.current_action = Idle()
                    self._start_next_charger(dock, completion_at)
                    continue

                if elapsed > 0:
                    charger.battery = min(1.0, charger.battery + elapsed / recharge_ms)
                charger.charging_updated_at = time
                break

    def _start_next_charger(self, dock: CellId, time: Time) -> None:
        """Promote the earliest waiting robot when a dock's charging slot is free."""
        from swarm_field.swarm.capabilities import capabilities_for

        if dock in self._charging_slots:
            return
        queue = self._charging_queues.setdefault(dock, [])
        while queue:
            robot_id = queue.pop(0)
            charger = self.robots.get(robot_id)
            if (
                charger is None
                or charger.failed
                or charger.position != dock
                or charger.charging_requested_at is None
                or charger.battery >= 1.0
                or capabilities_for(charger.body).dock_recharge_seconds <= 0.0
            ):
                if charger is not None:
                    charger.charging_requested_at = None
                    if not charger.failed:
                        charger.current_action = Idle()
                continue
            charger.charging_requested_at = None
            charger.charging_updated_at = time
            charger.current_action = Charge()
            self._charging_slots[dock] = robot_id
            return

    def _release_charging_slot(self, dock: CellId, robot: RobotState, time: Time) -> None:
        """Release a completed slot and clear the robot's charging state."""
        if self._charging_slots.get(dock) == robot.robot_id:
            self._charging_slots.pop(dock, None)
        robot.charging_updated_at = None
        robot.charging_requested_at = None
        self._start_next_charger(dock, time)

    def _refresh_dock_slots(self, time: Time) -> None:
        """Remove stale queue entries and ensure every free dock serves its next request."""
        from swarm_field.swarm.capabilities import capabilities_for

        for dock in sorted(self.docks):
            active_id = self._charging_slots.get(dock)
            if active_id is not None:
                active = self.robots.get(active_id)
                if (
                    active is None
                    or active.failed
                    or active.position != dock
                    or active.charging_updated_at is None
                    or active.battery >= 1.0
                    or capabilities_for(active.body).dock_recharge_seconds <= 0.0
                ):
                    if active is not None:
                        active.charging_updated_at = None
                        active.charging_requested_at = None
                    self._charging_slots.pop(dock, None)
            queue = self._charging_queues.setdefault(dock, [])
            eligible: list[RobotId] = []
            for robot_id in queue:
                robot = self.robots.get(robot_id)
                if (
                    robot is not None
                    and not robot.failed
                    and robot.position == dock
                    and robot.charging_requested_at is not None
                    and robot.charging_updated_at is None
                    and robot.battery < 1.0
                    and capabilities_for(robot.body).dock_recharge_seconds > 0.0
                ):
                    eligible.append(robot_id)
                elif robot is not None:
                    robot.charging_requested_at = None
                    if not robot.failed and robot.current_action == Charge():
                        robot.current_action = Idle()
            queue[:] = eligible
            self._start_next_charger(dock, time)

    def _settle_battery_failures(self, time: Time) -> None:
        """Fail empty robots after time advances, once committed transit is settled."""
        for robot in tuple(self.robots.values()):
            self._settle_battery_failure(robot, time)

    def _settle_battery_failure(self, robot: RobotState, time: Time) -> None:
        """Fail one robot away from a dock and not moving, whose battery is empty or stranded.

        A stranded robot cannot pay for even one straight hop. It never moves again unless a
        battery robot brings it a spare, so without one it is as dead as an empty robot, even
        with a sliver of charge left; counting only an empty battery would leave it idling as
        an active robot forever when decisions cost nothing.
        """
        if robot.failed or robot.position in self.docks or robot.move_target is not None:
            return
        empty = robot.battery <= 0.0
        stranded = robot.battery + 1e-12 < self._travel_cost(robot, self.field.cell_size)
        if empty or (stranded and not self._battery_supply_left()):
            self.fail_robot(robot.robot_id, FailureMode.DEAD_BATTERY, time=time)

    def _battery_supply_left(self) -> bool:
        """Whether a live battery robot still carries a spare to bring a stranded robot."""
        return any(
            robot.body is BodyType.BATTERY and not robot.failed and robot.spare_batteries > 0
            for robot in self.robots.values()
        )

    @staticmethod
    def _stop_treatment(robot: RobotState) -> None:
        robot.treating_cell: CellId | None = None
        robot.treatment_updated_at: Time | None = None

    def _stop_charging(self, robot: RobotState, time: Time) -> None:
        """Cancel an active or queued charging request and free its dock if needed."""
        freed_dock: CellId | None = None
        for dock, active_id in tuple(self._charging_slots.items()):
            if active_id == robot.robot_id:
                self._charging_slots.pop(dock, None)
                freed_dock = dock
        for dock, queue in self._charging_queues.items():
            if robot.robot_id in queue:
                queue[:] = [queued_id for queued_id in queue if queued_id != robot.robot_id]
                if freed_dock is None:
                    freed_dock = dock
        robot.charging_updated_at = None
        robot.charging_requested_at = None
        if freed_dock is not None:
            self._start_next_charger(freed_dock, time)

    @staticmethod
    def _stop_battery_replacement(robot: RobotState) -> None:
        robot.replacing_battery_for: RobotId | None = None
        robot.replacement_complete_at: Time | None = None

    @staticmethod
    def _spend(robot: RobotState, amount: float) -> None:
        """Draw ``amount`` of battery, never below empty, and record it as spent."""
        if amount <= 0.0:
            return
        drawn = min(robot.battery, amount)
        robot.battery -= drawn
        robot.energy_spent += drawn

    def _travel_cost(self, robot: RobotState, distance_m: float) -> float:
        """The battery a move of ``distance_m`` costs ``robot``: its travel time against its
        body's endurance, plus the energy model's cost per metre."""
        from swarm_field.swarm.capabilities import capabilities_for

        duration = max(1, ceil(distance_m / robot.speed_mps * TimeUnit.SECOND))
        endurance = capabilities_for(robot.body).endurance_seconds * TimeUnit.SECOND
        return duration / endurance + self.energy.per_metre * distance_m

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
