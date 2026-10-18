"""Scenario presets and construction of initial simulation worlds."""

from __future__ import annotations

from math import ceil, sqrt

from swarm_field.contracts import (
    BodyType,
    CellId,
    CropType,
    EnergySpec,
    FieldGeometry,
    PestDynamicsSpec,
    RobotSpec,
    ScenarioConfig,
    ScenarioId,
    TimeUnit,
)
from swarm_field.simulation.world import EnergyConfig, RobotState, WorldState

_DEFAULT_CELL_SIZE = 5.0
_DEFAULT_ROW_SPACING = 2.0
_DEFAULT_IN_ROW_SPACING = 1.0
_DEFAULT_HEADLAND = 5.0
_DEFAULT_ENERGY = EnergySpec(per_metre=2e-6, per_decision=1e-4, treat_per_action=2e-3)
"""Battery model shared by the presets: about fourteen hours of idling on a full charge."""


def build_world(config: ScenarioConfig) -> WorldState:
    """Construct and populate a ``WorldState`` from a scenario configuration.

    Crops, obstacles and initial infestations are copied into the ground-truth world. Robot
    specifications become mutable ``RobotState`` objects without changing the configuration.
    """
    _validate_config(config)
    world = WorldState(
        config.field, energy=EnergyConfig.from_spec(config.energy), roles=config.roles
    )

    for cell, crop in config.crops.items():
        world.cell(cell).crop = crop
        world.cell(cell).growth_stage = 1.0

    for cell in config.obstacles:
        world.cell(cell).obstacle = True

    for cell, severity in config.initial_infestation.items():
        world.cell(cell).infestation = severity

    for spec in config.robots:
        world.add_robot(
            RobotState(
                robot_id=spec.robot_id,
                position=spec.start,
                body=spec.body,
                speed_mps=spec.v_max,
                radio_range_m=spec.radio_range,
                battery=spec.battery,
            )
        )

    return world


def build_scenario(scenario: ScenarioId, seed: int | None = None) -> ScenarioConfig:
    """Return one of the five built-in swarm demonstration scenarios."""
    builders = {
        ScenarioId.S1: _build_s1,
        ScenarioId.S2: _build_s2,
        ScenarioId.S3: _build_s3,
        ScenarioId.S4: _build_s4,
        ScenarioId.S5: _build_s5,
    }
    return builders[scenario](seed if seed is not None else _default_seed(scenario))


def _build_s1(seed: int) -> ScenarioConfig:
    field = _field(30, 30)
    return ScenarioConfig(
        field=field,
        crops=_uniform_crop(field, CropType.MAIZE),
        robots=_robot_specs(field, 10, BodyType.SCOUT, ()),
        seed=seed,
        initial_infestation={},
        obstacles=(),
        pest_dynamics=PestDynamicsSpec(
            period=TimeUnit.MINUTE,
            spawn_probability=0.0,
            growth_rate=0.0,
            spread_probability=0.0,
            spawn_severity=0.1,
        ),
        energy=_DEFAULT_ENERGY,
    )


def _build_s2(seed: int) -> ScenarioConfig:
    field = _field(40, 40)
    infestation = _clusters(((8, 10), (30, 28), (20, 32)), radius=2, severity=0.6, field=field)
    robots = _typed_robot_specs(field, (10, 5, 0), BodyType.SCOUT, BodyType.KILLER, infestation)
    return ScenarioConfig(
        field=field,
        crops=_uniform_crop(field, CropType.MAIZE),
        robots=robots,
        seed=seed,
        initial_infestation=infestation,
        obstacles=(),
        pest_dynamics=PestDynamicsSpec(
            period=TimeUnit.MINUTE,
            spawn_probability=0.0,
            growth_rate=0.05,
            spread_probability=0.15,
            spawn_severity=0.10,
        ),
        energy=_DEFAULT_ENERGY,
    )


def _build_s3(seed: int) -> ScenarioConfig:
    field = _field(50, 50)
    crops = _striped_crops(field)
    infestation = _clusters(((12, 12), (37, 36), (38, 12)), radius=1, severity=0.45, field=field)
    robots = _typed_robot_specs(field, (12, 6, 2), BodyType.SCOUT, BodyType.KILLER, infestation)
    robots[-1].body = BodyType.BATTERY
    robots[-2].body = BodyType.BATTERY
    return ScenarioConfig(
        field=field,
        crops=crops,
        robots=robots,
        seed=seed,
        initial_infestation=infestation,
        obstacles=(),
        pest_dynamics=PestDynamicsSpec(
            period=TimeUnit.MINUTE,
            spawn_probability=0.005,
            growth_rate=0.04,
            spread_probability=0.18,
            spawn_severity=0.10,
        ),
        energy=_DEFAULT_ENERGY,
    )


def _build_s4(seed: int) -> ScenarioConfig:
    field = _field(60, 60)
    crops = _striped_crops(field, wider=True)
    infestation = _clusters(
        ((10, 10), (30, 40), (50, 18), (45, 50)),
        radius=2,
        severity=0.5,
        field=field,
    )
    robots = _typed_robot_specs(field, (18, 10, 2), BodyType.SCOUT, BodyType.KILLER, infestation)
    robots[-1].body = BodyType.BATTERY
    robots[-2].body = BodyType.BATTERY
    return ScenarioConfig(
        field=field,
        crops=crops,
        robots=robots,
        seed=seed,
        initial_infestation=infestation,
        obstacles=(),
        pest_dynamics=PestDynamicsSpec(
            period=30_000,
            spawn_probability=0.01,
            growth_rate=0.06,
            spread_probability=0.25,
            spawn_severity=0.10,
        ),
        energy=_DEFAULT_ENERGY,
    )


def _build_s5(seed: int) -> ScenarioConfig:
    field = _field(70, 70)
    crops = _striped_crops(field, wider=True)
    obstacles = _disturbed_obstacles(field)
    infestation = _clusters(
        ((10, 12), (28, 56), (52, 22), (60, 55), (38, 34)),
        radius=2,
        severity=0.65,
        field=field,
    )
    infestation = {
        cell: severity for cell, severity in infestation.items() if cell not in obstacles
    }
    robots = _typed_robot_specs(
        field,
        (20, 14, 6),
        BodyType.SCOUT,
        BodyType.KILLER,
        infestation,
        obstacles=obstacles,
    )
    return ScenarioConfig(
        field=field,
        crops=crops,
        robots=robots,
        seed=seed,
        initial_infestation=infestation,
        obstacles=obstacles,
        pest_dynamics=PestDynamicsSpec(
            period=20_000,
            spawn_probability=0.015,
            growth_rate=0.07,
            spread_probability=0.30,
            spawn_severity=0.10,
        ),
        energy=_DEFAULT_ENERGY,
    )


def _field(width: int, height: int) -> FieldGeometry:
    return FieldGeometry(
        width=width,
        height=height,
        cell_size=_DEFAULT_CELL_SIZE,
        row_sp=_DEFAULT_ROW_SPACING,
        in_row_sp=_DEFAULT_IN_ROW_SPACING,
        headland=_DEFAULT_HEADLAND,
        connectivity=8,
    )


def _uniform_crop(field: FieldGeometry, crop: CropType) -> dict[CellId, CropType]:
    return {(x, y): crop for y in range(field.height) for x in range(field.width)}


def _striped_crops(field: FieldGeometry, wider: bool = False) -> dict[CellId, CropType]:
    stripe_width = 8 if wider else 6
    crops = (CropType.MAIZE, CropType.WHEAT, CropType.POTATO, CropType.SUGAR_BEET)
    result: dict[CellId, CropType] = {}
    for y in range(field.height):
        stripe = min(y // stripe_width, len(crops) - 1)
        for x in range(field.width):
            result[(x, y)] = crops[stripe]
    return result


def _clusters(
    centres: tuple[CellId, ...],
    radius: int,
    severity: float,
    field: FieldGeometry,
) -> dict[CellId, float]:
    result: dict[CellId, float] = {}
    for centre_x, centre_y in centres:
        for y in range(centre_y - radius, centre_y + radius + 1):
            for x in range(centre_x - radius, centre_x + radius + 1):
                cell = (x, y)
                if not (0 <= x < field.width and 0 <= y < field.height):
                    continue
                distance = max(abs(x - centre_x), abs(y - centre_y))
                scaled = max(0.0, severity * (1.0 - 0.18 * distance))
                result[cell] = max(result.get(cell, 0.0), scaled)
    return result


def _disturbed_obstacles(field: FieldGeometry) -> tuple[CellId, ...]:
    cells: set[CellId] = set()
    for x in range(8, field.width - 8):
        if x % 7 in (0, 1):
            for y in range(23, 27):
                cells.add((x, y))
    for y in range(8, field.height - 8):
        if y % 9 in (0, 1):
            for x in range(33, 37):
                cells.add((x, y))
    return tuple(sorted(cells))


def _typed_robot_specs(
    field: FieldGeometry,
    counts: tuple[int, int, int],
    first_body: BodyType,
    second_body: BodyType,
    blocked_context: dict[CellId, float],
    obstacles: tuple[CellId, ...] = (),
) -> list[RobotSpec]:
    scouts, killers, batteries = counts
    return (
        _robot_specs(field, scouts, first_body, obstacles)
        + _robot_specs(
            field,
            killers,
            second_body,
            obstacles,
            start_offset=scouts,
            avoid=blocked_context,
        )
        + _robot_specs(
            field,
            batteries,
            BodyType.BATTERY,
            obstacles,
            start_offset=scouts + killers,
            avoid=blocked_context,
        )
    )


def _robot_specs(
    field: FieldGeometry,
    count: int,
    body: BodyType,
    obstacles: tuple[CellId, ...],
    start_offset: int = 0,
    avoid: dict[CellId, float] | None = None,
) -> list[RobotSpec]:
    starts = _distributed_starts(field, count, set(obstacles), start_offset, avoid or {})
    return [
        RobotSpec(
            robot_id=start_offset + index + 1,
            body=body,
            start=start,
            v_max=1.0 if body is BodyType.SCOUT else 0.8,
            battery=1.0,
            radio_range=25.0,
        )
        for index, start in enumerate(starts)
    ]


def _distributed_starts(
    field: FieldGeometry,
    count: int,
    blocked: set[CellId],
    start_offset: int,
    avoid: dict[CellId, float],
) -> list[CellId]:
    if count <= 0:
        return []

    columns = max(1, ceil(sqrt(count)))
    rows = ceil(count / columns)
    x_step = max(1, field.width // (columns + 1))
    y_step = max(1, field.height // (rows + 1))
    candidates: list[CellId] = []
    for row in range(rows):
        for column in range(columns):
            x = min(field.width - 1, (column + 1) * x_step)
            y = min(field.height - 1, (row + 1) * y_step)
            candidates.append((x, y))

    starts: list[CellId] = []
    used = set(blocked)
    for candidate in candidates:
        if candidate in used or candidate in avoid:
            continue
        starts.append(candidate)
        used.add(candidate)
        if len(starts) == count:
            return starts

    for y in range(field.height):
        for x in range(field.width):
            cell = (x, y)
            if cell in used or cell in avoid:
                continue
            starts.append(cell)
            used.add(cell)
            if len(starts) == count:
                return starts

    raise ValueError("not enough walkable start cells for requested robots")


def _validate_config(config: ScenarioConfig) -> None:
    """Validate scenario data before populating a world."""
    field = config.field
    if len({robot.robot_id for robot in config.robots}) != len(config.robots):
        raise ValueError("robot ids must be unique")

    for cell in config.crops:
        if not _inside(field, cell):
            raise ValueError(f"crop cell outside field: {cell}")

    for cell in config.obstacles:
        if not _inside(field, cell):
            raise ValueError(f"obstacle cell outside field: {cell}")

    for cell, severity in config.initial_infestation.items():
        if not _inside(field, cell):
            raise ValueError(f"infestation cell outside field: {cell}")
        if not 0.0 <= severity <= 1.0:
            raise ValueError("infestation severity must be between 0 and 1")
        if cell in config.obstacles:
            raise ValueError(f"infestation cannot start on obstacle cell: {cell}")

    for robot in config.robots:
        if not _inside(field, robot.start):
            raise ValueError(f"robot start outside field: {robot.start}")
        if robot.start in config.obstacles:
            raise ValueError(f"robot cannot start on obstacle cell: {robot.start}")
        if robot.v_max <= 0.0:
            raise ValueError("robot v_max must be positive")
        if not 0.0 <= robot.battery <= 1.0:
            raise ValueError("robot battery must be between 0 and 1")
        if robot.radio_range < 0.0:
            raise ValueError("robot radio_range cannot be negative")


def _inside(field: FieldGeometry, cell: CellId) -> bool:
    x, y = cell
    return 0 <= x < field.width and 0 <= y < field.height


def _default_seed(scenario: ScenarioId) -> int:
    return 100 + scenario.value
