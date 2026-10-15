"""Tests for metrics collection (B.5) and its CSV/JSON export."""

import csv
import json
from pathlib import Path
from random import Random

import pytest

from swarm_field.contracts import (
    BodyType,
    CropObservation,
    CropType,
    FieldGeometry,
    InfestationObservation,
    MetricsSample,
    Move,
    Terrain,
    TerrainObservation,
    Time,
)
from swarm_field.simulation import (
    EnergyConfig,
    MetricSample,
    MetricsCollector,
    MetricsConfig,
    PestDynamicsConfig,
    RobotState,
    SimulationContext,
    SimulationLoop,
    WorldState,
    WorldTick,
    write_metrics_csv,
    write_metrics_json,
)


def _field(width: int = 4, height: int = 4) -> FieldGeometry:
    return FieldGeometry(
        width=width,
        height=height,
        cell_size=10.0,
        row_sp=0.0,
        in_row_sp=0.0,
        headland=0.0,
        connectivity=4,
    )


def _cropped_world() -> WorldState:
    """A small field where every cell is cropped, so coverage has a known denominator."""
    world = WorldState(_field(), energy=EnergyConfig(per_decision=0.01))
    for cell in world.cells:
        world.cell(cell).crop = CropType.MAIZE
    return world


def _collector(world: WorldState) -> MetricsCollector:
    collector = MetricsCollector(config=MetricsConfig(period=100))
    collector.bind(world)
    return collector


def _latest(collector: MetricsCollector) -> MetricsSample:
    latest = collector.latest()
    assert latest is not None
    return latest


def _context(world: WorldState, collector: MetricsCollector) -> SimulationContext:
    return SimulationContext(now=0, rng=Random(0), agents={}, world=world, metrics=collector)


def _terrain(cell: tuple[int, int], now: Time = 0) -> TerrainObservation:
    return TerrainObservation(
        cell=cell, confidence=1.0, observed_at=now, source=1, terrain=Terrain.FREE
    )


def _infestation(cell: tuple[int, int], severity: float, now: Time = 0) -> InfestationObservation:
    return InfestationObservation(
        cell=cell, confidence=1.0, observed_at=now, source=1, severity=severity, pest=None
    )


def _crop(cell: tuple[int, int], now: Time = 0) -> CropObservation:
    return CropObservation(
        cell=cell,
        confidence=1.0,
        observed_at=now,
        source=1,
        crop=CropType.MAIZE,
        growth_stage=1.0,
    )


def test_sample_counts_targets_and_reads_coverage_from_truth() -> None:
    world = _cropped_world()
    collector = _collector(world)
    world.cell((0, 0)).infestation = 0.4
    world.cell((1, 1)).infestation = 0.2
    collector.observe([_terrain((0, 0)), _terrain((1, 1))], now=0)

    taken = collector.sample(world, now=0, messages_sent=3)

    assert isinstance(taken, MetricsSample)
    assert taken.time == 0
    assert taken.targets_remaining == 2
    assert taken.coverage == pytest.approx(2 / 16)
    assert taken.messages_sent == 3


def test_coverage_is_monotone_and_capped_by_the_cropped_cells() -> None:
    world = _cropped_world()
    collector = _collector(world)

    collector.observe([_terrain((0, 0))], now=0)
    first = collector.sample(world, now=0, messages_sent=0)
    collector.observe([_terrain((0, 0)), _terrain((0, 1))], now=10)
    second = collector.sample(world, now=10, messages_sent=0)

    assert second.coverage > first.coverage
    collector.observe([_terrain(cell) for cell in world.cells], now=20)
    assert collector.sample(world, now=20, messages_sent=0).coverage == 1.0


def test_detection_latency_is_the_delay_from_infestation_to_first_sighting() -> None:
    world = _cropped_world()
    collector = _collector(world)
    world.cell((2, 2)).infestation = 0.5
    collector.sample(world, now=100, messages_sent=0)
    assert _latest(collector).detection_latency_ms is None

    collector.observe([_infestation((2, 2), 0.5)], now=340)
    collector.observe([_infestation((2, 2), 0.5)], now=340)

    taken = collector.sample(world, now=400, messages_sent=0)
    assert taken.detection_latency_ms == pytest.approx(240)


def test_a_target_present_at_start_is_dated_from_the_start_not_the_first_sample() -> None:
    """Robots sense before the first sample, so bind() must date the initial infestation."""
    world = _cropped_world()
    world.cell((2, 2)).infestation = 0.5
    collector = _collector(world)
    collector.observe([_infestation((2, 2), 0.5)], now=0)

    taken = collector.sample(world, now=1000, messages_sent=0)

    assert taken.detection_latency_ms == pytest.approx(0)


def test_bare_soil_is_never_counted_as_covered() -> None:
    world = _cropped_world()
    world.cell((3, 3)).crop = None
    collector = _collector(world)

    collector.observe([_terrain((3, 3))], now=0)
    taken = collector.sample(world, now=0, messages_sent=0)

    assert taken.coverage == 0.0


def test_energy_per_area_normalises_by_the_field_area() -> None:
    world = _cropped_world()
    robot = RobotState(robot_id=1, position=(0, 0), body=BodyType.SCOUT, speed_mps=1.0)
    world.add_robot(robot)
    collector = _collector(world)

    world.apply_action(1, Move(to=(1, 0)), time=0)
    taken = collector.sample(world, now=0, messages_sent=0)

    # 16 cells of 10 m each is 1600 m^2; the move cost 0.01 for the decision and 10 m of travel.
    assert world.energy_spent() == pytest.approx(0.01 + 0.0)
    assert taken.energy_per_area == pytest.approx(world.energy_spent() / 1600)


def test_metric_sample_event_records_a_series_on_its_own_cadence() -> None:
    world = _cropped_world()
    collector = _collector(world)
    ctx = _context(world, collector)
    ctx.schedule(MetricSample(time=0, config=MetricsConfig(period=100)))

    SimulationLoop().run(ctx, until=250)

    assert [sample.time for sample in collector.series()] == [0, 100, 200]


def test_recording_metrics_leaves_the_run_bit_for_bit_identical() -> None:
    """A run's world and random stream must not depend on whether metrics are recorded."""
    pest = PestDynamicsConfig(period=100, spawn_probability=0.3, spread_probability=0.3)

    def run(collect: bool) -> tuple[dict[tuple[int, int], float], list[float]]:
        world = _cropped_world()
        rng = Random(7)
        ctx = SimulationContext(
            now=0,
            rng=rng,
            agents={},
            world=world,
            metrics=_collector(world) if collect else None,
        )
        ctx.schedule(WorldTick(time=0, config=pest))
        if collect:
            ctx.schedule(MetricSample(time=0, config=MetricsConfig(period=100)))
        SimulationLoop().run(ctx, until=250)
        infestations = {cell: state.infestation for cell, state in sorted(world.cells.items())}
        return infestations, [rng.random() for _ in range(3)]

    assert run(collect=True) == run(collect=False)


def test_series_and_latest_expose_the_samples_in_order() -> None:
    world = _cropped_world()
    collector = _collector(world)
    ctx = _context(world, collector)
    ctx.schedule(MetricSample(time=0, config=MetricsConfig(period=50)))

    assert collector.latest() is None
    SimulationLoop().run(ctx, until=100)

    assert [sample.time for sample in collector.series()] == [0, 50, 100]
    assert _latest(collector).time == 100


def test_target_threshold_ignores_cells_below_it() -> None:
    world = _cropped_world()
    world.cell((0, 0)).infestation = 0.02
    world.cell((0, 1)).infestation = 0.5
    collector = MetricsCollector(config=MetricsConfig(target_threshold=0.1))
    collector.bind(world)

    taken = collector.sample(world, now=0, messages_sent=0)

    assert taken.targets_remaining == 1


def test_write_metrics_csv_round_trips_the_series(tmp_path: Path) -> None:
    world = _cropped_world()
    collector = _collector(world)
    collector.sample(world, now=0, messages_sent=1)
    collector.sample(world, now=100, messages_sent=4)
    path = tmp_path / "run.csv"

    write_metrics_csv(collector.series(), path)

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["time"]) for row in rows] == [0, 100]
    assert [int(row["messages_sent"]) for row in rows] == [1, 4]
    assert list(rows[0]) == list(MetricsSample.__dataclass_fields__)


def test_write_metrics_json_round_trips_the_series(tmp_path: Path) -> None:
    world = _cropped_world()
    collector = _collector(world)
    taken = collector.sample(world, now=0, messages_sent=2)
    path = tmp_path / "run.json"

    write_metrics_json(collector.series(), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == [taken.to_row()]


def test_export_writes_into_a_missing_directory(tmp_path: Path) -> None:
    world = _cropped_world()
    collector = _collector(world)
    collector.sample(world, now=0, messages_sent=0)
    path = tmp_path / "nested" / "run.csv"

    write_metrics_csv(collector.series(), path)

    assert path.exists()


def test_metrics_config_rejects_a_non_positive_period() -> None:
    with pytest.raises(ValueError):
        MetricsConfig(period=0)


def test_metrics_config_rejects_a_threshold_outside_zero_to_one() -> None:
    with pytest.raises(ValueError):
        MetricsConfig(target_threshold=1.5)
