"""Tests for the ground-truth coverage tracker."""

from random import Random

import pytest

from swarm_field.contracts import BodyType, FieldGeometry
from swarm_field.simulation import (
    CoverageSample,
    CoverageTracker,
    RobotState,
    SimulationContext,
    SimulationLoop,
    WorldState,
)


def _world() -> WorldState:
    """A 2 x 1 field with one robot in cell (0, 0)."""
    field = FieldGeometry(width=2, height=1, cell_size=1.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)
    robot = RobotState(robot_id=1, position=(0, 0), body=BodyType.SCOUT, speed_mps=1.0)
    return WorldState(field, robots=[robot])


def test_visits_settle_coverage_redundancy_and_idleness() -> None:
    world = _world()
    tracker = CoverageTracker(world, fresh_age=50)

    tracker.record(world, now=0)
    world.robots[1].position = (1, 0)
    tracker.record(world, now=10)
    world.robots[1].position = (0, 0)
    tracker.record(world, now=20)

    summary = tracker.summary()
    assert summary.covered == 1.0
    assert summary.time_to_full == 10
    assert summary.visits == 3
    assert summary.redundancy == pytest.approx(1 / 3)
    # Ages per sample: (0, 0) -> 0, (0, 10) -> 5, (10, 0) -> 5.
    assert summary.mean_idleness == pytest.approx(10 / 3)
    assert summary.worst_idleness == 10


def test_obstacles_are_not_counted_and_an_unvisited_cell_leaves_coverage_short() -> None:
    world = _world()
    world.cell((1, 0)).obstacle = True
    tracker = CoverageTracker(world, fresh_age=50)

    tracker.record(world, now=0)

    assert tracker.summary().time_to_full == 0


def test_coverage_samples_record_at_their_period() -> None:
    world = _world()
    tracker = CoverageTracker(world, fresh_age=50, measure_from=100)
    ctx = SimulationContext(now=0, rng=Random(0), agents={}, world=world)
    ctx.schedule(CoverageSample(time=0, period=50, tracker=tracker))

    SimulationLoop().run(ctx, until=200)

    # Samples at 100, 150 and 200 count. The robot never moved from (0, 0), visited at 0, and
    # (1, 0) was never visited, so it is as old as the run: both are as old as the run.
    assert tracker.summary().mean_idleness == pytest.approx((100 + 150 + 200) / 3)
    assert tracker.summary().covered == 0.5
