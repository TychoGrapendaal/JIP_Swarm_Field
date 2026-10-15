"""Tests for the ground-truth pest measures."""

from swarm_field.contracts import FieldGeometry
from swarm_field.simulation import InfestationTracker, WorldState


def test_a_pest_is_cleared_with_its_response_time_and_another_stays_open() -> None:
    world = WorldState(
        FieldGeometry(width=4, height=4, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)
    )
    tracker = InfestationTracker()
    world.cell((1, 1)).infestation = 0.5
    tracker.record(world, 0)
    world.cell((2, 2)).infestation = 0.3
    tracker.record(world, 1_000)
    world.cell((1, 1)).infestation = 0.0
    tracker.record(world, 5_000)

    summary = tracker.summary()

    assert (summary.spawned, summary.cleared, summary.open) == (2, 1, 1)
    assert summary.mean_response == 5_000
    assert summary.worst_response == 5_000
    assert summary.mean_infested == 4 / 3
