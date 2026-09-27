"""Tests for the navigator: the one movement seam, its field cache, and the steering fallback."""

from swarm_field.contracts import (
    CellId,
    Direction,
    FieldGeometry,
    Idle,
    Move,
    Terrain,
    TerrainObservation,
)
from swarm_field.swarm import AgentMemory, Navigator, QuadtreeSensing
from swarm_field.swarm.field import Seed

_FIELD = FieldGeometry(
    width=40, height=40, cell_size=1.0, row_sp=0.0, in_row_sp=0.0, headland=0.0, connectivity=8
)


def _navigator(obstacles: set[CellId]) -> Navigator:
    """A navigator over a 40-by-40 field that knows the given cells as obstacles."""
    memory = AgentMemory(1, _FIELD)
    facts = [
        TerrainObservation(
            cell=cell, confidence=1.0, observed_at=0, source=1, terrain=Terrain.OBSTACLE
        )
        for cell in sorted(obstacles)
    ]
    memory.merge(facts)
    return Navigator(memory, flies=False)


def test_the_field_walks_the_robot_around_a_wall_to_its_target() -> None:
    """A field guides the robot the whole way, at a distance, around what it knows."""
    wall = {(20, y) for y in range(18, 24)}
    navigator = _navigator(wall)
    seeds = (Seed((30, 21), 0.0),)
    position: CellId = (10, 21)
    visited = [position]

    for _ in range(200):
        action = navigator.step_towards(seeds, position)
        if isinstance(action, Idle):
            break
        position = action.to
        visited.append(position)

    assert position == (30, 21)
    assert not wall.intersection(visited[1:])


def test_the_field_is_reused_until_the_block_or_the_terrain_changes() -> None:
    """A field is kept for as long as it still answers for the same plan."""
    navigator = _navigator(set())
    seeds = (Seed((30, 30), 0.0),)

    in_first_block = navigator.field_for(seeds, (0, 0))
    assert navigator.field_for(seeds, (3, 3)) is in_first_block

    in_next_block = navigator.field_for(seeds, (17, 0))
    assert in_next_block is not in_first_block
    assert navigator.field_for(seeds, (17, 0)) is in_next_block

    navigator.update_terrain([(1, 1)])
    assert navigator.field_for(seeds, (17, 0)) is not in_next_block


def test_a_walled_off_target_hands_over_to_the_steerer_and_then_reports() -> None:
    """No route at all: the steerer tries its own way, and the stall is the brain's cue."""
    wall = {(20, y) for y in range(40)}
    navigator = _navigator(wall)
    seeds = (Seed((30, 20), 0.0),)
    position: CellId = (10, 20)
    actions: list[Move | Idle] = []

    for _ in range(200):
        action = navigator.step_towards(seeds, position)
        actions.append(action)
        if isinstance(action, Idle):
            break
        position = action.to

    assert any(isinstance(action, Move) for action in actions)
    assert not {action.to for action in actions if isinstance(action, Move)} & wall
    assert isinstance(actions[-1], Idle)


def test_no_seeds_means_no_movement() -> None:
    assert isinstance(_navigator(set()).step_towards((), (0, 0)), Idle)


def test_the_map_backed_sensing_never_cuts_a_blocked_corner() -> None:
    """The steerer must not read as free a diagonal the quadtree's own routing refuses."""
    both_corners_blocked = QuadtreeSensing(_navigator({(11, 0), (10, 1)}).quadtree)
    assert not both_corners_blocked.clear((10, 0), Direction.NE)
    assert not both_corners_blocked.clear((10, 0), Direction.E)
    assert not both_corners_blocked.clear((10, 0), Direction.N)

    one_corner_blocked = QuadtreeSensing(_navigator({(11, 0)}).quadtree)
    assert not one_corner_blocked.clear((10, 0), Direction.NE)
    assert one_corner_blocked.clear((10, 0), Direction.N)

    both_corners_free = QuadtreeSensing(_navigator({(12, 0)}).quadtree)
    assert both_corners_free.clear((10, 0), Direction.NE)
