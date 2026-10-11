"""Tests for the EXPLORE seeder: cell and block values, seeds, key and peer discount."""

from math import exp

import numpy as np
import pytest

from swarm_field.contracts import (
    BodyType,
    CellId,
    FieldGeometry,
    InfestationObservation,
    PeerObservation,
    Terrain,
    TerrainObservation,
    Time,
)
from swarm_field.swarm import AgentMemory, ExploreConfig, ExploreSeeder
from swarm_field.swarm.explore import block_values, cell_values

_FIELD = FieldGeometry(width=32, height=16, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)
"""Two blocks of 16 x 16 cells: (0, 0) on the left and (1, 0) on the right."""
_TAU: Time = 1000


def _seen(cell: CellId, at: Time) -> InfestationObservation:
    return InfestationObservation(
        cell=cell, confidence=1.0, observed_at=at, source=1, severity=0.0, pest=None
    )


def _peer(peer: int, cell: CellId, target: CellId | None, at: Time) -> PeerObservation:
    return PeerObservation(
        cell=cell,
        confidence=1.0,
        observed_at=at,
        source=peer,
        peer=peer,
        body=BodyType.SCOUT,
        target=target,
    )


def _memory_with_left_block_seen(at: Time, except_cells: tuple[CellId, ...] = ()) -> AgentMemory:
    """A memory in which every cell of block (0, 0) was observed at ``at``, but ``except_cells``."""
    memory = AgentMemory(1, _FIELD)
    cells = [(x, y) for x in range(16) for y in range(16) if (x, y) not in except_cells]
    memory.merge([_seen(cell, at) for cell in cells])
    return memory


def _seeder(spread_peers: bool = False) -> ExploreSeeder:
    return ExploreSeeder(ExploreConfig(tau=_TAU, beta=10.0, spread_peers=spread_peers))


def test_cell_values_grow_with_age_and_ignore_cells_not_worth_monitoring() -> None:
    last_seen = np.array([[-np.inf, 100.0, 100.0 - _TAU]])
    monitorable = np.array([[True, True, False]])

    values = cell_values(last_seen, monitorable, now=100, tau=_TAU)

    assert values[0].tolist() == pytest.approx([1.0, 0.0, 0.0])
    aged = cell_values(np.array([[0.0]]), np.array([[True]]), now=_TAU, tau=_TAU)
    assert aged[0, 0] == pytest.approx(1 - exp(-1))


def test_block_values_average_monitorable_cells_and_find_the_stalest_one() -> None:
    values = np.zeros((6, 4))
    values[1, 2] = 0.8
    values[0, 0] = 0.4
    values[5, 3] = 1.0
    monitorable = np.ones((6, 4), dtype=np.bool_)
    monitorable[4, 0] = False

    blocks = {block.block: block for block in block_values(values, monitorable, side=4)}

    assert blocks[(0, 0)].stalest == (1, 2)
    assert blocks[(0, 0)].mean == pytest.approx(1.2 / 16)
    # The right block is cut off by the border at x = 6 and has one cell that is not monitorable.
    assert blocks[(1, 0)].stalest == (5, 3)
    assert blocks[(1, 0)].mean == pytest.approx(1.0 / 7)


def test_seeds_are_the_stale_cells_of_the_robots_block_and_one_per_other_block() -> None:
    memory = _memory_with_left_block_seen(at=100, except_cells=((3, 4), (9, 9)))

    seeds = _seeder().seeds(memory, robot=(0, 0), now=100)

    cells = {seed.cell: seed.value for seed in seeds}
    assert cells.keys() == {(3, 4), (9, 9), (16, 0)}
    assert cells[(3, 4)] == pytest.approx(-10.0)
    assert cells[(16, 0)] == pytest.approx(-10.0)


def test_known_obstacles_are_never_seeded() -> None:
    memory = _memory_with_left_block_seen(at=100, except_cells=((3, 4),))
    memory.merge(
        [
            TerrainObservation(
                cell=(3, 4), confidence=1.0, observed_at=0, source=1, terrain=Terrain.OBSTACLE
            )
        ]
    )

    seeds = _seeder().seeds(memory, robot=(0, 0), now=100)

    assert (3, 4) not in {seed.cell for seed in seeds}


def test_the_key_changes_per_swept_cell_of_the_robots_block_only() -> None:
    seeder = _seeder()
    memory = _memory_with_left_block_seen(at=100, except_cells=((3, 4),))
    key = seeder.field_key(memory, robot=(0, 0), now=100)

    memory.merge([_seen((5, 5), at=100), _seen((20, 3), at=100)])
    assert seeder.field_key(memory, robot=(0, 0), now=100) == key

    memory.merge([_seen((3, 4), at=100)])
    assert seeder.field_key(memory, robot=(0, 0), now=100) != key


def test_peers_discount_the_blocks_they_are_in_or_heading_to() -> None:
    memory = _memory_with_left_block_seen(at=100, except_cells=((3, 4),))
    memory.merge([_peer(2, cell=(1, 1), target=(20, 5), at=100)])

    plain = {seed.cell: seed.value for seed in _seeder().seeds(memory, (0, 0), now=100)}
    spread = {seed.cell: seed.value for seed in _seeder(True).seeds(memory, (0, 0), now=100)}

    assert spread[(3, 4)] == pytest.approx(plain[(3, 4)] / 2)
    assert spread[(16, 0)] == pytest.approx(plain[(16, 0)] / 2)


def test_old_peer_reports_are_left_out_of_the_discount() -> None:
    memory = _memory_with_left_block_seen(at=100, except_cells=((3, 4),))
    memory.merge([_peer(2, cell=(20, 5), target=None, at=0)])
    seeder = ExploreSeeder(ExploreConfig(tau=_TAU, beta=10.0, spread_peers=True, peer_max_age=50))

    seeds = {seed.cell: seed.value for seed in seeder.seeds(memory, (0, 0), now=100)}

    assert seeds[(16, 0)] == pytest.approx(-10.0)
