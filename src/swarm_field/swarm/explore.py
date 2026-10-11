"""The EXPLORE seeder: draws robots to the cells whose information is the stalest.

A cell is worth observing again by how long ago it was last observed, ``1 - exp(-age / tau)``:
0 right after an observation, close to 1 long after it, and 1 if it was never observed. So
exploring a new field and monitoring a known one are the same objective.

The robot's own block of ``block_side`` cells is seeded cell by cell, with its stale cells; every
other block gets one seed, at its stalest cell, worth the block's mean value. One field then does
both jobs: inside the robot's block it leads to the nearest stale cell, and once the block has no
stale cell left it leads to the best other block. A robot observes only the cell it stands on, so
every swept cell is a target used up; the key follows the stale cells of the robot's block, and
the field is rebuilt once per swept cell, but not while the robot travels over fresh ground.

With ``spread_peers`` the robots are made to spread out: a block's values are divided by one
plus the number of peers in it or heading to it.
"""

from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import CellId, Time, TimeUnit
from swarm_field.swarm.field import Seed
from swarm_field.swarm.memory import MemoryView


@dataclass(frozen=True)
class ExploreConfig:
    """The parameters of the EXPLORE field.

    Attributes:
        tau: How fast information goes stale: a cell is worth ``1 - exp(-age / tau)``.
        stale_threshold: A cell worth more than this is stale, and only stale cells of the
            robot's own block are seeded. The block is done when none is left.
        beta: How many cells of travel a fully stale cell, or a fully stale block, is worth.
        bucket: How long the mean values of the other blocks may drift before the field is
            rebuilt anyway.
        block_side: The side of a block in cells; matches ``Quadtree.max_leaf_side``.
        spread_peers: Whether to divide a block's values by one plus the number of peers in it
            or heading to it.
        peer_max_age: Peer reports older than this are left out of the discount.
    """

    tau: Time = 30 * TimeUnit.MINUTE
    stale_threshold: float = 0.5
    beta: float = 32.0
    bucket: Time = 5 * TimeUnit.MINUTE
    block_side: int = 16
    spread_peers: bool = False
    peer_max_age: Time = TimeUnit.MINUTE

    def block_of(self, cell: CellId) -> CellId:
        """The block containing ``cell``, in block coordinates."""
        return (cell[0] // self.block_side, cell[1] // self.block_side)


@dataclass(frozen=True)
class BlockValue:
    """What one block is worth to explore.

    Attributes:
        block: The block, in block coordinates.
        stalest: The block's most valuable cell, where its seed goes.
        mean: The mean value of the block's monitorable cells.
    """

    block: CellId
    stalest: CellId
    mean: float


class ExploreSeeder:
    """Seeds the EXPLORE field: the stale cells of the robot's block, and one seed per other block.

    Attributes:
        config: The parameters of the field.
    """

    name: str = "explore"

    def __init__(self, config: ExploreConfig | None = None) -> None:
        self.config: ExploreConfig = ExploreConfig()
        if config is not None:
            self.config = config

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        """Changes when a cell of the robot's block is swept or goes stale, and every ``bucket``.

        With ``spread_peers`` it also changes when the peers' blocks do. The mean values of the
        other blocks drift in between; the bucket bounds for how long.
        """
        stale = self._own_block_values(memory, robot, now) > self.config.stale_threshold
        peers: frozenset[tuple[CellId, int]] = frozenset()
        if self.config.spread_peers:
            peers = frozenset(peers_per_block(memory, now, self.config).items())
        return (now // self.config.bucket, stale.tobytes(), peers)

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        """A seed on every stale cell of the robot's block, and one on every other block.

        A stale cell starts at ``-beta * v``; another block starts at ``-beta * mean`` on its
        stalest cell. Both are divided by one plus the peers of their block with
        ``spread_peers``.
        """
        config = self.config
        monitorable = memory.monitorable_cells()
        values = cell_values(memory.last_observed_times(), monitorable, now, config.tau)
        peers: Counter[CellId] = Counter()
        if config.spread_peers:
            peers = peers_per_block(memory, now, config)

        own = config.block_of(robot)
        seeds: list[Seed] = []
        for block in block_values(values, monitorable, config.block_side):
            if block.block != own:
                worth = block.mean / (1 + peers[block.block])
                seeds.append(Seed(block.stalest, -config.beta * worth))

        corner_x = own[0] * config.block_side
        corner_y = own[1] * config.block_side
        own_values = values[
            corner_x : corner_x + config.block_side, corner_y : corner_y + config.block_side
        ]
        for dx, dy in np.argwhere(own_values > config.stale_threshold).tolist():
            worth = float(own_values[dx, dy]) / (1 + peers[own])
            seeds.append(Seed((corner_x + dx, corner_y + dy), -config.beta * worth))
        return tuple(seeds)

    def _own_block_values(
        self, memory: MemoryView, robot: CellId, now: Time
    ) -> NDArray[np.float64]:
        """The values of the cells of the robot's block, ``[dx, dy]`` from its corner."""
        side = self.config.block_side
        block_x, block_y = self.config.block_of(robot)
        cells = (
            slice(block_x * side, (block_x + 1) * side),
            slice(block_y * side, (block_y + 1) * side),
        )
        return cell_values(
            memory.last_observed_times()[cells],
            memory.monitorable_cells()[cells],
            now,
            self.config.tau,
        )


def cell_values(
    last_seen: NDArray[np.float64], monitorable: NDArray[np.bool_], now: Time, tau: Time
) -> NDArray[np.float64]:
    """What observing each cell again is worth, ``1 - exp(-age / tau)``, 0 if not monitorable.

    A cell never observed (``-inf``) is worth 1.
    """
    values = 1.0 - np.exp(-(now - last_seen) / tau)
    values[~monitorable] = 0.0
    return values


def block_values(
    values: NDArray[np.float64], monitorable: NDArray[np.bool_], side: int
) -> list[BlockValue]:
    """Every block with something to explore, with its mean value and its stalest cell.

    Blocks are the aligned squares of ``side`` cells; those along the far edges may be cut off by
    the border. All blocks are reduced at once, by padding the arrays to whole blocks and
    reshaping them to one row of ``side * side`` cells per block.
    """
    width, height = values.shape
    padding = ((0, -width % side), (0, -height % side))
    cells = _cells_per_block(np.pad(values, padding), side)
    counts = _cells_per_block(np.pad(monitorable, padding), side).sum(axis=2)
    means = cells.sum(axis=2) / np.maximum(counts, 1)
    stalest = cells.argmax(axis=2)

    found: list[BlockValue] = []
    for block_x, block_y in np.argwhere(means > 0.0).tolist():
        offset = int(stalest[block_x, block_y])
        cell = (block_x * side + offset // side, block_y * side + offset % side)
        found.append(BlockValue((block_x, block_y), cell, float(means[block_x, block_y])))
    return found


def peers_per_block(memory: MemoryView, now: Time, config: ExploreConfig) -> Counter[CellId]:
    """How many peers are in, or heading to, each block, from their recent reports.

    A peer counts once per block, also when it is heading to a cell of the block it is in.
    """
    counts: Counter[CellId] = Counter()
    for peer in memory.peers():
        if now - peer.observed_at > config.peer_max_age:
            continue
        blocks = {config.block_of(peer.cell)}
        if peer.target is not None:
            blocks.add(config.block_of(peer.target))
        counts.update(blocks)
    return counts


def _cells_per_block[T: np.generic](cells: NDArray[T], side: int) -> NDArray[T]:
    """Reshape cells ``[x, y]`` to ``[block_x, block_y, dx * side + dy]``.

    The shape of ``cells`` must be a multiple of ``side``.
    """
    blocks_x = cells.shape[0] // side
    blocks_y = cells.shape[1] // side
    by_block = cells.reshape(blocks_x, side, blocks_y, side).transpose(0, 2, 1, 3)
    return by_block.reshape(blocks_x, blocks_y, side * side)
