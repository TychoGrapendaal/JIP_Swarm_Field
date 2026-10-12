"""The EXPLORE seeder: draws robots to the cells whose information is the stalest.

A cell is worth observing again by how long ago it was last observed. It is worth nothing until
``revisit_age`` ``A`` has passed; then it is stale, worth ``1 - 2^(-age / A)``: 0.5 at the revisit
age, approaching 1, and 1 if it was never observed. So exploring a new field and monitoring a
known one are the same objective.

The robot's own block is seeded cell by cell, with its stale cells; every other block gets one
seed, on its stalest cell and worth that cell's value, so a forgotten cell is never hidden in a
block of fresh ones. Blocks may grow with the distance from the robot (``block_sides``). One
field then does both jobs: inside the robot's block it leads to the nearest stale cell, and once
the block has no stale cell left it leads to the best other block. A robot observes only the
cell it stands on, so every swept cell is a target used up; the key follows the stale cells of
the robot's block, and the field is rebuilt once per swept cell, but not while the robot travels
over fresh ground.

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

    The defaults did best in ``tests/benchmark_explore.py`` (see ``docs/explore.md``) for five
    robots on 32 x 32 and 64 x 64 fields.

    Attributes:
        revisit_age: How long a cell is worth nothing after it was observed. From then on it is
            stale, worth ``1 - 2^(-age / revisit_age)``, and only the stale cells of the robot's
            own block are seeded; the block is done when none is left.
        beta: How many cells of travel a fully stale cell, or a fully stale block, is worth.
        bucket: How long the mean values of the other blocks may drift before the field is
            rebuilt anyway.
        block_sides: The sides of the blocks in cells, finest first. The robot's own block, of
            the first side, is seeded cell by cell. The blocks of each side within
            ``near_blocks`` blocks of the robot's are seeded one per block, and the blocks of
            the next side cover what lies beyond; the last side covers everything left. With
            ``(16,)``, every other block is a 16 x 16 block, the quadtree's largest leaf.
        near_blocks: How many blocks around the robot's each side but the last reaches.
        spread_peers: Whether to divide a block's values by one plus the number of peers in it
            or heading to it.
        peer_max_age: Peer reports older than this are left out of the discount.
        window_blocks: How many blocks of ``Quadtree.max_leaf_side`` cells around the robot's
            block the field is searched cell by cell. The sweep needs only the robot's own
            block, but beyond the window a block is one leaf, measured from its centre, which
            makes the neighbouring blocks look farther than they are; 1 keeps them exact.
    """

    revisit_age: Time = 28 * TimeUnit.MINUTE
    beta: float = 64.0
    bucket: Time = 5 * TimeUnit.MINUTE
    block_sides: tuple[int, ...] = (8,)
    near_blocks: int = 1
    spread_peers: bool = True
    peer_max_age: Time = TimeUnit.MINUTE
    window_blocks: int = 1

    @property
    def block_side(self) -> int:
        """The side of the robot's own block: the finest blocks."""
        return self.block_sides[0]

    def block_of(self, cell: CellId) -> CellId:
        """The robot's own block if it stands on ``cell``, in block coordinates."""
        return block_containing(cell, self.block_side)


@dataclass(frozen=True)
class BlockValue:
    """What one block is worth to explore.

    Attributes:
        block: The block, in block coordinates.
        stalest: The block's most valuable cell, where its seed goes.
        worth: The value of that cell.
    """

    block: CellId
    stalest: CellId
    worth: float


class ExploreSeeder:
    """Seeds the EXPLORE field: the stale cells of the robot's block, one seed per other block.

    Attributes:
        config: The parameters of the field.
        window_blocks: From ``config``; read by the field provider.
    """

    name: str = "explore"

    def __init__(self, config: ExploreConfig | None = None) -> None:
        self.config: ExploreConfig = ExploreConfig()
        if config is not None:
            self.config = config
        self.window_blocks: int = self.config.window_blocks

    def field_key(self, memory: MemoryView, robot: CellId, now: Time) -> Hashable:
        """Changes when the robot changes block, when a cell of its block is swept or goes stale,
        and every ``bucket``.

        With ``spread_peers`` it also changes when the peers' blocks do. The mean values of the
        other blocks drift in between; the bucket bounds for how long.
        """
        config = self.config
        cells = _cells_of_block(config.block_of(robot), config.block_side)
        own_values = cell_values(
            memory.last_observed_times()[cells],
            memory.monitorable_cells()[cells],
            now,
            config.revisit_age,
        )
        stale = own_values > 0.0
        peers = frozenset(self._peers_per_block(memory, now, config.block_side).items())
        return (now // config.bucket, config.block_of(robot), stale.tobytes(), peers)

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        """A seed on every stale cell of the robot's block, and one on every other block.

        A stale cell starts at ``-beta * v``; another block starts at ``-beta * v`` on its
        stalest cell among those no finer block has claimed. With
        ``spread_peers``, both are divided by one plus the peers of their block.
        """
        config = self.config
        monitorable = memory.monitorable_cells()
        values = cell_values(memory.last_observed_times(), monitorable, now, config.revisit_age)

        own = config.block_of(robot)
        own_cells = _cells_of_block(own, config.block_side)
        own_values = values[own_cells]
        own_peers = self._peers_per_block(memory, now, config.block_side)[own]
        seeds: list[Seed] = []
        for dx, dy in np.argwhere(own_values > 0.0).tolist():
            worth = float(own_values[dx, dy]) / (1 + own_peers)
            cell = (own[0] * config.block_side + dx, own[1] * config.block_side + dy)
            seeds.append(Seed(cell, -config.beta * worth))

        claimed = np.zeros(values.shape, dtype=np.bool_)
        claimed[own_cells] = True
        for level, side in enumerate(config.block_sides):
            region = ~claimed
            if level < len(config.block_sides) - 1:
                region &= _near_cells(robot, side, config.near_blocks, values.shape)
            peers = self._peers_per_block(memory, now, side)
            for block in block_values(np.where(region, values, 0.0), side):
                worth = block.worth / (1 + peers[block.block])
                seeds.append(Seed(block.stalest, -config.beta * worth))
            claimed |= region
        return tuple(seeds)

    def _peers_per_block(self, memory: MemoryView, now: Time, side: int) -> Counter[CellId]:
        """The peers per block of ``side`` cells with ``spread_peers``, else none."""
        if not self.config.spread_peers:
            return Counter()
        return peers_per_block(memory, now, self.config.peer_max_age, side)


def cell_values(
    last_seen: NDArray[np.float64], monitorable: NDArray[np.bool_], now: Time, revisit_age: Time
) -> NDArray[np.float64]:
    """What observing each cell again is worth: ``1 - 2^(-age / revisit_age)`` once it is stale.

    A cell is worth 0 younger than ``revisit_age``, or if it is not monitorable, and 1 if it was
    never observed (``-inf``).
    """
    age = now - last_seen
    values = 1.0 - np.exp2(-age / revisit_age)
    values[age < revisit_age] = 0.0
    values[~monitorable] = 0.0
    return values


def block_values(values: NDArray[np.float64], side: int) -> list[BlockValue]:
    """Every block with something to explore: its stalest cell and that cell's value.

    Blocks are the aligned squares of ``side`` cells; those along the far edges may be cut off by
    the border. All blocks are reduced at once, by padding the values to whole blocks and
    reshaping them to one row of ``side * side`` cells per block.
    """
    width, height = values.shape
    padding = ((0, -width % side), (0, -height % side))
    cells = _cells_per_block(np.pad(values, padding), side)
    stalest = cells.argmax(axis=2)
    worths = cells.max(axis=2)

    found: list[BlockValue] = []
    for block_x, block_y in np.argwhere(worths > 0.0).tolist():
        offset = int(stalest[block_x, block_y])
        cell = (block_x * side + offset // side, block_y * side + offset % side)
        found.append(BlockValue((block_x, block_y), cell, float(worths[block_x, block_y])))
    return found


def peers_per_block(memory: MemoryView, now: Time, max_age: Time, side: int) -> Counter[CellId]:
    """How many peers are in, or heading to, each block of ``side`` cells.

    Only reports younger than ``max_age`` count. A peer counts once per block, also when it is
    heading to a cell of the block it is in.
    """
    counts: Counter[CellId] = Counter()
    for peer in memory.peers():
        if now - peer.observed_at > max_age:
            continue
        blocks = {block_containing(peer.cell, side)}
        if peer.target is not None:
            blocks.add(block_containing(peer.target, side))
        counts.update(blocks)
    return counts


def block_containing(cell: CellId, side: int) -> CellId:
    """The block of ``side`` cells containing ``cell``, in block coordinates."""
    return (cell[0] // side, cell[1] // side)


def _cells_of_block(block: CellId, side: int) -> tuple[slice, slice]:
    """The cells of ``block``, as slices of an array indexed ``[x, y]``."""
    return (
        slice(block[0] * side, (block[0] + 1) * side),
        slice(block[1] * side, (block[1] + 1) * side),
    )


def _near_cells(
    robot: CellId, side: int, near_blocks: int, shape: tuple[int, ...]
) -> NDArray[np.bool_]:
    """The cells of the blocks of ``side`` cells within ``near_blocks`` of the robot's."""
    block_x, block_y = block_containing(robot, side)
    near = np.zeros(shape, dtype=np.bool_)
    near[
        max(0, (block_x - near_blocks) * side) : (block_x + near_blocks + 1) * side,
        max(0, (block_y - near_blocks) * side) : (block_y + near_blocks + 1) * side,
    ] = True
    return near


def _cells_per_block[T: np.generic](cells: NDArray[T], side: int) -> NDArray[T]:
    """Reshape cells ``[x, y]`` to ``[block_x, block_y, dx * side + dy]``.

    The shape of ``cells`` must be a multiple of ``side``.
    """
    blocks_x = cells.shape[0] // side
    blocks_y = cells.shape[1] // side
    by_block = cells.reshape(blocks_x, side, blocks_y, side).transpose(0, 2, 1, 3)
    return by_block.reshape(blocks_x, blocks_y, side * side)
