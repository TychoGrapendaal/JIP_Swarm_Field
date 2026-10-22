"""The EXPLORE seeder: draws robots to the cells whose information is the stalest.

A cell is worth observing again by how long ago it was last observed. It is worth nothing until
``revisit_age`` ``A`` has passed; then it is stale, worth ``1 - 2^(-age / A)``: 0.5 at the revisit
age, approaching 1, and 1 if it was never observed. So exploring a new field and monitoring a
known one are the same objective.

The robot's own block is seeded cell by cell, with its stale cells; every other block gets one
seed, on its stalest cell and worth that cell's value, so a forgotten cell is never hidden in a
block of fresh ones. One field then does both jobs: inside the robot's block it leads to the
nearest stale cell, and once the block has no stale cell left it leads to the best other block.
A robot observes only the cell it stands on, so every swept cell is a target used up; the key
follows the stale cells of the robot's block, and the field is rebuilt once per swept cell, but
not while the robot travels over fresh ground.

With ``spread_peers`` the robots are made to spread out: a block's values are divided by one
plus the number of peers in it or heading to it.
"""

from collections import Counter
from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import CellId, Time, TimeUnit
from swarm_field.swarm.field import Seed
from swarm_field.swarm.memory import MemoryView


@dataclass(frozen=True)
class ExploreConfig:
    """The parameters of the EXPLORE field.

    The defaults did best in ``studies/explore_variants.py`` (see ``docs/explore.md``) for five
    robots on 32 x 32 and 64 x 64 fields.

    Attributes:
        revisit_age: How long a cell is worth nothing after it was observed. From then on it is
            stale, worth ``1 - 2^(-age / revisit_age)``, and only the stale cells of the robot's
            own block are seeded; the block is done when none is left.
        beta: How many cells of travel a fully stale cell, or a fully stale block, is worth.
        rebuild_period: How long the worths of the other blocks may drift before the field is
            rebuilt anyway.
        block_side: The side of a block in cells.
        spread_peers: Whether to divide a block's worths by one plus the number of peers in it
            or heading to it.
        peer_max_age: Peer reports older than this are left out of the discount.
        window_blocks: How many blocks of ``Quadtree.max_leaf_side`` cells around the robot's
            block the field is searched cell by cell. The sweep needs only the robot's own
            block, but beyond the window a block is one leaf, measured from its centre, which
            makes the neighbouring blocks look farther than they are; 1 keeps them exact.
    """

    revisit_age: Time = 28 * TimeUnit.MINUTE
    beta: float = 64.0
    rebuild_period: Time = 5 * TimeUnit.MINUTE
    block_side: int = 8
    spread_peers: bool = True
    peer_max_age: Time = TimeUnit.MINUTE
    window_blocks: int = 1

    def block_of(self, cell: CellId) -> CellId:
        """The block containing ``cell``, in block coordinates."""
        return (cell[0] // self.block_side, cell[1] // self.block_side)

    def cell_in_block(self, block: CellId, dx: int, dy: int) -> CellId:
        """The cell ``dx`` and ``dy`` cells from the corner of ``block``."""
        return (block[0] * self.block_side + dx, block[1] * self.block_side + dy)

    def cells_of_block(self, block: CellId) -> tuple[slice, slice]:
        """The cells of ``block``, as slices of an array indexed ``[x, y]``."""
        side = self.block_side
        return (
            slice(block[0] * side, (block[0] + 1) * side),
            slice(block[1] * side, (block[1] + 1) * side),
        )


@dataclass(frozen=True)
class BlockTarget:
    """Where one block's seed goes, and what visiting it is worth.

    Attributes:
        block: The block, in block coordinates.
        cell: The block's stalest cell.
        worth: What revisiting that cell is worth.
    """

    block: CellId
    cell: CellId
    worth: float


@dataclass(frozen=True)
class PeerDiscount:
    """Divides a target's worth by one plus the number of peers in, or heading to, its block.

    Attributes:
        peers: How many peers are in, or heading to, each block, in block coordinates. Empty
            without ``spread_peers``, so the discount changes nothing.
    """

    peers: Mapping[CellId, int]

    @classmethod
    def from_memory(cls, memory: MemoryView, now: Time, config: ExploreConfig) -> "PeerDiscount":
        """The discount from the peers' recent reports, or none without ``spread_peers``.

        Only reports younger than ``peer_max_age`` count. A peer counts once per block, also when
        it is heading to a cell of the block it is in.
        """
        counts: Counter[CellId] = Counter()
        if not config.spread_peers:
            return cls(counts)
        for peer in memory.peers():
            if now - peer.observed_at > config.peer_max_age:
                continue
            blocks = {config.block_of(peer.cell)}
            if peer.target is not None:
                blocks.add(config.block_of(peer.target))
            counts.update(blocks)
        return cls(counts)

    def apply(self, worth: float, block: CellId) -> float:
        """The ``worth`` of a target in ``block``, after the discount."""
        return worth / (1 + self.peers.get(block, 0))


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
        and every ``rebuild_period``.

        With ``spread_peers`` it also changes when the peers' blocks do. The worths of the other
        blocks drift in between; the rebuild period bounds for how long.
        """
        config = self.config
        own_block = config.block_of(robot)
        cells = config.cells_of_block(own_block)
        own_worths = revisit_worths(
            memory.last_observed_times()[cells],
            memory.monitorable_cells()[cells],
            now,
            config.revisit_age,
        )
        stale = own_worths > 0.0
        peers = frozenset(PeerDiscount.from_memory(memory, now, config).peers.items())
        return (now // config.rebuild_period, own_block, stale.tobytes(), peers)

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        """A seed on every stale cell of the robot's block, and one on every other block.

        A stale cell starts at ``-beta * worth``, and another block at ``-beta * worth`` of its
        stalest cell. With ``spread_peers``, both are divided by one plus the peers of their
        block.
        """
        config = self.config
        worths = revisit_worths(
            memory.last_observed_times(), memory.monitorable_cells(), now, config.revisit_age
        )
        # The peer discount: each worth divided by one plus the peers of its block. Without
        # spread_peers it changes nothing.
        discount = PeerDiscount.from_memory(memory, now, config)

        # The robot's own block: one seed on every stale cell.
        own_block = config.block_of(robot)
        own_worths = worths[config.cells_of_block(own_block)]
        seeds: list[Seed] = []
        for dx, dy in np.argwhere(own_worths > 0.0).tolist():
            worth = discount.apply(float(own_worths[dx, dy]), own_block)
            seeds.append(Seed(config.cell_in_block(own_block, dx, dy), -config.beta * worth))

        # Every other block: one seed on its stalest cell.
        for target in block_targets(worths, config):
            if target.block != own_block:
                worth = discount.apply(target.worth, target.block)
                seeds.append(Seed(target.cell, -config.beta * worth))
        return tuple(seeds)


def revisit_worths(
    last_seen: NDArray[np.float64], monitorable: NDArray[np.bool_], now: Time, revisit_age: Time
) -> NDArray[np.float64]:
    """What revisiting each cell is worth: ``1 - 2^(-age / revisit_age)`` once it is stale.

    A cell is worth 0 younger than ``revisit_age``, or if it is not monitorable, and 1 if it was
    never observed (``-inf``).
    """
    age = now - last_seen
    worths = 1.0 - np.exp2(-age / revisit_age)
    worths[age < revisit_age] = 0.0
    worths[~monitorable] = 0.0
    return worths


def block_targets(worths: NDArray[np.float64], config: ExploreConfig) -> list[BlockTarget]:
    """Every block with something to explore: its stalest cell and what revisiting it is worth.

    Blocks are the aligned squares of ``block_side`` cells; those along the far edges may be cut
    off by the border. All blocks are reduced at once, by padding the worths to whole blocks and
    reshaping them to one row of ``block_side * block_side`` cells per block.
    """
    side = config.block_side
    width, height = worths.shape
    padding = ((0, -width % side), (0, -height % side))
    cells = _cells_per_block(np.pad(worths, padding), side)
    stalest = cells.argmax(axis=2)
    best = cells.max(axis=2)

    found: list[BlockTarget] = []
    for block_x, block_y in np.argwhere(best > 0.0).tolist():
        block = (block_x, block_y)
        dx, dy = divmod(int(stalest[block]), side)
        found.append(BlockTarget(block, config.cell_in_block(block, dx, dy), float(best[block])))
    return found


def _cells_per_block[T: np.generic](cells: NDArray[T], side: int) -> NDArray[T]:
    """Reshape cells ``[x, y]`` to ``[block_x, block_y, dx * side + dy]``.

    The shape of ``cells`` must be a multiple of ``side``.
    """
    blocks_x = cells.shape[0] // side
    blocks_y = cells.shape[1] // side
    by_block = cells.reshape(blocks_x, side, blocks_y, side).transpose(0, 2, 1, 3)
    return by_block.reshape(blocks_x, blocks_y, side * side)
