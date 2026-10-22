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

With ``yield_to_peers`` robots that stand together split up. Two robots on the same cell share
their memory, so they see the same field and would take the same step forever: the discount
lowers a block equally for both and cannot tell them apart. So of the robots next to each
other the one later in the order of ``swarm.priority`` gives way: it leaves out the target that
a neighbour before it reported heading for, and takes its next best one.
"""

from collections import Counter
from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from swarm_field.contracts import CellId, Time, TimeUnit
from swarm_field.swarm.field import Seed
from swarm_field.swarm.memory import MemoryView
from swarm_field.swarm.priority import ranks_before


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
        yield_to_peers: Whether a robot gives way to a peer next to it that goes before it
            (``swarm.priority``), by leaving out the target that peer reported heading for.
        yield_distance: How close, in cells along either axis, a peer must be to count as next
            to the robot; 0 is the same cell only.
        yield_spacing: The cells of the robot's own block this close to a target it gives way
            on are left out with it, so the two robots sweep different parts of the block.
        reach_cells: With robots kept in radio contact (``swarm.tether``), the link range in
            cells. The stale cells within it of a peer are then seeded too, and when there are
            any, only cells within reach are: a robot always has a target it can get to without
            losing contact, and the robots spread out together. ``None`` seeds as usual.
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
    yield_to_peers: bool = True
    yield_distance: int = 1
    yield_spacing: int = 3
    reach_cells: float | None = None
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


def yielded_targets(
    memory: MemoryView, robot: CellId, now: Time, config: ExploreConfig
) -> frozenset[CellId]:
    """The targets this robot gives way on: those of fresh peers next to it that go first.

    Empty without ``yield_to_peers``. Every robot applies the same rule, so of two robots
    standing together exactly one gives way, and they do not both turn away.
    """
    if not config.yield_to_peers:
        return frozenset()
    targets: set[CellId] = set()
    for peer in memory.peers():
        if peer.peer is None or peer.target is None:
            continue
        if not ranks_before(peer.peer, memory.robot_id, now):
            continue
        if now - peer.observed_at > config.peer_max_age:
            continue
        apart = max(abs(peer.cell[0] - robot[0]), abs(peer.cell[1] - robot[1]))
        if apart <= config.yield_distance:
            targets.add(peer.target)
    return frozenset(targets)


def _near_any(cell: CellId, targets: frozenset[CellId], spacing: int) -> bool:
    """Whether ``cell`` is within ``spacing`` cells, along both axes, of one of ``targets``."""
    for target in targets:
        if abs(cell[0] - target[0]) <= spacing and abs(cell[1] - target[1]) <= spacing:
            return True
    return False


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
        yielded = yielded_targets(memory, robot, now, config)
        return (now // config.rebuild_period, own_block, stale.tobytes(), peers, yielded)

    def seeds(self, memory: MemoryView, robot: CellId, now: Time) -> tuple[Seed, ...]:
        """A seed on every stale cell of the robot's block, and one on every other block.

        A stale cell starts at ``-beta * worth``, and another block at ``-beta * worth`` of its
        stalest cell. With ``spread_peers``, both are divided by one plus the peers of their
        block. With ``yield_to_peers``, the targets the robot gives way on are left out: the
        stale cells of its block near them, and the other blocks that hold them; if that leaves
        nothing, nothing is left out.
        """
        config = self.config
        worths = revisit_worths(
            memory.last_observed_times(), memory.monitorable_cells(), now, config.revisit_age
        )
        # The peer discount: each worth divided by one plus the peers of its block. Without
        # spread_peers it changes nothing.
        discount = PeerDiscount.from_memory(memory, now, config)

        yielded = yielded_targets(memory, robot, now, config)
        yielded_blocks = {config.block_of(target) for target in yielded}

        # The robot's own block: one seed on every stale cell.
        own_block = config.block_of(robot)
        own_worths = worths[config.cells_of_block(own_block)]
        seeds: list[Seed] = []
        kept: list[Seed] = []
        for dx, dy in np.argwhere(own_worths > 0.0).tolist():
            worth = discount.apply(float(own_worths[dx, dy]), own_block)
            seed = Seed(config.cell_in_block(own_block, dx, dy), -config.beta * worth)
            seeds.append(seed)
            if not _near_any(seed.cell, yielded, config.yield_spacing):
                kept.append(seed)

        # Every other block: one seed on its stalest cell.
        for target in block_targets(worths, config):
            if target.block != own_block:
                worth = discount.apply(target.worth, target.block)
                seed = Seed(target.cell, -config.beta * worth)
                seeds.append(seed)
                if target.block not in yielded_blocks:
                    kept.append(seed)
        within = _seeds_within_reach(memory, now, worths, discount, kept, config)
        if within:
            return within
        if kept:
            return tuple(kept)
        return tuple(seeds)


def _seeds_within_reach(
    memory: MemoryView,
    now: Time,
    worths: NDArray[np.float64],
    discount: PeerDiscount,
    seeds: list[Seed],
    config: ExploreConfig,
) -> tuple[Seed, ...]:
    """The stale cells within ``reach_cells`` of a fresh peer, and those of ``seeds`` in reach.

    Empty without ``reach_cells``, or when no peer is known or nothing stale is in reach.
    """
    reach = config.reach_cells
    if reach is None:
        return ()
    peers: list[CellId] = []
    for peer in memory.peers():
        if peer.peer is not None and now - peer.observed_at <= config.peer_max_age:
            peers.append(peer.cell)
    if not peers:
        return ()
    width, height = worths.shape
    xs, ys = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
    in_reach = np.zeros(worths.shape, dtype=np.bool_)
    for x, y in peers:
        in_reach |= (xs - x) ** 2 + (ys - y) ** 2 <= reach * reach
    found: dict[CellId, Seed] = {}
    for seed in seeds:
        if in_reach[seed.cell]:
            found[seed.cell] = seed
    for x, y in np.argwhere(in_reach & (worths > 0.0)).tolist():
        cell = (x, y)
        if cell not in found:
            worth = discount.apply(float(worths[x, y]), config.block_of(cell))
            found[cell] = Seed(cell, -config.beta * worth)
    return tuple(found.values())


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
