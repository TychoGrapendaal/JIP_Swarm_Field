"""Tests for keeping robots in radio contact: which steps are safe, and what replaces the rest."""

from swarm_field.contracts import (
    AgentObjective,
    BodyType,
    FieldGeometry,
    Idle,
    Move,
    PeerObservation,
    Terrain,
    TerrainObservation,
    Treat,
)
from swarm_field.swarm import AgentMemory, FieldProvider, TetherConfig, tethered_action
from swarm_field.swarm.field import Seed, build_grid_potential_field

_FIELD = FieldGeometry(width=20, height=20, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)
_CONFIG = TetherConfig(radio_cells=5.0, margin_cells=1.0)
"""Links hold within 4 cells."""


def _memory(*peers: PeerObservation) -> AgentMemory:
    memory = AgentMemory(1, _FIELD)
    terrain: list[TerrainObservation] = []
    for x in range(20):
        for y in range(20):
            fact = TerrainObservation(
                cell=(x, y), confidence=1.0, observed_at=0, source=1, terrain=Terrain.FREE
            )
            terrain.append(fact)
    memory.merge(terrain)
    memory.merge(peers)
    return memory


def _peer(
    peer: int, cell: tuple[int, int], objective: AgentObjective | None = None
) -> PeerObservation:
    return PeerObservation(
        cell=cell,
        confidence=1.0,
        observed_at=0,
        source=peer,
        peer=peer,
        body=BodyType.SCOUT,
        objective=objective,
    )


def test_a_step_that_keeps_a_neighbour_is_taken_unchanged() -> None:
    memory = _memory(_peer(2, (5, 5)))

    assert tethered_action(Move((6, 5)), (5, 5), None, memory, 0, _CONFIG) == Move((6, 5))


def test_a_step_out_of_range_becomes_the_safe_step_nearest_the_goal() -> None:
    memory = _memory(_peer(2, (0, 5)))

    action = tethered_action(Move((5, 5)), (4, 5), (15, 5), memory, 0, _CONFIG)

    assert action in {Idle(), Move((4, 4)), Move((4, 6))}


def test_a_robot_does_not_walk_away_from_a_lone_anchor() -> None:
    anchor = _peer(2, (3, 5), AgentObjective.VERIFY)
    explorer = _peer(3, (9, 5), AgentObjective.EXPLORE)
    memory = _memory(anchor, explorer)

    # Stepping east still keeps robot 3 as a neighbour, but leaves the anchor alone.
    action = tethered_action(Move((8, 5)), (7, 5), (12, 5), memory, 0, _CONFIG)

    assert action == Idle()


def test_a_robot_out_of_contact_heads_back_to_the_nearest_peer() -> None:
    memory = _memory(_peer(2, (0, 0)))

    action = tethered_action(Move((11, 10)), (10, 10), None, memory, 0, _CONFIG)

    assert action == Move((9, 9))


def test_only_steps_are_checked() -> None:
    memory = _memory(_peer(2, (0, 0)))

    assert tethered_action(Treat((10, 10)), (10, 10), None, memory, 0, _CONFIG) == Treat((10, 10))


def test_an_unsafe_step_is_replaced_along_the_field_not_the_straight_line() -> None:
    """Behind a wall the goal is east but the route goes north; the safe step follows the route."""
    memory = _memory(_peer(2, (5, 1)))
    wall: list[TerrainObservation] = []
    for y in range(15):
        fact = TerrainObservation(
            cell=(7, y), confidence=1.0, observed_at=1, source=1, terrain=Terrain.OBSTACLE
        )
        wall.append(fact)
    memory.merge(wall)
    quadtree = FieldProvider(memory, flies=False).quadtree
    potential = build_grid_potential_field((Seed((12, 2), 0.0),), quadtree, (5, 5))

    along_field = tethered_action(Move((5, 6)), (5, 5), (12, 2), memory, 0, _CONFIG, potential)
    straight = tethered_action(Move((5, 6)), (5, 5), (12, 2), memory, 0, _CONFIG)

    assert along_field == Idle()
    assert straight == Move((6, 4))


def test_a_robot_that_has_heard_no_peer_lately_heads_back_to_the_last_one_it_knew() -> None:
    memory = _memory(_peer(2, (0, 0)))

    action = tethered_action(Move((11, 10)), (10, 10), None, memory, 60_000, _CONFIG)

    assert action == Move((9, 9))


def test_a_lost_robot_at_the_last_known_spot_of_a_peer_goes_about_its_work() -> None:
    """The peer has moved on; waiting where it was would strand the robot."""
    memory = _memory(_peer(2, (10, 12)))

    action = tethered_action(Move((11, 10)), (10, 10), None, memory, 60_000, _CONFIG)

    assert action == Move((11, 10))


def test_a_robot_lost_for_long_goes_about_its_work() -> None:
    memory = _memory(_peer(2, (0, 0)))

    action = tethered_action(Move((11, 10)), (10, 10), None, memory, 10 * 60_000, _CONFIG)

    assert action == Move((11, 10))
