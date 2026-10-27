"""Tests for role rebalancing between monitors and killers (#64)."""

from dataclasses import replace

from swarm_field.contracts import (
    AgentObjective,
    BodyType,
    FieldGeometry,
    InfestationObservation,
    PeerObservation,
    RobotSelfState,
    RoleSwitchSpec,
    SwitchRole,
    TimeUnit,
)
from swarm_field.simulation import RobotState, WorldState
from swarm_field.swarm import (
    AgentMemory,
    BodyGate,
    DecisionContext,
    FollowField,
    RebalanceConfig,
    RebalanceRoles,
    TreatInfestation,
)

_CONFIG = RebalanceConfig(
    min_killers=1, min_monitors=1, pests_per_killer=1.0, persistence=0, role_dwell=TimeUnit.HOUR
)


def _memory(
    robot: int, monitors: dict[int, tuple[int, int]], pests: list[tuple[int, int]]
) -> AgentMemory:
    """A memory that knows of ``monitors`` (id to cell) and of ``pests``."""
    memory = AgentMemory(robot, FieldGeometry(10, 10, 10.0, 0.0, 0.0, 0.0))
    memory.merge(
        PeerObservation(cell, 1.0, 0, peer, peer, BodyType.SCOUT) for peer, cell in monitors.items()
    )
    memory.merge(InfestationObservation(cell, 1.0, 0, robot, 0.5, None) for cell in pests)
    return memory


def _ctx(
    robot: int, cell: tuple[int, int], body: BodyType, memory: AgentMemory, time: int = 0
) -> DecisionContext:
    state = RobotSelfState(robot, cell, 1.0, body, 1.0, 1.0, 1.0)
    return DecisionContext(time=time, self_state=state, sensed=[], memory=memory)


def test_the_monitor_nearest_a_pest_becomes_the_missing_killer() -> None:
    # Three monitors and no killer, one known pest: the swarm needs 1 + 1 = 2 killers.
    monitors = {2: (9, 9), 3: (0, 1)}
    near = _ctx(1, (1, 1), BodyType.SCOUT, _memory(1, monitors, [(1, 2)]))
    far = _ctx(2, (9, 9), BodyType.SCOUT, _memory(2, {1: (1, 1), 3: (0, 1)}, [(1, 2)]))

    (bid,) = RebalanceRoles(_CONFIG).propose(near)

    assert bid.action == SwitchRole(BodyType.KILLER)
    assert bid.score >= 0.9
    assert RebalanceRoles(_CONFIG).propose(far) == ()


def test_a_robot_keeps_its_new_role_for_the_dwell_time() -> None:
    behaviour = RebalanceRoles(replace(_CONFIG, min_killers=0))
    memory = _memory(1, {2: (9, 9), 3: (0, 1)}, [(1, 2)])
    behaviour.propose(_ctx(1, (1, 1), BodyType.SCOUT, memory, time=0))

    # It became a killer; with the pest gone and no killer kept, it is now an excess killer.
    clean = _memory(1, {2: (9, 9), 3: (0, 1)}, [])
    later = _ctx(1, (1, 1), BodyType.KILLER, clean, time=10 * TimeUnit.MINUTE)

    assert behaviour.propose(later) == ()


def test_the_world_gives_a_switching_robot_its_new_speed_and_keeps_it_busy() -> None:
    roles = RoleSwitchSpec(speeds={BodyType.KILLER: 0.5}, duration=60_000)
    world = WorldState(FieldGeometry(5, 5, 10.0, 0.0, 0.0, 0.0), roles=roles)
    world.add_robot(RobotState(1, (0, 0), BodyType.SCOUT, speed_mps=1.5))

    world.apply_action(1, SwitchRole(BodyType.KILLER), time=0)
    busy = world.apply_action(1, SwitchRole(BodyType.SCOUT), time=30_000)

    robot = world.robots[1]
    assert (robot.body, robot.speed_mps, robot.role_switches) == (BodyType.KILLER, 0.5, 1)
    assert busy is None


def test_a_gated_behaviour_bids_only_for_its_bodies() -> None:
    gate = BodyGate(TreatInfestation(), (BodyType.KILLER,))
    memory = _memory(1, {}, [(1, 1)])

    assert gate.propose(_ctx(1, (1, 1), BodyType.SCOUT, memory)) == ()
    assert gate.propose(_ctx(1, (1, 1), BodyType.KILLER, memory)) != ()


_DOCKS = ((0, 0),)


def test_a_robot_picked_to_switch_at_a_dock_first_drives_to_it() -> None:
    config = replace(_CONFIG, switch_at_dock=True)
    memory = _memory(1, {2: (9, 9), 3: (5, 5)}, [(1, 2)])

    (bid,) = RebalanceRoles(config, _DOCKS).propose(_ctx(1, (1, 1), BodyType.SCOUT, memory))

    assert bid.action == FollowField("dock")
    assert bid.objective is AgentObjective.SWITCH_ROLE
    assert bid.score >= 0.9


def test_a_robot_on_the_dock_switches_there() -> None:
    config = replace(_CONFIG, switch_at_dock=True)
    memory = _memory(1, {2: (9, 9), 3: (5, 5)}, [(1, 2)])

    (bid,) = RebalanceRoles(config, _DOCKS).propose(_ctx(1, (0, 0), BodyType.SCOUT, memory))

    assert bid.action == SwitchRole(BodyType.KILLER)
    assert bid.objective is AgentObjective.SWITCH_ROLE


def test_the_robot_nearest_a_dock_is_picked_when_switching_at_a_dock() -> None:
    # The pest is next to robot 1, but robot 2 is nearer the dock.
    config = replace(_CONFIG, switch_at_dock=True)
    near_pest = _ctx(1, (8, 8), BodyType.SCOUT, _memory(1, {2: (1, 0), 3: (5, 5)}, [(8, 9)]))
    near_dock = _ctx(2, (1, 0), BodyType.SCOUT, _memory(2, {1: (8, 8), 3: (5, 5)}, [(8, 9)]))

    assert RebalanceRoles(config, _DOCKS).propose(near_pest) == ()
    assert RebalanceRoles(config, _DOCKS).propose(near_dock) != ()


def test_a_peer_on_its_way_to_switch_already_counts_in_its_new_role() -> None:
    # One pest, so two killers are needed; robot 2 is already driving off to become one.
    config = replace(_CONFIG, switch_at_dock=True, min_killers=1)
    memory = _memory(3, {1: (1, 1)}, [(1, 2)])
    switching = PeerObservation(
        (2, 2), 1.0, 0, 2, 2, BodyType.SCOUT, objective=AgentObjective.SWITCH_ROLE
    )
    memory.merge([switching])
    third = _ctx(3, (0, 1), BodyType.SCOUT, memory)

    # Robot 3 is nearest the dock, but with robot 2 counted as a killer only one more is missing,
    # and robot 3 is the one the ranking picks; without the count it would be two.
    (bid,) = RebalanceRoles(config, _DOCKS).propose(third)
    assert bid.objective is AgentObjective.SWITCH_ROLE

    memory_one_more = _memory(1, {3: (0, 1)}, [(1, 2)])
    memory_one_more.merge([switching])
    first = _ctx(1, (1, 1), BodyType.SCOUT, memory_one_more)
    assert RebalanceRoles(config, _DOCKS).propose(first) == ()


def test_without_docks_a_robot_that_must_switch_at_one_never_switches() -> None:
    config = replace(_CONFIG, switch_at_dock=True)
    memory = _memory(1, {2: (9, 9), 3: (5, 5)}, [(1, 2)])

    assert RebalanceRoles(config, ()).propose(_ctx(1, (1, 1), BodyType.SCOUT, memory)) == ()


def test_a_switch_on_a_dock_hands_the_robot_a_full_tank() -> None:
    field = FieldGeometry(5, 5, 10.0, 0.0, 0.0, 0.0)
    world = WorldState(field, docks=((0, 0),), treatment_doses=10)
    world.add_robot(RobotState(1, (0, 0), BodyType.SCOUT, speed_mps=1.5))
    world.robots[1].doses_left = 2

    world.apply_action(1, SwitchRole(BodyType.KILLER), time=0)

    assert world.robots[1].doses_left == 10
