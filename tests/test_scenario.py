"""Tests for the swarm scenario: building a swarm from presets, their brains, starts and maps."""

from collections import deque
from dataclasses import replace

import pytest

from swarm_field.contracts import BodyType, CellId, MessageKind, RobotSelfState, TimeUnit
from swarm_field.experiments import (
    DEFAULT_EXPLORERS,
    DEFAULT_KILLERS,
    DEFAULT_ROBOTS,
    EXPLORER,
    HOP_BEST,
    HOP_KILLER,
    LATTICE_ROBOTS,
    OPEN_LARGE,
    OPEN_SMALL,
    RELAY,
    RELAYED_KILLERS,
    RIVER_SMALL,
    RIVERS_LARGE,
    SEEDED_BEST,
    SEEDED_KILLER,
    SMALL_FIELD_PESTS,
    CustomPreset,
    ExplorerPreset,
    LiveRun,
    MapConfig,
    SeededKillerPreset,
    SwarmScenario,
)
from swarm_field.runner import BrainSpec
from swarm_field.swarm import (
    AgentMemory,
    BeaconConfig,
    DecisionContext,
    ExploreConfig,
    KnowledgeSharing,
    Resupply,
)

_OPEN = SwarmScenario(map=MapConfig(size=16, obstacles=()), until=TimeUnit.MINUTE)


def test_lattice_starts_are_inside_the_field_and_reproducible() -> None:
    starts = RELAYED_KILLERS.starts()

    relays = RELAYED_KILLERS.count(RELAY.name)
    assert len(starts) == relays + RELAYED_KILLERS.killers
    assert len(set(starts[relays:])) == RELAYED_KILLERS.killers
    for cell in starts:
        assert RELAYED_KILLERS.map.is_inside(cell)
    assert RELAYED_KILLERS.starts() == starts
    assert replace(RELAYED_KILLERS, seed=1).starts() != starts


def test_lattice_starts_avoid_obstacles() -> None:
    scenario = (
        SwarmScenario()
        .with_map(MapConfig())
        .with_robot_config(LATTICE_ROBOTS)
        .add_robots(RELAY, 64)
        .add_robots(SEEDED_KILLER, 1)
    )
    blocked = set(scenario.map.blocked_cells())

    for cell in scenario.starts():
        assert cell not in blocked


def test_initial_pests_lie_on_distinct_free_cells_drawn_from_the_seed() -> None:
    scenario = SwarmScenario().with_pests(replace(SMALL_FIELD_PESTS, initial_pests=20))
    blocked = set(scenario.map.blocked_cells())

    infestation = scenario.initial_infestation()

    assert len(infestation) == 20
    for cell, severity in infestation.items():
        assert scenario.map.is_inside(cell) and cell not in blocked
        assert severity == SMALL_FIELD_PESTS.spawn_severity
    assert scenario.initial_infestation() == infestation
    assert replace(scenario, seed=1).initial_infestation() != infestation
    assert scenario.into_experiment().scenario.initial_infestation == infestation


def test_the_default_scenarios_run_the_default_swarm_on_static_pests() -> None:
    for scenario, hours in ((RIVER_SMALL, 2), (OPEN_SMALL, 2), (RIVERS_LARGE, 4), (OPEN_LARGE, 4)):
        assert scenario.count(EXPLORER.name) == DEFAULT_EXPLORERS
        assert scenario.count(SEEDED_BEST.name) == DEFAULT_KILLERS
        assert scenario.robots == DEFAULT_ROBOTS
        assert scenario.pests.growth_rate == 0.0 and scenario.pests.spread_probability == 0.0
        assert len(scenario.initial_infestation()) == scenario.pests.initial_pests > 0
        assert scenario.until == hours * TimeUnit.HOUR
        assert scenario.metrics.period == TimeUnit.MINUTE
        assert scenario.metrics.measure_from == 0
        hop = scenario.without_robots(BodyType.KILLER).add_robots(HOP_BEST, DEFAULT_KILLERS)
        assert hop.count(EXPLORER.name) == DEFAULT_EXPLORERS
        assert hop.into_experiment().scenario.dock_beacons is not None


def test_the_bridges_join_every_free_cell_of_the_two_rivers_field() -> None:
    size = RIVERS_LARGE.map.size
    blocked = set(RIVERS_LARGE.map.blocked_cells())
    start = (0, 0)
    reached: set[CellId] = {start}
    queue: deque[CellId] = deque([start])
    while queue:
        x, y = queue.popleft()
        for cell in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if RIVERS_LARGE.map.is_inside(cell) and cell not in blocked and cell not in reached:
                reached.add(cell)
                queue.append(cell)

    assert len(reached) == size * size - len(blocked)


def test_robots_are_added_in_order_each_with_the_brain_of_its_preset() -> None:
    scenario = (
        _OPEN.add_robots(EXPLORER, 2)
        .add_robots(RELAY, 1)
        .add_robots(SEEDED_KILLER, 1)
        .add_robots(HOP_KILLER, 2)
    )

    config = scenario.into_experiment()

    bodies = [robot.body for robot in config.scenario.robots]
    assert bodies == [BodyType.SCOUT] * 3 + [BodyType.KILLER] * 3
    for robot, preset in zip(config.scenario.robots, scenario.presets(), strict=True):
        assert config.brain_of(robot.robot_id, robot.body) == scenario.brain(preset)
    assert scenario.without_robots(BodyType.KILLER).count() == 3


def test_hop_killers_and_explorers_run_together_each_with_its_own_memory() -> None:
    scenario = _OPEN.add_robots(EXPLORER, 3).add_robots(HOP_KILLER, 2)
    run = LiveRun.from_config(scenario.into_experiment())

    run.advance_to(scenario.until)

    for robot_id, preset in enumerate(scenario.presets(), start=1):
        agent = run.agents()[robot_id]
        assert agent.beacon_link is not None
        assert agent.memory.keeps_map is (preset is EXPLORER)


def test_only_a_swarm_that_relays_dock_beacons_gets_the_docks_radio() -> None:
    seeded = _OPEN.add_robots(EXPLORER, 2).add_robots(SEEDED_KILLER, 1)
    hop = _OPEN.add_robots(EXPLORER, 2).add_robots(HOP_KILLER, 1)
    no_docks = hop.with_map(replace(hop.map, docks=()))

    assert seeded.into_experiment().scenario.dock_beacons is None
    spec = hop.into_experiment().scenario.dock_beacons
    assert spec is not None and spec.radio_range_m == hop.robots.radio_range_m
    assert no_docks.into_experiment().scenario.dock_beacons is None
    assert "dock-hop" in hop.brain(HOP_KILLER).seeders


def test_a_hop_killer_restocks_only_an_empty_tank() -> None:
    hop = _OPEN.add_robots(EXPLORER, 2).add_robots(HOP_KILLER, 1)
    resupply = Resupply(docks=((0, 0),), config=hop.brain(HOP_KILLER).params["resupply"])
    memory = AgentMemory(1, hop.map.field_geometry(), keeps_map=False)

    def bids(doses_left: int) -> int:
        killer = RobotSelfState(
            robot_id=1,
            position=(5, 5),
            battery=1.0,
            body=BodyType.KILLER,
            speed_mps=0.5,
            consumable=doses_left / 10,
            health=1.0,
            doses_left=doses_left,
        )
        ctx = DecisionContext(time=0, self_state=killer, sensed=[], memory=memory)
        return len(resupply.propose(ctx))

    assert bids(1) == 0
    assert bids(0) == 1


def test_the_best_hop_killer_keeps_its_tuned_times_and_scales_its_hops_with_the_radio() -> None:
    near = _OPEN.add_robots(EXPLORER, 2).add_robots(HOP_BEST, 1)
    far = near.with_robot_config(replace(near.robots, radio_range_m=100.0))

    beacons = near.relaying().beacons
    wider = far.relaying().beacons
    assert beacons is not None and wider is not None
    assert beacons.fresh_for == beacons.source_for == 3 * TimeUnit.MINUTE
    assert beacons.hop_cells == near.robots.radio_range_m / near.map.cell_size
    assert wider.hop_cells == 100.0 / near.map.cell_size


def test_relays_pass_on_what_the_killers_read() -> None:
    seeded = _OPEN.add_robots(RELAY, 4).add_robots(SEEDED_KILLER, 2)
    without_claims = _OPEN.add_robots(RELAY, 4).add_robots(SeededKillerPreset(claims=False), 2)
    hop = _OPEN.add_robots(RELAY, 4).add_robots(HOP_KILLER, 2)

    assert seeded.brain(RELAY).sharing.facts is KnowledgeSharing.INFESTATIONS_AND_TARGETS
    assert seeded.brain(RELAY).beacons is None
    assert without_claims.brain(RELAY).sharing.facts is KnowledgeSharing.INFESTATIONS
    assert hop.brain(RELAY).sharing.facts is KnowledgeSharing.NOTHING
    assert not hop.brain(RELAY).keeps_map
    assert hop.brain(RELAY).beacons == hop.relaying().beacons


def test_a_custom_preset_with_beacons_makes_the_swarm_relay_them() -> None:
    beacons = BeaconConfig(hop_cells=2.0)
    custom = CustomPreset(
        "beacon-killer",
        BodyType.KILLER,
        BrainSpec(behaviors=("treat-hop",), seeders=("treat-hop",), beacons=beacons),
    )

    scenario = _OPEN.add_robots(EXPLORER, 2).add_robots(custom, 1)

    assert scenario.brain(EXPLORER).beacons == beacons
    assert scenario.brain(custom) == custom.brain_spec
    assert scenario.count("beacon-killer") == 1


def test_presets_that_need_different_beacons_are_refused() -> None:
    custom = CustomPreset("other", BodyType.KILLER, BrainSpec(beacons=BeaconConfig(gamma=0.9)))

    with pytest.raises(ValueError):
        _OPEN.add_robots(HOP_KILLER, 1).add_robots(custom, 1).relaying()


def test_mock_pest_sources_send_only_what_the_killers_read() -> None:
    relays = RELAYED_KILLERS.without_robots(BodyType.KILLER)

    seeded = relays.add_robots(SEEDED_KILLER, 2).pest_sources()
    hop = relays.add_robots(HOP_KILLER, 2).pest_sources()
    nobody = relays.pest_sources()

    assert seeded is not None and seeded.kinds == {MessageKind.KNOWLEDGE}
    assert hop is not None and hop.kinds == {MessageKind.BEACON}
    assert nobody is None


def test_explorers_revisit_after_half_the_time_their_swarm_needs_to_sweep_the_field() -> None:
    # 256 free cells, 4 explorers; a 10 m move at 2 m/s takes 5 s, and the next starts on arrival.
    scenario = _OPEN.add_robots(EXPLORER, 4).add_robots(SEEDED_KILLER, 1)
    sweep = 256 * 5 * TimeUnit.SECOND // 4

    assert scenario.sweep_time() == sweep
    assert scenario.brain(EXPLORER).params["explore"].revisit_age == sweep // 2
    assert scenario.add_robots(EXPLORER, 4).sweep_time() == sweep // 2


def test_an_explorer_without_revisit_sweeps_keeps_its_own_revisit_age() -> None:
    fixed = ExplorerPreset(explore=ExploreConfig(revisit_age=TimeUnit.HOUR), revisit_sweeps=None)

    scenario = _OPEN.add_robots(fixed, 4)

    assert scenario.brain(fixed).params["explore"].revisit_age == TimeUnit.HOUR


def test_a_preset_with_no_robots_adds_no_brain() -> None:
    # A hop killer's brain needs relayed beacons, which a swarm without hop killers lacks.
    scenario = (
        SwarmScenario()
        .add_robots(EXPLORER, 2)
        .add_robots(SEEDED_KILLER, 2)
        .add_robots(HOP_KILLER, 0)
    )

    config = scenario.into_experiment()

    assert len(config.scenario.robots) == 4


def test_a_swarm_larger_than_the_bottom_row_shares_its_start_cells() -> None:
    scenario = SwarmScenario(map=MapConfig(size=4, obstacles=())).add_robots(EXPLORER, 6)

    assert scenario.starts() == [(0, 0), (1, 0), (2, 0), (3, 0), (0, 0), (1, 0)]
