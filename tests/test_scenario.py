"""Tests for the swarm scenario: building a swarm from presets, their brains, starts and maps."""

from collections import deque
from dataclasses import replace

import pytest

from swarm_field.contracts import BodyType, CellId, MessageKind, TimeUnit
from swarm_field.experiments import (
    EXPLORER,
    HOP_KILLER,
    RELAY,
    RELAYED_KILLERS,
    RIVERS_LARGE,
    SEEDED_KILLER,
    CustomPreset,
    ExplorerPreset,
    LiveRun,
    MapConfig,
    SeededKillerPreset,
    SwarmScenario,
    pest_response_configurations,
)
from swarm_field.runner import BrainSpec
from swarm_field.swarm import BeaconConfig, ExploreConfig, KnowledgeSharing

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
        RELAYED_KILLERS.without_robots()
        .with_map(MapConfig())
        .add_robots(RELAY, 64)
        .add_robots(SEEDED_KILLER, 1)
    )
    blocked = set(scenario.map.blocked_cells())

    for cell in scenario.starts():
        assert cell not in blocked


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
    configurations = pest_response_configurations(RELAYED_KILLERS)

    seeded = configurations["seeded"].pest_sources()
    hop = configurations["hop-tuned"].pest_sources()
    nobody = RELAYED_KILLERS.without_robots(BodyType.KILLER).pest_sources()

    assert seeded is not None and seeded.kinds == {MessageKind.KNOWLEDGE}
    assert hop is not None and hop.kinds == {MessageKind.BEACON}
    assert nobody is None


def test_the_pest_responses_keep_the_number_of_killers() -> None:
    configurations = pest_response_configurations(RELAYED_KILLERS, seeded_no_yield=True)

    assert list(configurations) == ["seeded", "seeded-no-yield", "hop", "hop-repel", "hop-tuned"]
    for configured in configurations.values():
        assert configured.killers == RELAYED_KILLERS.killers
        assert configured.count(RELAY.name) == RELAYED_KILLERS.count(RELAY.name)
    assert configurations["hop-tuned"].count(HOP_KILLER.name) == RELAYED_KILLERS.killers


def test_explorers_revisit_after_half_the_time_their_swarm_needs_to_sweep_the_field() -> None:
    # 256 free cells, 4 explorers; a 10 m move at 2 m/s takes 5 s, a step a whole 15 s interval.
    scenario = _OPEN.add_robots(EXPLORER, 4).add_robots(SEEDED_KILLER, 1)
    sweep = 256 * 15 * TimeUnit.SECOND // 4

    assert scenario.sweep_time() == sweep
    assert scenario.brain(EXPLORER).params["explore"].revisit_age == sweep // 2
    assert scenario.add_robots(EXPLORER, 4).sweep_time() == sweep // 2


def test_an_explorer_without_revisit_sweeps_keeps_its_own_revisit_age() -> None:
    fixed = ExplorerPreset(explore=ExploreConfig(revisit_age=TimeUnit.HOUR), revisit_sweeps=None)

    scenario = _OPEN.add_robots(fixed, 4)

    assert scenario.brain(fixed).params["explore"].revisit_age == TimeUnit.HOUR
