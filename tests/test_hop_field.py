"""Tests for the hop field and the pest response on hop beacons."""

from dataclasses import replace
from random import Random

from swarm_field.contracts import (
    Beacon,
    BeaconKind,
    BodyType,
    CellId,
    Charge,
    FieldGeometry,
    InfestationObservation,
    Message,
    MessageKind,
    Move,
    Observation,
    Perception,
    RobotSelfState,
    Terrain,
    TerrainObservation,
    TimeUnit,
    Treat,
)
from swarm_field.runner import BrainSpec, build_agent
from swarm_field.swarm import (
    AgentMemory,
    BeaconConfig,
    BeaconMemory,
    ComposedAgent,
    DecisionContext,
    DockBeacons,
    EscapeConfig,
    FollowField,
    Heading,
    HopDockField,
    HopField,
    KnowledgeSharing,
    ReturnToDock,
    SharingConfig,
    TreatByBeacon,
    treat_beacon,
)
from swarm_field.swarm.hop import cell_ahead, dock_beacon, hop_pull

_FIELD = FieldGeometry(width=10, height=10, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)
_CONFIG = BeaconConfig()
_ESCAPE = EscapeConfig(window=0)
"""An escape that judges a robot stuck at a single decision without progress."""
_MAP = AgentMemory(1, _FIELD)
"""The shared memory the hop field and behaviour are handed, empty: hop rules read beacons."""


def _memory_hearing(*beacons: tuple[float, CellId]) -> BeaconMemory:
    memory = BeaconMemory()
    for sender, (value, cell) in enumerate(beacons, start=2):
        memory.hear(sender, 0, [Beacon(BeaconKind.TREAT, value, cell)])
    return memory


def _heading(config: BeaconConfig, beacons: BeaconMemory, robot: CellId) -> Heading | None:
    return HopField(config, beacons).guidance_for(_MAP, robot, 0)


def _self_state(position: CellId) -> RobotSelfState:
    return RobotSelfState(
        robot_id=1,
        position=position,
        battery=1.0,
        body=BodyType.KILLER,
        speed_mps=1.0,
        consumable=1.0,
        health=1.0,
    )


def test_only_the_lowest_beacons_pull_the_robot() -> None:
    memory = _memory_hearing((1.0, (5, 0)), (2.0, (0, 5)), (3.0, (0, 0)), (3.0, (2, 2)))

    heading = _heading(_CONFIG, memory, (0, 0))

    assert heading is not None
    assert heading.offset == (5.0, 0.0)
    assert heading.target == (5, 0)


def test_beacons_at_the_same_value_pull_together() -> None:
    memory = _memory_hearing((1.0, (4, 0)), (1.0, (0, 4)))

    heading = _heading(_CONFIG, memory, (0, 0))

    assert heading is not None
    assert heading.offset[0] > 0
    assert heading.offset[0] == heading.offset[1]


def test_a_robot_on_a_relay_is_not_pulled_back_to_a_higher_neighbour() -> None:
    memory = _memory_hearing((1.5, (3, 1)), (1.7, (3, 2)))

    assert _heading(BeaconConfig(hop_cells=2.0), memory, (3, 1)) is None


def test_between_two_equal_beacons_a_distance_weighted_pull_leads_to_the_nearer_one() -> None:
    memory = _memory_hearing((0.0, (0, 0)), (0.0, (10, 0)))

    unit = _heading(_CONFIG, memory, (4, 0))
    weighted = _heading(BeaconConfig(distance_power=1.0), memory, (4, 0))

    assert unit is None
    assert weighted is not None
    assert weighted.offset[0] < 0


def _bids(memory: BeaconMemory, position: CellId, sensed: list[Observation]) -> list[object]:
    ctx = DecisionContext(time=0, self_state=_self_state(position), sensed=sensed, memory=_MAP)
    return [candidate.action for candidate in TreatByBeacon(_CONFIG, memory).propose(ctx)]


def test_a_robot_on_a_pest_treats_it_and_bids_on_beacons_otherwise() -> None:
    memory = _memory_hearing((1.0, (5, 0)))
    pest = InfestationObservation(
        cell=(3, 3), confidence=1.0, observed_at=0, source=1, severity=0.5, pest=None
    )

    assert _bids(memory, (3, 3), [pest]) == [Treat((3, 3))]
    assert _bids(memory, (0, 0), []) == [FollowField(HopField.name)]
    assert _bids(BeaconMemory(), (0, 0), []) == []


def test_a_robot_does_not_bid_on_a_beacon_on_its_own_cell() -> None:
    cleared_pest = _memory_hearing((0.0, (3, 3)))

    assert _bids(cleared_pest, (3, 3), []) == []


def _hop_killer(config: BeaconConfig) -> ComposedAgent:
    spec = BrainSpec(
        behaviors=(TreatByBeacon.name,),
        seeders=(HopField.name,),
        sharing=SharingConfig(facts=KnowledgeSharing.NOTHING),
        beacons=config,
        keeps_map=False,
    )
    return build_agent(spec, 1, _FIELD, Random(0))


def test_a_hop_killer_steps_towards_the_beacon_it_hears() -> None:
    killer = _hop_killer(_CONFIG)
    beacon = Message(
        sender=2,
        sent_at=0,
        kind=MessageKind.BEACON,
        payload=(Beacon(BeaconKind.TREAT, 0.0, (5, 0)),),
    )

    decision = killer.decide(
        Perception(time=0, self_state=_self_state((0, 0)), sensed=[], inbox=[beacon])
    )

    assert decision.action == Move((1, 0))
    assert killer.inspect().current_target == (5, 0)


def test_with_metric_hops_a_small_gamma_follows_the_lowest_beacon() -> None:
    memory = _memory_hearing((1.0, (0, 4)), (2.0, (4, 0)))
    near_literal = BeaconConfig(gamma=0.01, hop_cells=2.0)

    heading = _heading(near_literal, memory, (0, 0))

    assert heading is not None
    assert heading.target == (0, 4)
    assert heading.offset[1] > 0
    assert abs(heading.offset[0]) < 0.1


def test_with_metric_hops_a_robot_on_a_relay_follows_the_relays_downhill_beacons() -> None:
    memory = _memory_hearing((2.0, (0, 0)), (1.5, (3, 0)))

    heading = _heading(BeaconConfig(hop_cells=2.0), memory, (0, 0))

    assert heading is not None
    assert heading.target == (3, 0)


def test_two_robots_on_one_cell_do_not_hold_each_others_value() -> None:
    memory = _memory_hearing((3.0, (0, 0)))

    heading = _heading(BeaconConfig(hop_cells=2.0), memory, (0, 0))
    beacon = treat_beacon(memory, (0, 0), 0, BeaconConfig(hop_cells=2.0))

    assert heading is None
    assert beacon == Beacon(BeaconKind.TREAT, 3.5, (0, 0))


def test_beacons_above_the_cap_are_ignored_and_not_relayed() -> None:
    capped = BeaconConfig(max_value=4.0)

    assert treat_beacon(_memory_hearing((4.5, (1, 0))), (0, 0), 0, capped) is None
    assert treat_beacon(_memory_hearing((3.5, (1, 0))), (0, 0), 0, capped) is None
    assert treat_beacon(_memory_hearing((3.0, (1, 0))), (0, 0), 0, capped) == Beacon(
        BeaconKind.TREAT, 4.0, (0, 0)
    )


def _memory_with_a_killer_east() -> BeaconMemory:
    memory = _memory_hearing((1.0, (0, 4)), (1.0, (4, 0)))
    memory.note_body(3, BodyType.KILLER, 0)
    return memory


def test_a_killer_ahead_counters_the_pull_of_its_level() -> None:
    config = BeaconConfig(killer_repulsion=1.0)

    heading = _heading(config, _memory_with_a_killer_east(), (0, 0))

    assert heading is not None
    assert heading.target == (0, 4)
    assert heading.offset[0] < 0


def test_without_repulsion_a_killers_beacon_pulls_like_any_other() -> None:
    heading = _heading(_CONFIG, _memory_with_a_killer_east(), (0, 0))

    assert heading is not None
    assert heading.offset[0] == heading.offset[1]


def test_a_robot_does_not_bid_when_only_a_killer_is_downhill() -> None:
    memory = _memory_hearing((1.0, (4, 0)))
    memory.note_body(2, BodyType.KILLER, 0)
    ctx = DecisionContext(time=0, self_state=_self_state((0, 0)), sensed=[], memory=_MAP)

    assert TreatByBeacon(BeaconConfig(killer_repulsion=1.0), memory).propose(ctx) == ()
    assert TreatByBeacon(_CONFIG, memory).propose(ctx) != ()


def test_the_cell_ahead_is_the_neighbour_along_the_offset() -> None:
    assert cell_ahead((5, 5), (3.0, 0.2)) == (6, 5)
    assert cell_ahead((5, 5), (2.0, 2.1)) == (6, 6)
    assert cell_ahead((5, 5), (-0.1, -4.0)) == (5, 4)


def test_a_placed_beacon_turns_the_heading_away_from_it() -> None:
    memory = _memory_hearing((1.0, (5, 0)))
    memory.place((1, 1), 0, 1.0, _ESCAPE.escape_decay)

    heading = _heading(_CONFIG, memory, (0, 0))

    assert heading is not None
    assert heading.offset[0] > 0
    assert heading.offset[1] < 0


def _killer_facing_a_wall(killer: ComposedAgent) -> None:
    """One decision of a killer at (0, 0), pulled east into a wall on (1, 0)."""
    wall = TerrainObservation(
        cell=(1, 0), confidence=1.0, observed_at=0, source=1, terrain=Terrain.OBSTACLE
    )
    beacon = Message(
        sender=2,
        sent_at=0,
        kind=MessageKind.BEACON,
        payload=(Beacon(BeaconKind.TREAT, 0.0, (5, 0)),),
    )
    killer.decide(Perception(time=0, self_state=_self_state((0, 0)), sensed=[wall], inbox=[beacon]))


def _beacons_of(killer: ComposedAgent) -> BeaconMemory:
    assert killer.beacon_link is not None
    return killer.beacon_link.memory


def test_a_killer_stuck_against_a_wall_places_an_escape_beacon_on_it() -> None:
    killer = _hop_killer(BeaconConfig(escape=_ESCAPE))

    _killer_facing_a_wall(killer)

    placed = [beacon.cell for beacon in _beacons_of(killer).placed_beacons()]
    assert placed == [(1, 0)]


def test_an_escape_does_not_weaken_a_stronger_beacon_on_its_cell() -> None:
    killer = _hop_killer(BeaconConfig(escape=_ESCAPE))
    _beacons_of(killer).place((1, 0), 0, 4.0, _ESCAPE.escape_decay)

    _killer_facing_a_wall(killer)

    kept = _beacons_of(killer).placed_beacon_at((1, 0))
    assert kept is not None
    assert kept.strength == 4.0


def test_beacon_memory_forgets_stale_beacons_old_sightings_and_faded_pushes() -> None:
    memory = _memory_hearing((1.0, (5, 0)))
    memory.hear(3, 60, [Beacon(BeaconKind.TREAT, 2.0, (0, 5))])
    memory.see((2, 2), 0, infested=True)
    memory.see((4, 4), 60, infested=True)
    memory.place((1, 1), 0, 1.0, decay=10)
    memory.place((2, 1), 0, 1.0, decay=1000)

    memory.forget({BeaconKind.TREAT: 30}, seen_before=30, now=100)

    assert [heard.sender for heard in memory.beacons(BeaconKind.TREAT)] == [3]
    assert memory.latest_sighting(since=0) == (4, 4)
    assert [placed.cell for placed in memory.placed_beacons()] == [(2, 1)]


def test_beacon_memory_keeps_at_most_its_cap_of_each_sort_and_forgets_old_killers() -> None:
    memory = BeaconMemory(cap=2)
    for sender in (1, 2, 3):
        memory.hear(sender, sent_at=sender, beacons=[Beacon(BeaconKind.TREAT, 1.0, (sender, 0))])
        memory.note_body(sender, BodyType.KILLER, at=sender)
        memory.see((sender, 5), at=sender, infested=True)

    heard: list[int] = []
    for beacon in memory.beacons(BeaconKind.TREAT):
        heard.append(beacon.sender)
    assert sorted(heard) == [2, 3]
    assert memory.killers() == frozenset({2, 3})
    assert memory.latest_sighting(since=0) == (3, 5)

    memory.forget({BeaconKind.TREAT: 3}, seen_before=0, now=3)
    assert memory.killers() == frozenset({3})


_DOCKING = BeaconConfig(hop_cells=2.0, max_value=4.0, killer_repulsion=2.0, dock=DockBeacons())
"""TREAT beacons capped at 4 and repelled by killers; DOCK beacons with neither."""


def test_dock_beacons_are_relayed_past_the_treat_cap() -> None:
    memory = BeaconMemory()
    memory.hear(2, 0, [Beacon(BeaconKind.TREAT, 8.0, (4, 0)), Beacon(BeaconKind.DOCK, 8.0, (4, 0))])

    assert treat_beacon(memory, (0, 0), 0, _DOCKING) is None
    relayed = dock_beacon(memory, (0, 0), 0, _DOCKING)
    assert relayed == Beacon(BeaconKind.DOCK, 10.0, (0, 0))


def test_a_killers_dock_beacon_pulls_although_its_treat_beacon_does_not() -> None:
    memory = BeaconMemory()
    memory.hear(2, 0, [Beacon(BeaconKind.TREAT, 0.0, (4, 0)), Beacon(BeaconKind.DOCK, 0.0, (4, 0))])
    memory.note_body(2, BodyType.KILLER, 0)

    to_the_pest = hop_pull(memory, (0, 0), 0, _DOCKING)
    to_the_dock = HopDockField(_DOCKING, memory).guidance_for(_MAP, (0, 0), 0)

    # A killer's TREAT beacon alone never pulls; its DOCK beacon does, like any other.
    assert to_the_pest is None
    assert to_the_dock is not None and to_the_dock.offset[0] > 0


def test_a_dock_beacon_still_pulls_at_full_strength_hours_after_treat_beacons_expire() -> None:
    killer = _hop_killer(_DOCKING)
    beacon = Message(
        sender=2,
        sent_at=0,
        kind=MessageKind.BEACON,
        payload=(Beacon(BeaconKind.TREAT, 0.0, (5, 0)), Beacon(BeaconKind.DOCK, 0.0, (0, 5))),
    )
    later = 5 * TimeUnit.HOUR
    killer.decide(Perception(time=0, self_state=_self_state((0, 0)), sensed=[], inbox=[beacon]))
    killer.decide(Perception(time=later, self_state=_self_state((0, 0)), sensed=[], inbox=[]))

    memory = _beacons_of(killer)
    pull = hop_pull(memory, (0, 0), later, _DOCKING, BeaconKind.DOCK)
    assert list(memory.beacons(BeaconKind.TREAT)) == []
    assert len(list(memory.beacons(BeaconKind.DOCK))) == 1
    assert pull is not None and pull.strength == 1.0


def test_a_hop_killer_low_on_battery_walks_down_the_dock_beacons_and_charges_on_the_dock() -> None:
    spec = BrainSpec(
        behaviors=(TreatByBeacon.name, ReturnToDock.name),
        seeders=(HopField.name, HopDockField.name),
        sharing=SharingConfig(facts=KnowledgeSharing.NOTHING),
        beacons=_DOCKING,
        keeps_map=False,
    )
    killer = build_agent(spec, 1, _FIELD, Random(0), docks=((5, 0),))
    dock = Message(
        sender=-500,
        sent_at=0,
        kind=MessageKind.BEACON,
        payload=(Beacon(BeaconKind.DOCK, 0.0, (5, 0)),),
    )
    low = replace(_self_state((0, 0)), battery=0.05)

    on_the_way = killer.decide(Perception(time=0, self_state=low, sensed=[], inbox=[dock]))
    on_the_dock = killer.decide(
        Perception(
            time=TimeUnit.MINUTE, self_state=replace(low, position=(5, 0)), sensed=[], inbox=[]
        )
    )

    assert on_the_way.action == Move((1, 0))
    assert on_the_dock.action == Charge()


def test_a_hop_killer_low_on_battery_that_hears_no_dock_beacon_heads_for_the_dock() -> None:
    spec = BrainSpec(
        behaviors=(TreatByBeacon.name, ReturnToDock.name),
        seeders=(HopField.name, HopDockField.name),
        sharing=SharingConfig(facts=KnowledgeSharing.NOTHING),
        beacons=_DOCKING,
        keeps_map=False,
    )
    killer = build_agent(spec, 1, _FIELD, Random(0), docks=((5, 0), (9, 9)))
    low = replace(_self_state((0, 0)), battery=0.05)

    decision = killer.decide(Perception(time=0, self_state=low, sensed=[], inbox=[]))

    assert decision.action == Move((1, 0))
