"""Tests for body capability enforcement and body-specific agent assembly."""

from swarm_field.contracts import (
    BodyType,
    Charge,
    FieldGeometry,
    Idle,
    Move,
    PeerObservation,
    Perception,
    ReplaceBattery,
    RobotSelfState,
    ScenarioConfig,
    Terrain,
    TerrainObservation,
    Treat,
)
from swarm_field.simulation import RobotState, WorldState
from swarm_field.simulation.scenario import build_world
from swarm_field.swarm import (
    ActionCandidate,
    AgentMemory,
    CapabilitySet,
    ComposedAgent,
    FieldProvider,
    Navigator,
    build_agent,
    capabilities_for,
)
from swarm_field.swarm.behavior import DecisionContext, Intent

FIELD = FieldGeometry(
    width=8,
    height=8,
    cell_size=10.0,
    row_sp=2.0,
    in_row_sp=1.0,
    headland=0.0,
    connectivity=8,
)


class _CandidateBehavior:
    name = "test-candidates"

    def __init__(self, *actions: tuple[Intent, float]) -> None:
        self._actions = actions

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        return tuple(
            ActionCandidate(action=action, score=score, source=self.name, target=None)
            for action, score in self._actions
        )


def _perception(body: BodyType, *, battery: float = 1.0, spare_batteries: int = 0) -> Perception:
    return Perception(
        time=0,
        self_state=RobotSelfState(
            robot_id=1,
            position=(2, 2),
            battery=battery,
            body=body,
            speed_mps=capabilities_for(body).speed_mps,
            consumable=1.0,
            health=1.0,
            spare_batteries=spare_batteries,
        ),
        sensed=[],
        inbox=[],
    )


def _manual_agent(body: BodyType, behavior: _CandidateBehavior) -> ComposedAgent:
    memory = AgentMemory(1, FIELD)
    capabilities = capabilities_for(body)
    fields = FieldProvider(memory, flies=capabilities.can_fly, seeders=[])
    return ComposedAgent(
        robot_id=1,
        behaviors=[behavior],
        memory=memory,
        fields=fields,
        navigator=Navigator(fields.quadtree),
        capabilities=capabilities,
    )


def test_capability_profiles_match_the_three_body_roles() -> None:
    scout = capabilities_for(BodyType.SCOUT)
    killer = capabilities_for(BodyType.KILLER)
    battery = capabilities_for(BodyType.BATTERY)

    assert scout.can_fly and scout.can_detect_pests
    assert scout.pest_sensing_radius == 1
    assert scout.speed_mps == 8.0
    assert scout.endurance_seconds == 30 * 60
    assert scout.dock_recharge_seconds == 45 * 60
    assert not scout.can(Treat) and not scout.can(ReplaceBattery)

    assert not killer.can_fly and killer.can_detect_pests and killer.can_treat
    assert killer.pest_sensing_radius == 0
    assert killer.speed_mps == 1.0
    assert killer.endurance_seconds == 8 * 60 * 60
    assert killer.can(Treat) and not killer.can(ReplaceBattery)

    assert not battery.can_fly and battery.can_supply_batteries
    assert battery.spare_battery_capacity == 3
    assert battery.can(ReplaceBattery) and not battery.can(Treat)


def test_scout_capability_gate_rejects_treat_even_if_it_scores_highest() -> None:
    agent = _manual_agent(
        BodyType.SCOUT,
        _CandidateBehavior((Idle(), 1.0), (Treat((2, 2)), 100.0)),
    )

    decision = agent.decide(_perception(BodyType.SCOUT))

    assert decision.action == Idle()


def test_killer_capability_gate_allows_treatment_but_not_battery_supply() -> None:
    agent = _manual_agent(
        BodyType.KILLER,
        _CandidateBehavior(
            (Idle(), 0.1),
            (Treat((2, 2)), 0.8),
            (ReplaceBattery(robot_id=2), 1.0),
        ),
    )

    decision = agent.decide(_perception(BodyType.KILLER))

    assert decision.action == Treat((2, 2))


def test_scout_capability_gate_rejects_battery_replacement() -> None:
    agent = _manual_agent(
        BodyType.SCOUT,
        _CandidateBehavior((Idle(), 0.1), (ReplaceBattery(robot_id=2), 100.0)),
    )

    decision = agent.decide(_perception(BodyType.SCOUT))

    assert decision.action == Idle()


def test_battery_capability_gate_rejects_pest_treatment() -> None:
    agent = _manual_agent(
        BodyType.BATTERY,
        _CandidateBehavior((Idle(), 0.1), (Treat((2, 2)), 100.0)),
    )

    decision = agent.decide(_perception(BodyType.BATTERY))

    assert decision.action == Idle()


def test_body_mismatch_cannot_expand_capabilities() -> None:
    # Even a misconfigured profile that grants Treat cannot expand what SCOUT may emit.
    scout_profile = capabilities_for(BodyType.SCOUT)
    expanded = CapabilitySet(
        body=BodyType.SCOUT,
        action_types=scout_profile.action_types | {Treat},
        can_fly=scout_profile.can_fly,
        speed_mps=scout_profile.speed_mps,
        pest_sensing_radius=scout_profile.pest_sensing_radius,
        terrain_sensing_radius=scout_profile.terrain_sensing_radius,
        peer_sensing_radius=scout_profile.peer_sensing_radius,
        endurance_seconds=scout_profile.endurance_seconds,
        dock_recharge_seconds=scout_profile.dock_recharge_seconds,
        spare_battery_capacity=scout_profile.spare_battery_capacity,
        can_detect_pests=scout_profile.can_detect_pests,
    )
    memory = AgentMemory(1, FIELD)
    fields = FieldProvider(memory, flies=True, seeders=[])
    agent = ComposedAgent(
        1,
        [_CandidateBehavior((Treat((2, 2)), 10.0))],
        memory,
        fields,
        Navigator(fields.quadtree),
        capabilities=expanded,
    )

    assert agent.decide(_perception(BodyType.SCOUT)).action == Idle()


def test_every_body_registers_all_seeders() -> None:
    expected_seeders = {"explore", "treat", "battery-support", "dock"}

    for body in BodyType:
        agent = build_agent(1, body, FIELD, docks=((1, 1),))

        assert agent.fields.names() == expected_seeders


def test_build_agent_uses_a_supplied_behavior_list() -> None:
    custom_behavior = _CandidateBehavior((Idle(), 0.4))

    agent = build_agent(1, BodyType.SCOUT, FIELD, behaviors=[custom_behavior])
    no_behavior_agent = build_agent(2, BodyType.SCOUT, FIELD, behaviors=[])

    assert agent.behaviors == [custom_behavior]
    assert no_behavior_agent.behaviors == []


def test_return_to_dock_charges_when_at_a_configured_dock() -> None:
    agent = build_agent(1, BodyType.SCOUT, FIELD, docks=((2, 2),))
    state = RobotSelfState(
        robot_id=1,
        position=(2, 2),
        battery=0.8,
        body=BodyType.SCOUT,
        speed_mps=8.0,
        consumable=1.0,
        health=1.0,
    )

    decision = agent.decide(Perception(0, state, [], []))

    assert decision.action == Charge()


def test_body_assembly_configures_flying_traversability_and_roles() -> None:
    scout = build_agent(1, BodyType.SCOUT, FIELD)
    killer = build_agent(2, BodyType.KILLER, FIELD)
    battery = build_agent(3, BodyType.BATTERY, FIELD)

    assert scout.fields._flies is True
    assert [behavior.name for behavior in scout.behaviors] == ["explore-field", "return-to-dock"]
    assert killer.fields._flies is False
    assert [behavior.name for behavior in killer.behaviors] == [
        "treat-infestation",
        "return-to-dock",
    ]
    assert battery.fields._flies is False
    assert [behavior.name for behavior in battery.behaviors] == ["supply-battery", "return-to-dock"]


def test_scout_can_move_across_obstacles_but_killer_cannot() -> None:
    obstacle = TerrainObservation(
        cell=(3, 2), confidence=1.0, observed_at=0, source=1, terrain=Terrain.OBSTACLE
    )
    scout = build_agent(1, BodyType.SCOUT, FIELD)
    killer = build_agent(2, BodyType.KILLER, FIELD)

    scout.decide(Perception(0, _perception(BodyType.SCOUT).self_state, [obstacle], []))
    killer_state = RobotSelfState(
        robot_id=2,
        position=(2, 2),
        battery=1.0,
        body=BodyType.KILLER,
        speed_mps=1.0,
        consumable=1.0,
        health=1.0,
    )
    killer.decide(Perception(0, killer_state, [obstacle], []))

    assert scout.fields.quadtree is not None
    assert killer.fields.quadtree is not None
    assert scout.fields.quadtree.is_traversable((3, 2))
    assert not killer.fields.quadtree.is_traversable((3, 2))


def test_battery_assembly_replaces_a_low_battery_peer_in_same_cell() -> None:
    battery_agent = build_agent(1, BodyType.BATTERY, FIELD)
    peer = PeerObservation(
        cell=(2, 2),
        confidence=1.0,
        observed_at=0,
        source=1,
        peer=2,
        body=BodyType.SCOUT,
        battery=0.1,
    )
    state = RobotSelfState(
        robot_id=1,
        position=(2, 2),
        battery=1.0,
        body=BodyType.BATTERY,
        speed_mps=1.0,
        consumable=1.0,
        health=1.0,
        spare_batteries=3,
    )

    decision = battery_agent.decide(Perception(0, state, [peer], []))

    assert decision.action == ReplaceBattery(robot_id=2)


def test_battery_assembly_follows_field_to_a_remote_low_battery_peer() -> None:
    battery_agent = build_agent(1, BodyType.BATTERY, FIELD)
    peer = PeerObservation(
        cell=(5, 2),
        confidence=1.0,
        observed_at=0,
        source=2,
        peer=2,
        body=BodyType.SCOUT,
        battery=0.1,
    )
    state = RobotSelfState(
        robot_id=1,
        position=(2, 2),
        battery=1.0,
        body=BodyType.BATTERY,
        speed_mps=1.0,
        consumable=1.0,
        health=1.0,
        spare_batteries=3,
    )

    decision = battery_agent.decide(Perception(0, state, [peer], []))

    assert isinstance(decision.action, Move)


def test_sensing_footprints_are_body_specific() -> None:
    from swarm_field.contracts import InfestationObservation

    robots = [
        RobotState(1, (2, 2), BodyType.SCOUT, speed_mps=8.0),
        RobotState(2, (2, 2), BodyType.KILLER, speed_mps=1.0),
        RobotState(3, (2, 2), BodyType.BATTERY, speed_mps=1.0, spare_batteries=3),
    ]
    world = WorldState(FIELD, robots=robots)
    world.cell((3, 3)).infestation = 0.7

    scout_obs = world.observe_for(1, 0)
    killer_obs = world.observe_for(2, 0)
    battery_obs = world.observe_for(3, 0)

    assert any(isinstance(obs, InfestationObservation) and obs.cell == (3, 3) for obs in scout_obs)
    assert not any(
        isinstance(obs, InfestationObservation) and obs.cell == (3, 3) for obs in killer_obs
    )
    assert not any(isinstance(obs, InfestationObservation) for obs in battery_obs)
    scout_terrain = [obs for obs in scout_obs if isinstance(obs, TerrainObservation)]
    assert len(scout_terrain) == 9


def test_scenario_world_preserves_configured_charging_docks() -> None:
    scenario = ScenarioConfig(
        field=FIELD,
        crops={},
        robots=[],
        seed=0,
        docks=((1, 1),),
    )

    world = build_world(scenario)

    assert world.docks == frozenset({(1, 1)})
