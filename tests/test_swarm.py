"""Tests for the composed brain: candidate gathering and selection."""

from random import Random

from swarm_field.contracts import (
    Action,
    BodyType,
    Direction,
    Idle,
    InfestationObservation,
    Move,
    ObservationBase,
    ObstacleObservation,
    PeerObservation,
    Perception,
    RobotSelfState,
    Turn,
)
from swarm_field.swarm import (
    ActionCandidate,
    ComposedAgent,
    DecisionContext,
    Explore,
    PotentialField,
)


def _perception() -> Perception:
    return Perception(
        time=0,
        self_state=RobotSelfState(
            robot_id=1,
            position=(0, 0),
            facing=Direction.N,
            battery=1.0,
            body=BodyType.SCOUT,
            consumable=1.0,
            health=1.0,
        ),
        sensed=[],
        inbox=[],
    )


class _FixedBehavior:
    """Returns a single candidate."""

    def __init__(self, name: str, action: Action, score: float) -> None:
        self.name: str = name
        self._action: Action = action
        self._score: float = score

    def propose(self, ctx: DecisionContext) -> ActionCandidate:
        return ActionCandidate(action=self._action, score=self._score, source=self.name)


class _MultiBehavior:
    """Returns a tuple of candidates."""

    def __init__(self, name: str, candidates: tuple[ActionCandidate, ...]) -> None:
        self.name: str = name
        self._candidates: tuple[ActionCandidate, ...] = candidates

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        return self._candidates


def test_agent_picks_highest_scoring_candidate() -> None:
    agent = ComposedAgent(
        robot_id=1,
        behaviors=[
            _FixedBehavior("low", Idle(), 0.1),
            _MultiBehavior(
                "multi",
                (
                    ActionCandidate(action=Turn(Direction.N), score=0.9, source="multi"),
                    ActionCandidate(action=Turn(Direction.S), score=0.5, source="multi"),
                ),
            ),
        ],
    )

    decision = agent.decide(_perception())

    assert decision.action == Turn(Direction.N)
    assert agent.inspect().current_goal == "multi"


def test_explore_mock_goes_straight_or_turns_left() -> None:
    explore = Explore(Random(0))
    ctx = DecisionContext(
        time=0,
        self_state=RobotSelfState(
            robot_id=1,
            position=(0, 0),
            facing=Direction.N,
            battery=1.0,
            body=BodyType.SCOUT,
            consumable=1.0,
            health=1.0,
        ),
        sensed=[],
    )

    actions: set[Action] = set()
    for _ in range(20):
        candidates = explore.propose(ctx)
        assert len(candidates) == 1
        actions.add(candidates[0].action)

    # Straight ahead is Move(); 45 degrees left of North is North-West.
    assert actions <= {Move(), Turn(Direction.NW)}
    assert Move() in actions
    assert Turn(Direction.NW) in actions


def test_agent_idles_without_candidates() -> None:
    agent = ComposedAgent(robot_id=1, behaviors=[])

    decision = agent.decide(_perception())

    assert decision.action == Idle()
    assert agent.inspect().current_goal is None


def _context(
    time: int,
    position: tuple[int, int],
    sensed: list[ObservationBase | InfestationObservation | ObstacleObservation],
) -> DecisionContext:
    return DecisionContext(
        time=time,
        self_state=RobotSelfState(
            robot_id=1,
            position=position,
            facing=Direction.N,
            battery=1.0,
            body=BodyType.SCOUT,
            consumable=1.0,
            health=1.0,
        ),
        sensed=sensed,
    )


def test_potential_field_shifts_persistent_exploration() -> None:
    field = PotentialField(radius=2)
    field.update(
        _context(
            time=0,
            position=(0, 0),
            sensed=[ObservationBase(cell=(1, 0), confidence=1.0, observed_at=0, source=2)],
        )
    )

    assert field.exploration == {(1, 0): 1.0}

    field.update(_context(time=1, position=(1, 0), sensed=[]))

    assert field.exploration == {(0, 0): 1.0}
    assert field.potential_map[(0, 0)] == 1.0


def test_potential_field_rebuilds_transient_maps_from_observations() -> None:
    field = PotentialField(radius=2)
    infestation = InfestationObservation(
        cell=(1, 0),
        confidence=1.0,
        observed_at=0,
        source=2,
        severity=0.5,
        pest=None,
    )
    field.update(_context(time=0, position=(0, 0), sensed=[infestation]))

    assert field.infestation_observations == [infestation]
    assert field.potential_map[(1, 0)] < 0.0

    field.update(_context(time=1, position=(0, 0), sensed=[]))

    assert field.infestation_observations == [infestation]
    assert field.potential_map[(1, 0)] < 0.0

    obstacle = ObstacleObservation(
        cell=(0, 1),
        confidence=1.0,
        observed_at=2,
        source=2,
        is_obstacle=True,
    )
    field.update(_context(time=2, position=(0, 0), sensed=[obstacle]))
    assert field.potential_map[(1, 0)] == field.potential_map[(-1, 0)]
    assert field.potential_map[(1, 0)] > 0.0
    assert field.potential_map[(0, 1)] == field.potential_map[(0, -1)]

    clear_obstacle = ObstacleObservation(
        cell=(0, 1),
        confidence=1.0,
        observed_at=3,
        source=2,
        is_obstacle=False,
    )
    field.update(_context(time=3, position=(0, 0), sensed=[clear_obstacle]))
    assert field.obstacle_observations == []
    assert (0, 1) not in field.potential_map


def test_potential_field_updates_infestation_and_peer_observations() -> None:
    field = PotentialField(radius=2)
    infestation = InfestationObservation(
        cell=(1, 0),
        confidence=1.0,
        observed_at=0,
        source=2,
        severity=0.5,
        pest=None,
    )
    peer = PeerObservation(
        cell=(0, 1),
        confidence=1.0,
        observed_at=0,
        source=2,
        peer=3,
    )
    field.update(_context(time=0, position=(0, 0), sensed=[infestation, peer]))

    assert field.infestation_observations == [infestation]
    assert field.peer_observations == [peer]
    assert field.potential_map[(1, 0)] < 0.0
    assert field.potential_map[(0, 1)] > 0.0

    updated_infestation = InfestationObservation(
        cell=(1, 0),
        confidence=1.0,
        observed_at=1,
        source=2,
        severity=1.0,
        pest=None,
    )
    field.update(_context(time=1, position=(0, 0), sensed=[updated_infestation]))

    assert field.infestation_observations == [updated_infestation]
    assert field.potential_map[(1, 0)] < 0.0
    assert field.peer_observations == [peer]

    absent_infestation = InfestationObservation(
        cell=(1, 0),
        confidence=1.0,
        observed_at=2,
        source=2,
        severity=0.0,
        pest=None,
    )
    absent_peer = PeerObservation(
        cell=(0, 1),
        confidence=0.0,
        observed_at=2,
        source=2,
        peer=None,
    )
    field.update(_context(time=2, position=(0, 0), sensed=[absent_infestation, absent_peer]))

    assert field.infestation_observations == []
    assert field.peer_observations == []
    assert field.potential_map == {}
