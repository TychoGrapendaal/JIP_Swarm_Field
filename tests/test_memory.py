"""Tests for the robot's local memory: storing, merging and decaying facts."""

from math import exp, isclose

from swarm_field.contracts import (
    BodyType,
    CellId,
    CropObservation,
    FieldGeometry,
    InfestationObservation,
    Observation,
    PeerObservation,
    Perception,
    RobotId,
    RobotSelfState,
    Terrain,
    TerrainObservation,
    Time,
)
from swarm_field.swarm import AgentMemory, DecayConfig

_SELF: RobotId = 1


def _memory() -> AgentMemory:
    field = FieldGeometry(
        width=10, height=10, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0
    )
    return AgentMemory(_SELF, field)


def _pest(cell: CellId, severity: float, at: Time, source: RobotId = 2) -> InfestationObservation:
    return InfestationObservation(
        cell=cell, confidence=1.0, observed_at=at, source=source, severity=severity, pest=None
    )


def _terrain(cell: CellId, terrain: Terrain, at: Time) -> TerrainObservation:
    return TerrainObservation(cell=cell, confidence=1.0, observed_at=at, source=2, terrain=terrain)


def _peer(peer: RobotId | None, cell: CellId, at: Time) -> PeerObservation:
    return PeerObservation(
        cell=cell, confidence=1.0, observed_at=at, source=2, peer=peer, body=BodyType.KILLER
    )


def test_newer_fact_replaces_older_and_older_is_ignored() -> None:
    memory = _memory()

    memory.merge([_pest((3, 3), 0.5, at=10), _pest((3, 3), 0.9, at=20), _pest((3, 3), 0.1, at=5)])

    fact = memory.infestation_at((3, 3))
    assert fact is not None
    assert fact.severity == 0.9


def test_tie_keeps_the_stored_fact() -> None:
    memory = _memory()
    own = _pest((3, 3), 0.5, at=10, source=_SELF)

    memory.merge([own, _pest((3, 3), 0.7, at=10, source=2)])

    assert memory.infestation_at((3, 3)) == own


def test_clean_observation_replaces_a_pest() -> None:
    memory = _memory()

    memory.merge([_pest((3, 3), 0.8, at=10), _pest((3, 3), 0.0, at=20)])

    assert [fact.severity for fact in memory.infestations()] == [0.0]


def test_merge_order_does_not_matter() -> None:
    first: list[Observation] = [_pest((1, 1), 0.3, at=10), _terrain((1, 1), Terrain.FREE, at=10)]
    second: list[Observation] = [_pest((1, 1), 0.6, at=30), _peer(5, (2, 2), at=15)]
    forward = _memory()
    backward = _memory()

    forward.merge(first)
    forward.merge(second)
    backward.merge(second)
    backward.merge(first)

    assert set(forward.records()) == set(backward.records())


def test_ingest_stores_crop_terrain_infestation_and_named_peers_only() -> None:
    memory = _memory()
    crop = CropObservation(
        cell=(2, 2), confidence=1.0, observed_at=0, source=_SELF, crop=None, growth_stage=0.5
    )
    perception = Perception(
        time=0,
        self_state=RobotSelfState(
            robot_id=_SELF,
            position=(2, 2),
            battery=1.0,
            body=BodyType.SCOUT,
            speed_mps=8.0,
            consumable=1.0,
            health=1.0,
        ),
        sensed=[
            crop,
            _terrain((2, 3), Terrain.OBSTACLE, at=0),
            _pest((2, 2), 0.4, at=0),
            _peer(5, (2, 2), at=0),
            _peer(None, (2, 2), at=0),
            _peer(_SELF, (2, 2), at=0),
        ],
        inbox=[],
    )

    memory.ingest(perception)

    assert memory.terrain_at((2, 3)) is Terrain.OBSTACLE
    assert memory.infestation_at((2, 2)) is not None
    assert [peer.peer for peer in memory.peers()] == [5]
    assert crop in set(memory.records())


def test_newer_crop_observation_replaces_older_and_is_in_memory_snapshot() -> None:
    memory = _memory()
    older = CropObservation(
        cell=(3, 3), confidence=0.7, observed_at=10, source=2, crop=None, growth_stage=0.2
    )
    newer = CropObservation(
        cell=(3, 3), confidence=1.0, observed_at=20, source=3, crop=None, growth_stage=0.8
    )

    memory.merge([newer, older])

    assert older not in memory.snapshot()
    assert newer in memory.snapshot()


def test_terrain_is_unknown_until_observed_and_outside_beyond_the_field() -> None:
    memory = _memory()

    assert memory.terrain_at((4, 4)) is None
    assert memory.terrain_at((-1, 4)) is Terrain.OUTSIDE
    assert memory.terrain_at((4, 10)) is Terrain.OUTSIDE


def test_confidence_decays_per_kind() -> None:
    memory = _memory()
    decay = DecayConfig()
    pest = _pest((3, 3), 0.5, at=0)
    terrain = _terrain((3, 3), Terrain.FREE, at=0)

    assert isclose(memory.confidence(pest, now=decay.infestation), exp(-1))
    assert memory.confidence(terrain, now=10 * decay.infestation) == 1.0


def test_last_observed_times_keep_the_newest_observation_per_cell() -> None:
    memory = _memory()

    memory.merge([_pest((3, 3), 0.0, at=20), _pest((3, 3), 0.4, at=10), _pest((5, 1), 0.0, at=7)])

    times = memory.last_observed_times()
    assert times[3, 3] == 20
    assert times[5, 1] == 7
    assert times[0, 0] == float("-inf")


def test_known_obstacles_are_not_monitorable() -> None:
    memory = _memory()

    memory.merge([_terrain((2, 2), Terrain.OBSTACLE, at=1), _terrain((4, 4), Terrain.FREE, at=1)])

    monitorable = memory.monitorable_cells()
    assert not monitorable[2, 2]
    assert monitorable[4, 4]
    assert monitorable[0, 0]
