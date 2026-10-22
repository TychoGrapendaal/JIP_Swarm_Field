"""Tests for knowledge sharing: exact snapshot merges, and what each strategy broadcasts."""

import numpy as np
import pytest

from swarm_field.contracts import (
    BodyType,
    CellId,
    FieldGeometry,
    InfestationObservation,
    Observation,
    PeerObservation,
    RobotId,
    Terrain,
    TerrainObservation,
    Time,
    TimeUnit,
)
from swarm_field.experiments import PestScenario
from swarm_field.runner import run_experiment
from swarm_field.swarm import (
    AgentMemory,
    KnowledgeExchange,
    KnowledgeSnapshot,
    SharingConfig,
    SharingStrategy,
)

_FIELD = FieldGeometry(width=10, height=10, cell_size=10.0, row_sp=0.0, in_row_sp=0.0, headland=0.0)


def _pest(cell: CellId, severity: float, at: Time, source: RobotId) -> InfestationObservation:
    return InfestationObservation(
        cell=cell, confidence=1.0, observed_at=at, source=source, severity=severity, pest=None
    )


def _terrain(cell: CellId, terrain: Terrain, at: Time) -> TerrainObservation:
    return TerrainObservation(cell=cell, confidence=1.0, observed_at=at, source=2, terrain=terrain)


def _peer(peer: RobotId, at: Time) -> PeerObservation:
    return PeerObservation(
        cell=(0, 0), confidence=1.0, observed_at=at, source=peer, peer=peer, body=BodyType.SCOUT
    )


def _memory(robot: RobotId, facts: list[Observation]) -> AgentMemory:
    memory = AgentMemory(robot, _FIELD)
    memory.merge(facts)
    return memory


def _receiver() -> AgentMemory:
    """A robot that knows some cells older, newer or as new as the sender, and some peers."""
    return _memory(
        1,
        [
            _pest((1, 1), 0.2, at=10, source=1),
            _pest((2, 2), 0.9, at=50, source=1),
            _pest((3, 3), 0.4, at=20, source=1),
            _terrain((4, 4), Terrain.FREE, at=0),
            _peer(3, at=30),
        ],
    )


def _sender() -> AgentMemory:
    return _memory(
        2,
        [
            _pest((5, 5), 0.7, at=40, source=2),
            _pest((1, 1), 0.5, at=30, source=2),
            _pest((2, 2), 0.1, at=40, source=2),
            _pest((3, 3), 0.6, at=20, source=2),
            _terrain((4, 4), Terrain.OBSTACLE, at=5),
            _terrain((6, 6), Terrain.OBSTACLE, at=5),
            _peer(3, at=60),
            _peer(4, at=60),
            _peer(1, at=60),
        ],
    )


def _assert_same_memory(merged: AgentMemory, reference: AgentMemory) -> None:
    assert merged.snapshot() == reference.snapshot()
    assert np.array_equal(merged.last_observed_times(), reference.last_observed_times())
    assert np.array_equal(merged.monitorable_cells(), reference.monitorable_cells())
    assert merged.version == reference.version


def test_merging_a_snapshot_equals_merging_its_facts_one_by_one() -> None:
    sender = _sender()
    sender.merge([_pest((7, 7), 0.3, at=70, source=2)])
    snapshot = sender.knowledge()
    merged = _receiver()
    reference = _receiver()

    merged.merge_snapshot(snapshot)
    reference.merge(snapshot.facts())

    _assert_same_memory(merged, reference)


def test_an_unchanged_part_is_shared_again_and_skipped_by_a_peer_that_merged_it() -> None:
    sender = _sender()
    first = sender.knowledge()
    sender.merge([_pest((7, 7), 0.3, at=70, source=2)])
    second = sender.knowledge()
    merged = _receiver()
    reference = _receiver()

    merged.merge_snapshot(first)
    merged.merge_snapshot(second, merged_before=first)
    reference.merge(first.facts())
    reference.merge(second.facts())

    assert second.terrain is first.terrain
    assert second.infestation is not first.infestation
    _assert_same_memory(merged, reference)


_knowledge = AgentMemory.knowledge


def _knowledge_built_anew(memory: AgentMemory) -> KnowledgeSnapshot:
    """``AgentMemory.knowledge`` without reusing the parts of the previous snapshot."""
    memory._shared = None
    return _knowledge(memory)


def _merge_fact_by_fact(
    memory: AgentMemory, snapshot: KnowledgeSnapshot, merged_before: KnowledgeSnapshot | None = None
) -> None:
    memory.merge(snapshot.facts())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_pest_run_is_the_same_as_with_every_snapshot_merged_fact_by_fact(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A small field, so that the robots' paths cross and what they share matters early on.
    scenario = PestScenario(size=16, seed=seed, until=5 * TimeUnit.MINUTE)
    config = scenario.into_experiment().with_sharing(SharingConfig(SharingStrategy.FULL))

    fast = run_experiment(config)
    monkeypatch.setattr(AgentMemory, "knowledge", _knowledge_built_anew)
    monkeypatch.setattr(AgentMemory, "merge_snapshot", _merge_fact_by_fact)
    reference = run_experiment(config)

    assert fast.series == reference.series


def test_a_periodic_robot_shares_nothing_between_whole_memory_broadcasts() -> None:
    exchange = KnowledgeExchange(_sender(), SharingConfig(SharingStrategy.PERIODIC, full_every=3))

    sent = [exchange.knowledge_to_send() for _ in range(3)]

    assert sent[:2] == [None, None]
    assert sent[2] is not None
    assert exchange.facts_sent == len(sent[2])
