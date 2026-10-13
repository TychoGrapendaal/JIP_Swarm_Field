# Repository fix notes

This package contains the combined peer-target-sharing and typed `AgentObjective` enum changes.

- Target announcements are shared and ingested into memory.
- Peer-target discounting is intentionally not performed in `ComposedAgent`; objective-specific scoring belongs in the relevant field seeder.
- `PeerTargetObservation.objective` and related APIs use `AgentObjective | None` rather than an arbitrary string.
- `src/swarm_field/swarm/brain.py` is syntactically valid in this snapshot.

Verification performed against this snapshot:

- `PYTHONPATH=src python -m pytest -q`: 95 passed.
- `python -m compileall -q src tests`: passed.

Note: a separate error log mentioning `self.arbiter.choose(...)` and `tests/test_arbiter.py` refers to code that is not present in this snapshot. This package therefore does not claim to repair that separate arbiter implementation.
