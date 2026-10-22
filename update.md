# Body & Capability Model — Update

## Scope

This update implements the requirements shown in the uploaded Body & Capability Model issue description: capabilities are defined per `BodyType`, behaviours are assembled by body type, and `ComposedAgent` prevents a robot from emitting unsupported actions.

## Changes

- **Per-body capability profiles:** `CapabilitySet` defines the supported action classes and the current high-level sensing, movement, flight, endurance, recharge, and battery-supply assumptions for `SCOUT`, `KILLER`, and `BATTERY`.
- **Capability gate:** `ComposedAgent` uses the body type reported by the current perception as authoritative. It filters unsupported candidates before arbitration and validates the concrete action after resolving a `FollowField` intent. A disallowed action is replaced with `Idle`.
- **Body-specific assembly:** `build_agent` configures SCOUT with exploration, KILLER with infestation treatment, and BATTERY with battery support, while wiring each body's navigation and field seeders.
- **Original treatment logic preserved:** `TreatInfestation` keeps its existing in-cell treatment, infestation counting, KILLER peer counting, urgency score, and `FollowField` proposal. Target selection and treatment logic were not redesigned for this issue.
- **Battery status sharing retained:** every `STATUS` message includes the sending robot's current battery fraction, including BATTERY robots. The value is represented in the peer observation so other agents can identify low-battery peers.
- **Crop-observation memory relay retained:** `AgentMemory` stores the newest crop observation per cell, includes it in memory snapshots and `KNOWLEDGE` messages, and merges received crop facts so they can be relayed by other agents.
- **Typing fixes:** behavior assembly explicitly types its list against the common `Behavior` protocol; status/knowledge tests narrow the opaque `Message.payload` to a tuple before iterating; and world-state reset assignments explicitly annotate nullable fields for Pyrefly.
- **Regression coverage:** tests now verify SCOUT cannot treat pests or replace batteries, KILLER can treat but cannot replace batteries, and BATTERY cannot treat pests. Existing tests cover body assembly, battery status, crop relay, memory merge, and world-state behavior.

## Current information shared between robots

Each agent broadcasts two messages per decision:

1. **`STATUS`:** the robot's current cell, body type, battery fraction, and current target/objective when available.
2. **`KNOWLEDGE`:** the agent's stored observations, which may include crop type and growth stage, terrain, infestation severity and pest type, and known peer state including battery fraction and reported target/objective.

Memory retains the newest observation per cell for crop, terrain, and infestation, and the newest report for each named peer. Crop observations are relayed as knowledge; this change does not add crop-based decision logic.

## Verification

- `python -m compileall -q src tests`: passed.
- Focused capability, memory, swarm-message, and world tests: **54 passed**.
- `PYTHONPATH=src python -m pytest -q --ignore=tests/test_explore_swarm.py`: **236 passed**.
- `tests/test_explore_swarm.py`: did not finish within a 120-second execution limit in this environment, so it remains unverified.
- Pyrefly and Ruff could not be run here because the project tools/dependencies could not be downloaded in this environment. The type-related fixes above address the reported diagnostics, but the user's local checks should confirm them.

Recommended local verification from the repository root:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
```

Do not mark checks green until the exploration-swarm test and local static checks complete successfully.
