# Swarm Field

A decentralized swarm algorithm and simulation for robots cultivating and monitoring a crop field. Each robot follows simple local rules and shares information peer-to-peer, so that swarm-level behaviour emerges and the robots together form a "collective brain" — a shared body of knowledge about the field that no single robot holds on its own.

See [`docs/system-decomposition.md`](docs/system-decomposition.md) for the module boundaries and interfaces, and [`docs/architecture.md`](docs/architecture.md) for the inter- and intra-module architecture.

## Requirements

- [uv](https://docs.astral.sh/uv/) for environment and dependency management (run `uv --version` in the terminal).
- Python 3.12 (uv will provision it automatically from `.python-version`).

## Setup

Create the virtual environment and install all dependencies (including dev tools):

```bash
uv sync
```

## Development commands

All commands run inside the project environment via `uv run`.

| Task | Command |
|------|---------|
| Run the tests | `uv run pytest` |
| Lint | `uv run ruff check .` |
| Auto-fix lint issues | `uv run ruff check --fix .` |
| Format code | `uv run ruff format .` |
| Check formatting (no changes) | `uv run ruff format --check .` |
| Type-check | `uv run pyrefly check` |

Run all checks the way CI and reviewers will, before committing or pushing:

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyrefly check && uv run pytest
```

## Modules

- `contracts/` — the shared vocabulary (dataclasses, enums, the `SwarmAgent` protocol) that crosses module boundaries. Depends on nothing; every other package depends on it.
- `swarm/` (A) — the decentralized algorithm, the robot "brain": a `ComposedAgent` combines `Behavior`s that propose scored actions.
- `sim/` (B) — the environment simulation: a `SimLoop` runs an ordered list of `SimStage`s over a shared `SimContext`, and owns all ground truth.
- `viz/` (C) — the visualization adapter: `Renderer`s draw read-only `WorldSnapshot`s.

## Project layout

```
src/swarm_field/   # the four packages above
tests/             # pytest suite, mirroring the source
docs/              # design documents and source material
```
