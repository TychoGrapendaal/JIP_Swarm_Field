# AGENTS.md

Shared guide for contributors and AI agents working in this repository.

## The project

A decentralized swarm of robots that monitor and cultivate a crop field. Each robot follows simple local rules and shares information peer-to-peer, so that swarm-level behaviour emerges — a "collective brain" no single robot holds on its own. The deliverable is a simulation used to design and benchmark the swarm algorithm: a research proof of concept for the TU Delft Robotics Institute.

## Where to find things

- [`docs/system-decomposition.md`](docs/system-decomposition.md) — the module boundaries and the interface contracts between them.
- [`docs/architecture.md`](docs/architecture.md) — inter- and intra-module architecture: the composable brain (A) and the discrete-event simulation (B).
- `docs/*.pdf` — the problem statement and literature review (domain background on swarm robotics for crop fields).
- Source lives under `src/swarm_field/<package>/`; tests mirror it under `tests/`.

## Modules

- `contracts/` — shared boundary types (dataclasses, enums, the `SwarmAgent` protocol). Depends on nothing; everything depends on it.
- `swarm/` (A) — the robot brain: `ComposedAgent` combines `Behavior`s that propose scored actions.
- `simulation/` (B) — the environment simulation: `SimulationLoop` processes timestamped events from an `EventQueue`, and owns ground truth.
- `visualization/` (C) — the visualization adapter: `Renderer`s draw read-only `WorldSnapshot`s.

## Architecture rules

- `contracts` is the only cross-component dependency; `swarm`, `simulation` and `visualization` never import each other's internals.
- The simulation owns all ground truth. The algorithm sees only `Perception` (perceived, partial, noisy). Visualization is read-only, via `WorldSnapshot`.
- The `SwarmAgent` protocol is the single boundary between algorithm and simulation.
- Time is discrete-event: an integer `Time` timestamp (no fixed timestep); events are processed in time order.
- The brain and the loop are each assembled from small, single-purpose parts (`Behavior`s / event handlers), built programmatically.

## Development

The project uses [uv](https://docs.astral.sh/uv/) with Ruff (lint + format), Pyrefly (type check) and Pytest. Run `uv sync` once to set up. See [`README.md`](README.md) for the full command list.

- **Run all checks before committing or pushing, and make sure they pass:**

  ```bash
  uv run ruff check . && uv run ruff format --check . && uv run pyrefly check && uv run pytest
  ```

  Fix formatting with `uv run ruff format .` and auto-fixable lint with `uv run ruff check --fix .`.
- **Strict typing is enforced.** Pyrefly runs in strict mode: every parameter, return type and attribute must be annotated (locals are inferred). Fix the annotation rather than silencing it with a type-ignore.

## Documentation conventions

- **Never hard-wrap prose.** In Markdown and LaTeX write one paragraph as one physical line and let the editor soft-wrap. This applies to prose, bullets and list items; line breaks belong only in source code.
- Put dataclass field docs in the class docstring (an `Attributes:` section), not as trailing `#` comments, so they show on hover.
- Prefer Mermaid for diagrams. Avoid reserved keywords as node/participant identifiers in sequence diagrams (`loop`, `alt`, `opt`, `par`, `end`, `note`, `activate`) — they cause parse errors.

## Python house style

Classes for state, functions for pure processing:

- The codebase mixes object-oriented and functional style. Use **classes** for things that store and mutate state; use **plain functions** for pure, stateless processing.
- **Simplicity over complexity.** Don't reach for a pattern you don't need — if something can be a list, make it a list. When you foresee future growth, keep it simple now and refactor later. Only generalise a repeated pattern on its third occurrence.
- **Refactor often.** When a feature works, review it once more for (a) readability for a human developer, (b) architectural soundness, (c) low-level code smells.

Naming and types:

- **Descriptive, purpose-revealing names.** A function's name should say what it does and what it returns — `extract_robot_statistics()`, not `statistics()`. (snake_case, per PEP 8.)
- Use a **`@dataclass` for composite return types.** Anything more composite than a simple coordinate pair should be a named dataclass with named members, not a bare tuple.
- Use **enums** for a small set of named variants.
- Use **`tuple[T, ...]`** for immutable lists.
- **Conversions live on the class.** To turn one dataclass into another, prefer Rust-like `into_*()` / `from_*()` methods on the class over free-standing conversion functions. More generally, a function tightly bound to a dataclass should be a method on it.

Control flow:

- **No nested one-line `if … else` (ternaries) or multi-clause comprehensions** (`[x for a in b for c in d]`). Write explicit loops and blocks — flatten with a loop and `.extend(...)` where needed.

Docs, comments and logging:

- Write docstrings for public functions and classes, except trivial one/two-liners, getters/setters and other obvious cases. Always document complex or flow-central functions.
- Keep comments and docstrings relevant to the code and its caller. Don't put `.md`-level documentation in source. To refer to other code, name the class or file instead of describing it, so the note doesn't rot when that code changes.
- **Logging** goes through the single, globally configured logger — don't scatter `print` statements for diagnostics.

## Tests

- This is a research project: there is no 100% coverage goal. Prefer informative tests that don't clutter the suite or make refactoring painful.
- Test the happy path. Skip most negative branches, except failures that are expected to happen often.
- Assert the **error type**, never the literal error message.
