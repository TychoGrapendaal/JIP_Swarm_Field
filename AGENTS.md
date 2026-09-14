# AGENTS.md

Shared guide for contributors and AI agents working in this repository. This is a **work in progress** — it grows as the project is scaffolded (tooling, layout and commands land in the repository‑initialization phase).

## Project

A decentralized swarm algorithm plus a simulation for robots cultivating and monitoring a crop field. Three top‑level components — the swarm algorithm (A), the environment simulation (B) and the visualization (C) — plus a shared `contracts` foundation. See [`docs/system-decomposition.md`](docs/system-decomposition.md) for the module boundaries and interfaces, and [`docs/domain-mapping.md`](docs/domain-mapping.md) for the original mind map.

## Documentation conventions

- **Never hard‑wrap prose.** In Markdown and LaTeX, write **one paragraph as one physical line** and let the editor soft‑wrap it. Do not insert manual line breaks inside a paragraph — they make the text painful to edit. This applies to prose, bullet points and list items.
- **Line breaks belong only in source code** (and in code blocks / docstrings inside documentation), where they are structural.
- Prefer Mermaid for diagrams so they render on GitHub. Avoid reserved keywords as participant/node identifiers in Mermaid sequence diagrams (e.g. `loop`, `alt`, `opt`, `par`, `end`, `note`, `activate`) — they cause parse errors.

## Design conventions

- **`contracts` is the only cross‑component dependency.** A, B and C depend on `contracts` and never on each other's internals.
- Shared boundary types are plain data: `@dataclass` (frozen for value types), `Enum`, or simple type aliases. No behaviour in `contracts`.
- Simulation time is discrete: timestamps are **tick counts (`int`)**, never floating‑point seconds. Seconds derive as `tick * dt`.
- The simulation owns all ground truth; the algorithm sees only `Perception`. Visualization is read‑only except through enumerated `Command`s.
