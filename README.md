# Swarm Field

A decentralized swarm algorithm and simulation for robots cultivating and monitoring a crop field. The simulation is deliberately coarse-grained: one grid cell represents a field patch rather than one plant, and the model is intended to scale to hundreds of robots.

See [`docs/high-level-world-model.md`](docs/high-level-world-model.md) for the current world abstraction, [`docs/system-decomposition.md`](docs/system-decomposition.md) for module boundaries, and [`docs/architecture.md`](docs/architecture.md) for the broader architecture.

## Current B.1 world model

- One grid cell represents an area of field.
- Obstacles can span any number of cells, such as a three-cell-wide river.
- Multiple robots may occupy the same cell.
- Robots move with `Move(to=CellId)`; there is no turn/facing model.
- Each robot has its own speed in metres per second.
- Movement time is calculated from cell distance, cell size and robot speed.
- Infestation is a per-cell severity value, suitable for later dynamic pest-spread models.
- Robots receive local obstacle and infestation observations; the algorithm can turn those into potential-field attraction and repulsion values.

## Requirements

- [uv](https://docs.astral.sh/uv/) for environment and dependency management.
- Python 3.12 (uv will provision it from `.python-version`).

## Setup

```bash
uv sync
```

## Development commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
```

Run all checks together:

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyrefly check && uv run pytest
```

## Demonstrations

Small WorldState demonstration:

```bash
uv run python examples/worldstate_demo.py
```

Simple animated field-cell demonstration:

```bash
uv run --with matplotlib python examples/simple_world_sim.py
```

Hundreds-of-robots potential-field proof of concept:

```bash
uv run --with matplotlib python examples/mass_swarm_visualization.py
```

The mass-swarm demo defaults to 300 robots. Try 500 or more:

```bash
uv run --with matplotlib python examples/mass_swarm_visualization.py --robots 500 --seed 7
```

## Project layout

```text
src/swarm_field/   # contracts, swarm brain, simulation and visualization adapter
tests/             # pytest suite
examples/          # executable demonstrations
docs/              # design documents and source material
```
