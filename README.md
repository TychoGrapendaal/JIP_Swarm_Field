# Swarm Field

A decentralized swarm algorithm and simulation for robots cultivating and monitoring a crop field. The simulation is deliberately coarse-grained: one grid cell represents a field patch rather than one plant, and the model is intended to scale to hundreds of robots.

See [`docs/high-level-world-model.md`](docs/high-level-world-model.md) for the current world abstraction, [`docs/system-decomposition.md`](docs/system-decomposition.md) for module boundaries, and [`docs/architecture.md`](docs/architecture.md) for the broader architecture.

## Current B.1 world model

- One grid cell represents an area of field rather than an individual plant.
- Obstacles can span any number of cells, such as a three-cell-wide river.
- Multiple robots may occupy the same cell; there is no cell-capacity limit.
- Robots move directly with `Move(to=CellId)`; there is no turn/facing model.
- Each robot has its own movement speed in metres per second.
- Movement time is calculated from the distance between cells, cell size, and robot speed.
- Infestation is represented as a per-cell severity value, providing a coarse-grained basis for later dynamic pest-spread models.
- A robot senses only the cell it stands on: its crop, infestation and terrain, and the peers sharing it. A cell is a patch of field (about 10 × 10 m) that the robot monitors as a whole.
- A robot does not know about obstacles or infestations elsewhere unless it starts with a map or learns them from peers.
- The B.3 range-based communication channel is implemented; higher-level belief sharing is still the algorithm’s responsibility, so distant infestation information only propagates when agents choose to send messages.

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

Time the quadtree potential field against a plain Dijkstra over every cell, and a robot's trip with one and with two fields (takes about two minutes; not part of `pytest`):

```bash
uv run python -m tests.benchmark_field --sizes 100 250 500
```

Compare variants and parameters of the EXPLORE seeder: five robots exploring a 32 × 32 field for three simulated hours, four seeds per variant, spread over processes (takes about ten minutes on 20 cores; not part of `pytest`). The results are in [`docs/explore.md`](docs/explore.md):

```bash
uv run python -m tests.benchmark_explore --size 32 --hours 3 --seeds 4 --jobs 20
```

## Demonstrations

> **Note:** The examples below are demonstration and visualization tools for the current world model and potential-field prototype. They are not the final swarm exploration or communication implementation.

### Small WorldState demonstration

```bash
uv run python examples/worldstate_demo.py
```

Terminal-based demonstration of the core `WorldState`. It shows field-cell state, multiple robots sharing a cell, different robot speeds, direct target-cell movement, obstacles, local observations, infestation, and treatment actions.

### Simple animated field-cell demonstration

```bash
uv run --with matplotlib python examples/simple_world_sim.py
```

A small animated visualization showing the coarse-grained field representation, multi-cell obstacles, infested areas, and direct cell-to-cell robot movement.

### Hundreds-of-robots potential-field proof of concept

```bash
uv run --with matplotlib python examples/mass_swarm_visualization.py
```

The main large-scale visualization. It demonstrates hundreds of robots moving through a coarse-grained field using a simple local potential-field controller. Known infestations provide attraction, known obstacles provide repulsion, and a small random component provides a basic exploration baseline. Robots move directly toward selected target cells at their own speeds, and multiple robots can occupy the same cell.

The mass-swarm demo defaults to 300 robots. Try 500 or more:

```bash
uv run --with matplotlib python examples/mass_swarm_visualization.py --robots 500 --seed 7
```

For an even larger test:

```bash
uv run --with matplotlib python examples/mass_swarm_visualization.py --robots 1000 --seed 7
```

### Exploration demonstration (#41)

```bash
uv run python examples/explore_demo.py
```

Five scouts explore and keep monitoring a field with the EXPLORE seeder, drawn by the general-purpose `SimulationViewer`. The overlay shades how stale each cell is in ground truth, trails show where the robots went, dashed lines the target each robot is heading for, and a panel the coverage measures (coverage, idleness, redundancy). `--no-discount` turns off the peer discount, `--block-side 16` uses 16 × 16 blocks, and `--obstacles` adds a river with one bridge. Record a video instead of opening a window:

```bash
uv run python examples/explore_demo.py --obstacles --save explore.mp4
```

`SimulationViewer` (in `swarm_field.visualization`) works for any run: give it a function from a time to a `WorldSnapshot` (e.g. `ExploreRun.snapshot_at`), and optionally a per-cell overlay, the robots' goals and the lines of the panel.

### Wall-following steering demonstration

```bash
uv run --with matplotlib python examples/steering_demo.py
```

A single land robot running the reactive A.3 `Steerer` towards a target across an obstacle wall, drawing the cells it visited. The steerer is the mapless policy: gradient alignment towards the goal, and a committed right-hand wall-follow whenever that heading is blocked. In the brain it is the fallback behind `Navigator.step_towards`, which prefers a potential field over the robot's believed terrain. See [`docs/navigation-module.md`](docs/navigation-module.md) §11.1 and [`docs/robot-memory-and-navigation.md`](docs/robot-memory-and-navigation.md).

### Event-driven robot decision demonstration (B.1 / issue #4)

```bash
uv run --with matplotlib python examples/robot_decision_demo.py
```

An animated demonstration of the real discrete-event robot loop. A scripted robot is scheduled as a `RobotDecision` event and repeatedly goes through `Perceive → Decide → Act → Re-queue` at a fixed decision interval. The visualization shows the robot moving between cells, its current action, the last decision time, and the next event waiting in the queue.

Write animation frames as PNGs instead of opening a window:

```bash
uv run --with matplotlib python examples/robot_decision_demo.py --frames 200 --save frames
```

This demo is specifically intended to make the event-driven behavior added in issue #4 visible; it is separate from the higher-level swarm behavior demonstrations.

### Treat-infestation demonstration

```bash
uv run python examples/treat_infestation_demo.py
```

A killer robot that already knows of three pests follows the TREAT field around a wall to each of them, treats it once, and then idles.

### Renderer demonstration (C.1)

```bash
uv run --with matplotlib python examples/renderer_demo.py
```

Draws a running scenario through the `Renderer` port: each `WorldState.snapshot` is rendered as read-only data, covering the field cells (crop growth, infestation severity, obstacles), the robots, and their movement targets. The demo runs an all-scout roster over a fully cropped field with infestation patches of different severities.

Write frames as PNGs instead of opening a window:

```bash
uv run --with matplotlib python examples/renderer_demo.py --frames 60 --save frames
```

## Current demonstration limitations

- Goal-directed navigation and exploration use potential fields over the robot's believed terrain, with a reactive wall-follow as their fallback. Exploration (`ExploreSeeder`, see [`docs/explore.md`](docs/explore.md)) has no behaviour of its own yet; the demo and the benchmark follow its field with a stand-in.
- Navigation is planned over what the robot believes, so a wrong obstacle belief is followed exactly; there is no observation-based recovery from it yet.
- The communication channel is implemented, but the current utility behaviour does not yet use peer messages to follow a pest found by another robot.
- Coarse-grained infestation dynamics are implemented as a scheduled `WorldTick`; the current model supports severity growth, neighbour spread, and stochastic spawning.
- Sensing covers the robot's own cell only; obstacles elsewhere are unknown unless the robot starts with the terrain map, and infestations elsewhere unless peers share them.
- The visualization adapter only renders snapshots so far; interactive inspection, time control, and injecting infestations are not implemented yet.

## Project layout

```text
src/swarm_field/   # contracts, swarm brain, simulation and visualization adapter
tests/             # pytest suite
examples/          # executable demonstrations
docs/              # design documents and source material
```