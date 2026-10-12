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
- Robots receive local obstacle and infestation observations and can use this information to calculate potential-field attraction and repulsion.
- A robot does not currently know about obstacles or infestations outside its local observation range.
- Robot-to-robot communication is delivered peer-to-peer through the `Channel` (inbox and broadcast within radio range), but no behavior encodes or decodes anything yet, so no information actually propagates through the swarm.
- Runs are reproducible: one seeded random source drives the world, a second drives fault injection, and `runner.py` builds every run from a `BrainSpec` so algorithms can be compared as configuration.
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

### Renderer demonstration (C.1)

```bash
uv run --with matplotlib python examples/renderer_demo.py
```

Draws a running scenario through the `Renderer` port: each `WorldState.snapshot` is rendered as read-only data, covering the field cells (crop growth, infestation severity, obstacles), the robots, and their movement targets. The demo runs an all-scout roster over a fully cropped field with infestation patches of different severities.

Write frames as PNGs instead of opening a window:

```bash
uv run --with matplotlib python examples/renderer_demo.py --frames 60 --save frames
```

### Interactive console (C.1)

```bash
uv run python examples/interactive_demo.py
```

An interactive running scenario: robots act through the discrete-event loop (`RobotDecision` events drive `SwarmAgent.decide`) while pests spread via `WorldTick`. The window is steerable:

- Click an empty cell to inject an infestation (`InjectInfestation` command).
- Click a robot to inspect its brain state via `SwarmAgent.inspect()` — a read-only `AgentInspection` panel shows the robot's current goal and belief.
- `space` / `p` pause and resume, `s` advances exactly one simulation event, `+` / `-` and `0` change the simulation speed.

The interactions flow through the typed `Command` port (`InjectInfestation` is applied by the simulation; `Pause` / `StepOnce` / `SetTimeScale` pace the render clock). Headless frame generation works like the renderer demo:

```bash
uv run python examples/interactive_demo.py --frames 240 --save frames/interactive
```

#### Recording and replaying a session

Add `--record` to save a session. The window works exactly as without it; the recording is written when the window is closed:

```bash
uv run python examples/interactive_demo.py --record recordings/demo.json
```

A recording is a small JSON `CommandLog`: the seed, the simulation time the session ended at, and every injected pest with the simulation time it was applied at. Because a run is a function of its seed, this is enough to reproduce the session exactly. Pause, step and speed changes are not recorded, since they change how fast the run is shown, not what happens in it.

Replay a recording in a window:

```bash
uv run python examples/interactive_demo.py --replay recordings/demo.json
```

The replay plays at 1x from the start and stops where the recording ended. Pausing, stepping, changing the speed and inspecting robots still work, but injecting is disabled so the run stays the recorded one.

Or render the replay to a video instead of opening a window:

```bash
uv run python examples/interactive_demo.py --replay recordings/demo.json --save demo.mp4
```

`--save` decides the output by its suffix: `.gif` writes a GIF (no extra software needed), `.mp4` writes a video and needs [ffmpeg](https://ffmpeg.org/) on the `PATH`, and any other path is a directory of PNG frames. By default a replay renders its full length; `--frames N` renders only the first `N` frames, at 0.5 s of simulation time per frame.

| Flag | Meaning |
|---|---|
| `--seed N` | Seed of a new session (default 7); a replay uses the seed from its recording. |
| `--frames N` | Number of frames to run (default 240, or the full length of a replay). |
| `--save PATH` | Save instead of opening a window: `.gif`, `.mp4`, or a directory of PNGs. |
| `--record PATH` | Write the session's `CommandLog` to `PATH` when it ends. |
| `--replay PATH` | Play back a recording written by `--record`. |

### Watching a real experiment (C.1 over B.5)

```bash
uv run python examples/scenario_viewer.py
```

The interactive console above drives a hand-built three-robot world, so it can exercise the `Command` port on its own. This demo drives a *real* experiment instead: `runner.build_experiment` resolves the scenario preset and `runner.build_run` assembles the robot roster and brains, so the window shows the same object graph `runner.run_experiment` would drive. It defaults to `S2` with seed `101` for one simulated hour — the horizon `runner.sweep_scenarios` uses — and offers the same steering controls: click a cell to inject an infestation, click a robot for its `AgentInspection` panel, and `space` / `s` / `+` / `-` / `0` to pause, step, and change speed. The run stops at its horizon rather than looping forever.

```bash
uv run python examples/scenario_viewer.py --scenario S3 --seed 7 --hours 6
```

A scenario preset carries no duration of its own, so `--hours` sets the horizon. The metrics cadence is shortened to match the pest dynamics, because the default of one sample per hour would leave the figures in the title static for the whole run; cadence changes what is measured, never what the world does. Headless frame generation works like the other demos, and stops when the run reaches its horizon:

```bash
uv run python examples/scenario_viewer.py --frames 240 --save frames/viewer
```

#### Recording an experiment in the background

Give `--save` a video file to record a whole experiment without opening a window. The run goes to its `--hours` horizon, writes the video and stops:

```bash
uv run python examples/scenario_viewer.py --scenario S2 --hours 6 --speed 16 --save runs/s2.mp4
```

Like in the interactive console, `.gif` needs no extra software and `.mp4` needs [ffmpeg](https://ffmpeg.org/) on the `PATH`. `--speed` sets how much simulated time one frame covers: 0.5 s times the speed, so the default speed of 4 shows 2 s per frame. The video plays at 16 frames per second, which gives these lengths for a run of `H` hours:

| `--speed` | Simulated time per frame | Frames per simulated hour | Video length per simulated hour |
|---|---|---|---|
| 4 (default) | 2 s | 1800 | about 1 min 53 s |
| 16 | 8 s | 450 | about 28 s |
| 64 | 32 s | 113 | about 7 s |

Drawing a frame takes roughly a tenth of a second, so a long run at a low speed takes a few minutes to record. `--frames N` caps the video at `N` frames; the run then stops before its horizon. Without `--save`, `--speed` is the speed the window starts at.

### Seeded scenario sweep with metrics (B.5)

```bash
uv run python examples/metrics_sweep_demo.py
```

Runs the cross product of three scenarios and two seeds, and prints one row per run followed by a full metrics series. Each run is reproducible from its seed: the same seed and configuration produce the same series.

Write the summary table and every run's series to a directory (CSV and JSON):

```bash
uv run python examples/metrics_sweep_demo.py --out runs/sweep
```

### Rendering a sweep (C.1 over B.5)

Both sweep demos take a `Renderer` and draw the run on every metric tick, so a batch run can be watched instead of only tabulated:

```bash
uv run --with matplotlib python examples/metrics_sweep_demo.py --render
uv run --with matplotlib python examples/robustness_demo.py --render failure-offline-0.005
```

Frames land in `runs/frames` (override with `--render-out`), one PNG per metric tick, with the current metrics in the title. In a robustness run a lost robot is drawn hollow and grey, so the frames show the swarm shrinking rather than only reporting it. Rendering only reads ground truth: the same run with and without a renderer produces an identical metrics series.

The ticks come from the metrics cadence rather than a render clock, because the runner drives the event loop to completion and there is nothing left to animate afterwards. For a window on a run you steer yourself — pause, step, speed, injection — use the scenario viewer.

### Robustness sweep: robot failure and position error (B.5)

```bash
uv run python examples/robustness_demo.py
```

Runs one scenario under a baseline and then one fault at a time: three failure modes at two rates, and two position-error rates. Every condition runs from the same seed, and faults draw from a random source separate from the world's, so a condition and its baseline see identical pest dynamics and the difference in the metrics is attributable to the fault.

```bash
uv run python examples/robustness_demo.py --scenario S5 --hours 3
uv run python examples/robustness_demo.py --out runs/robustness
```

**What this currently shows:** `active_robots` and `energy/m²` respond to the faults as intended, and the pest dynamics are provably unaffected. The outcome metrics (`coverage`, `targets_remaining`, `detection_latency`) are flat, because the swarm's behaviors are still stubs — `TreatSeeder.seeds()` returns no targets, so every robot commits `Idle` from the first decision and the swarm never moves. [`docs/research/robustness.md`](docs/research/robustness.md) explains what has to land before the sweep measures resilience rather than capacity.

## Current demonstration limitations

- The exploration behavior is currently a simple baseline and is not yet a dedicated coverage/exploration algorithm.
- Goal-directed navigation uses a potential field over the robot's believed terrain, with a reactive wall-follow as its fallback. Exploration is being rebuilt on the same fields (#41).
- `TreatSeeder` is a stub that returns no target cells, so every robot commits `Idle` and the swarm does not move. Coverage and detection-latency metrics are therefore flat in the sweeps; see [`docs/research/robustness.md`](docs/research/robustness.md).
- Navigation is planned over what the robot believes, so a wrong obstacle belief is followed exactly; there is no observation-based recovery from it yet.
- Robot-to-robot communication carries messages but nothing encodes or decodes them yet, so `messages_sent` is zero and the utility robot cannot follow a pest found by another robot.
- Coarse-grained infestation dynamics are implemented as a scheduled `WorldTick`; the current model supports severity growth, neighbour spread, and stochastic spawning.
- Robot batteries are drained by a coarse cost model (per metre, per decision, per treat) and robot failure and position error can be injected for robustness runs, but there is no charging station, no consumable replenishment and no recovery from a failure.
- Sensing is currently local; distant obstacles and infestations are unknown unless information is later propagated through communication.
- The visualization adapter renders read-only snapshots; interactive time control, infestation injection, robot inspection and recording and replaying a session are demonstrated in `examples/interactive_demo.py` (C.1).

## Project layout

```text
src/swarm_field/   # contracts, swarm brain, simulation, runner and visualization adapter
tests/             # pytest suite
examples/          # executable demonstrations
docs/              # design documents, research notes and source material
```