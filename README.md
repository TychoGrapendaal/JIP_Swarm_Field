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
- A robot senses the crop, the infestation and the peers of the cell it stands on, and the terrain of that cell and its neighbours. A cell is a patch of field (about 10 × 10 m) that the robot monitors as a whole, while obstacles such as a river are large enough to see from the next cell.
- A robot does not know about obstacles or infestations elsewhere unless it starts with a map or learns them from peers.
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

Time how much of a pest run goes to knowledge sharing, at 10, 20 and 40 robots (not part of `pytest`):

```bash
uv run python -m tests.benchmark_sharing --cases 10x32 20x32 40x64
```

Compare variants and parameters of the EXPLORE seeder: five robots exploring a 32 × 32 field for three simulated hours, four seeds per variant, spread over processes (takes about ten minutes on 20 cores; not part of `pytest`). The results are in [`docs/explore.md`](docs/explore.md):

```bash
uv run python -m studies.explore_variants --size 32 --hours 3 --seeds 4 --jobs 20
```

## Studies

`examples/` shows what the simulation and its tools can do; [`studies/`](studies/README.md) holds the experiments behind the report, one script per research question. A study runs fixed configurations with fixed seeds and writes every run's series and a summary row per run under `runs/<study>/`:

```bash
uv run python -m studies.radio_range --ranges 50 150 300 --seeds 3 --hours 1
```

See [`studies/README.md`](studies/README.md) for the list of studies and how to write one.

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

Five scouts explore and keep monitoring a field with the EXPLORE seeder, on the interactive console of the [scenario viewer](#scenario-viewer-the-interactive-console-c1), so pausing, stepping, changing the speed, injecting a pest and inspecting a robot work the same way. The overlay shades how stale each cell is in ground truth, trails show where the robots went, dashed lines the target each robot is heading for, and a panel the coverage measures (coverage, idleness, redundancy). `--no-discount` turns off the peer discount, `--block-side 16` uses 16 × 16 blocks, and `--obstacles` adds a river with one bridge. Record a video instead of opening a window:

```bash
uv run python examples/explore_demo.py --obstacles --save explore.mp4
```

Both demos take the console's `--seed`, `--hours`, `--speed` (default 40, twenty simulated seconds per frame) and `--save`, but not `--record` and `--replay`: a recording holds the scenario's name and seed, not the demo's options, so it could not rebuild the run. The overlay, trails, targets and side panel come from `swarm_field.visualization.overlays`, and any script can hand them to `InteractiveConsole`.

### Pest demonstration

```bash
uv run python examples/pest_demo.py
```

Five scouts explore and three killers explore and treat a field with a river and two rocks, while pests appear at random free cells over time. Every robot is a real `ComposedAgent` with a `UtilityArbiter`: scouts follow the EXPLORE field with a mock behaviour that always bids 0.5, and killers weigh it against `TreatInfestation` on the TREAT field. Robots share their memory and status only within radio range. The panel shows what each killer is doing and how many pests it knows of. `--killers-treat-only` leaves the explore behaviour out of the killers, so they wait until they hear of a pest. The panel also shows the measures of `MetricsCollector`: infested cells, detection latency, coverage, idleness and redundancy. `--radio`, `--pests-per-hour`, `--spread`, `--scouts` and `--killers` change the setup, and `--save videos/pests.mp4` records a video. The killers' time per goal is weighted by simulated time, so pausing or changing the speed does not change it. `PestScenario` in `swarm_field.experiments` describes the run, and `swarm_field.runner` assembles and runs it.

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

### Scenario viewer: the interactive console (C.1)

`examples/scenario_viewer.py` opens any scenario of the catalogue (see [Scenario catalogue](#scenario-catalogue)) in a steerable window, with each robot's trail and a dashed line to the cell it is heading for. It is a thin command line around the console, `swarm_field.console`, which steers an `experiments.LiveRun`: the same run, built by `runner.build_run`, that `runner.run_experiment` would produce, advanced in slices so that it can be paused, stepped and steered. It defaults to `S2` with seed `101` for the scenario's own horizon, and it records the run's metrics.

```bash
uv run python examples/scenario_viewer.py
```

```bash
uv run python examples/scenario_viewer.py --scenario S3 --seed 7 --hours 6
```

```bash
uv run python examples/scenario_viewer.py --scenario pests
```

In the window:

- Click an empty cell to inject an infestation (`InjectInfestation` command).
- Click a robot to inspect its brain state via `SwarmAgent.inspect()`: a read-only `AgentInspection` panel shows its current goal and a summary of its belief.
- `space` / `p` pause and resume, `s` advances exactly one simulation event, `+` / `-` double and halve the speed, and `0` resets it to 1x.
- `h` hides the status box in the bottom-left corner and the legend, so nothing covers the field; press `h` again to show them. Videos written with `--save` keep the status line and the legend, but leave out the list of controls.

The interactions flow through the typed `Command` port: `InjectInfestation` is applied by the simulation, and `Pause` / `StepOnce` / `SetTimeScale` pace the console's clock.

#### Flags

| Flag | Meaning | Default |
|---|---|---|
| `--scenario NAME` | Scenario of the catalogue to run. | `S2` |
| `--seed N` | Seed of a new run. | 101 |
| `--hours H` | Simulated hours the run lasts; it stops there. | the scenario's own horizon |
| `--speed X` | Simulated time per frame: 0.5 s times `X`. In a window this is the starting speed. | 4 |
| `--save PATH` | Run to the horizon without a window, into a `.gif`, an `.mp4` (needs [ffmpeg](https://ffmpeg.org/)) or, for any other path, a directory of PNG frames. | a window |
| `--record PATH` | When the run ends, write its recording to `PATH`. | |
| `--replay PATH` | Play back a recording instead of starting a new run. | |
| `--out DIR` | When the run ends, write its metrics series and a summary row to `DIR`. | |

Each flag controls one thing, which is why there are several: `--scenario`, `--seed` and `--replay` choose the run; `--hours` sets how long it lasts and `--speed` how much of it one frame shows; and the run can produce three different outputs, which can be combined:

- `--save` is what the run looks like: a video or PNG frames.
- `--record` is what the user did: a small JSON `CommandLog` with the scenario, the seed, the time the run ended and every injected pest with its simulation time. A run is a function of its seed, so this is enough to replay it exactly, and to render it again later at another speed. Pause, step and speed changes are not recorded, because they change how the run is shown, not what happens in it.
- `--out` is what the run measured: the metrics series as CSV and JSON, and a summary row in the same format as the sweeps below.

A replay takes its scenario, seed and length from the recording, so combining `--replay` with `--scenario`, `--seed`, `--hours` or `--record` is refused. During a replay, pausing, stepping, changing the speed and inspecting still work, but injecting is disabled so the run stays the recorded one.

#### Recording, replaying and exporting

Record a session in a window; the recording is written when the window is closed:

```bash
uv run python examples/scenario_viewer.py --record recordings/s2.json
```

Replay it in a window, or render it into a video:

```bash
uv run python examples/scenario_viewer.py --replay recordings/s2.json
```

```bash
uv run python examples/scenario_viewer.py --replay recordings/s2.json --speed 16 --save runs/s2.mp4
```

Run a whole experiment in the background, and get the video and the metrics of that same run:

```bash
uv run python examples/scenario_viewer.py --scenario S2 --hours 6 --speed 16 --save runs/s2.mp4 --out runs/s2
```

Without a window the run goes to its horizon, the scenario's own or `--hours`, and stops. The video plays at 16 frames per second, which gives these lengths:

| `--speed` | Simulated time per frame | Frames per simulated hour | Video length per simulated hour |
|---|---|---|---|
| 4 | 2 s | 1800 | about 1 min 53 s |
| 16 | 8 s | 450 | about 28 s |
| 64 | 32 s | 113 | about 7 s |

Drawing a frame takes roughly a tenth of a second, so a long run at a low speed takes a few minutes to record.

### Scenario catalogue

Every scenario has a name, and every script that runs scenarios takes them by that name: the presets `S1` to `S5` of the scenario builder, and the hand-built experiments `explore` (`ExploreScenario`, scouts monitoring a field) and `pests` (`PestScenario`, scouts and killers while pests keep appearing). `swarm_field.experiments.build_named_experiment(name, seed, until)` turns a name and a seed into the `ExperimentConfig` the runner runs, labelled `NAME/SEED`. Each scenario keeps its own horizon unless one is given: an hour for the presets, three hours for `explore` and `pests`. `scenario_names()` lists them, and `tests/test_catalogue.py` checks that every one of them reproduces its run exactly from the same seed.

### Seeded scenario sweep with metrics (B.5)

```bash
uv run python examples/metrics_sweep_demo.py
```

Runs the cross product of three scenarios and two seeds, and prints one row per run followed by a full metrics series. Each run is reproducible from its seed: the same seed and configuration produce the same series. `--scenarios` takes any names of the catalogue:

```bash
uv run python examples/metrics_sweep_demo.py --scenarios S2 explore pests --hours 1
```

Every summary row also records how its run was set up (`RunMetadata`): the horizon, the decision and sampling intervals, the field, the robots per body type, each brain by its parts, whether robots started with the terrain map, the pest dynamics, and the failure and position-error settings. These columns follow the result columns, so a row can be traced back to its setup without the script that produced it. Brain parameters such as an `ExploreConfig` are named but not spelled out.

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
uv run python examples/robustness_demo.py --scenario pests --hours 1
uv run python examples/robustness_demo.py --out runs/robustness
```

**What this currently shows:** `active_robots` and `energy/m²` respond to the faults as intended, and the pest dynamics are provably unaffected. The outcome metrics (`coverage`, `targets_remaining`, `detection_latency`) are flat, because the default brain only treats: `TreatInfestation` bids only on pests the robot knows of, a robot senses only its own cell, and nothing explores, so robots that start away from a pest idle from the first decision. [`docs/research/robustness.md`](docs/research/robustness.md) explains what has to land before the sweep measures resilience rather than capacity.

## Current demonstration limitations

- Goal-directed navigation and exploration use potential fields over the robot's believed terrain, with a reactive wall-follow as their fallback. Exploration (`ExploreSeeder`, see [`docs/explore.md`](docs/explore.md)) has no behaviour of its own yet; the demo and the benchmark follow its field with a stand-in.
- Navigation is planned over what the robot believes, so a wrong obstacle belief is followed exactly; there is no observation-based recovery from it yet.
- Coarse-grained infestation dynamics are implemented as a scheduled `WorldTick`; the current model supports severity growth, neighbour spread, and stochastic spawning.
- Sensing covers the robot's own cell, and the terrain of its neighbours; obstacles further away are unknown unless the robot starts with the terrain map, and infestations elsewhere unless peers share them.
- Robot batteries are drained by a coarse cost model (per metre, per decision, per treat) and robot failure and position error can be injected for robustness runs, but there is no charging station, no consumable replenishment and no recovery from a failure.
- The visualization adapter renders read-only snapshots; interactive time control, infestation injection, robot inspection, and recording and replaying a run live in `swarm_field.console` and `experiments.LiveRun`, with `examples/scenario_viewer.py` as their command line (C.1).

## Project layout

```text
src/swarm_field/   # contracts, swarm brain, simulation, runner and visualization adapter
tests/             # pytest suite
examples/          # executable demonstrations
studies/           # experiments behind the report, one per research question
docs/              # design documents, research notes and source material
```
