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
- Robots receive local terrain, infestation, crop, and peer observations according to their body-specific sensing profiles. The sensing footprints differ by body type.
- A robot does not have the complete world state; facts outside its local observation range are unknown unless they are learned through messages or provided by the initial configuration.
- Runs are reproducible: one seeded random source drives the world, a second drives fault injection, and `runner.py` builds every run from a `BrainSpec` so algorithms can be compared as configuration.
- The B.3 range-based communication channel is implemented; higher-level belief sharing is still the algorithm’s responsibility, so distant infestation information only propagates when agents choose to send messages.
- The body capability model defines the permitted actions and simulation assumptions for SCOUT, KILLER, and BATTERY robots. See [`docs/body-capabilities.md`](docs/body-capabilities.md) for the profiles, the capability gate, and current limitations.

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

## Studies

`examples/` shows what the simulation and its tools can do; [`studies/`](studies/README.md) holds the experiments behind the report, one script per research question. A study runs fixed configurations with fixed seeds through `swarm_field.experiments.Experiment` (a tutorial is [`docs/experiments.md`](docs/experiments.md)), which runs them in parallel and writes their results and tables to a new folder `output/<date>_<time>_<name>/`, which git ignores:

```bash
uv run python -m studies.radio_range --ranges 50 150 300 --seeds 3 --hours 1
```

Compare the four pest responses on the two fields with obstacles:

```bash
uv run python -m studies.benchmark_obstacles --seeds 12
```

Compare variants and parameters of the EXPLORE seeder: five robots exploring a 32 × 32 field for three simulated hours, four seeds per variant, spread over processes (takes about ten minutes on 20 cores). The results are in [`docs/explore.md`](docs/explore.md):

```bash
uv run python -m studies.explore_variants --size 32 --hours 3 --seeds 4 --jobs 20
```

Two studies time the code rather than the swarm, and print their table instead of writing to `output/`.

Time the quadtree potential field against a plain Dijkstra over every cell, and a robot's trip with one and with two fields (takes about two minutes):

```bash
uv run python -m studies.benchmark_field --sizes 100 250 500
```

Time how much of a pest run goes to knowledge sharing, at 10, 20 and 40 robots:

```bash
uv run python -m studies.benchmark_sharing --cases 10x32 20x32 20x64 40x64
```

See [`studies/README.md`](studies/README.md) for the list of studies and how to write one.

## Running an experiment from a JSON file

To run an experiment without writing code, describe it in a JSON file and run that file. [`configs/`](configs/) has four to copy: `pests.json` (one setup), `radio_range.json` (several conditions compared over the same seeds), `seeded_vs_hop.json` (the map-based seeded killers against the hop-based killers, switched by the robot counts in `swarm`) and `explore.json`.

```bash
uv run python -m swarm_field.experiments.from_json configs/radio_range.json
```

```json
{
  "name": "radio-range",
  "scenario": "pests",
  "seeds": [0, 1, 2],
  "hours": 1,
  "jobs": 6,
  "parameters": {"swarm": {"explorer": 5, "seeded-killer": 3}},
  "conditions": {
    "radio-50": {"robots": {"radio_range_m": 50}},
    "radio-300": {"robots": {"radio_range_m": 300}}
  }
}
```

- `scenario` is a ready scenario: `pests`, `explore`, `river-small`, `rivers-large` or `relayed-killers`. Every run starts from it.
- `parameters` changes it for every run, and each entry of `conditions` (optional) changes it further for one condition. Every condition runs with every seed.
- Parameters come in groups: `map`, `pests`, `robots` and `metrics` take the fields of `MapConfig`, `PestConfig`, `RobotConfig` and `MetricsConfig` (see [`docs/experiments.md`](docs/experiments.md)), with their names and units, so times are in milliseconds. `explore` takes the explorers' `ExploreConfig` fields, and `swarm` the number of robots of each preset by name (`explorer`, `relay`, `seeded-killer`, `hop-killer`); a count of 0 leaves a preset out, which is how a condition switches the killers' navigation, as in `seeded_vs_hop.json`. Hop killers cannot find a dock yet, so give them a large `robots.battery_capacity` and `pests.treatment_doses` for a fair comparison. List them all with their values with `uv run python -m swarm_field.experiments.from_json --list pests`. An unknown name is an error rather than silently ignored.
- `hours` sets every run's length in whole hours, `warmup_hours` how much of it is left out of the infested cells (0 by default), and `jobs` how many runs go at once (one per CPU by default).

Every run is measured with `measure_pest_response`. The run prints one row per condition, and writes every run's result, the table and a copy of the JSON file to a new folder in `output/`.

## Integrated battery and live-field demonstrations

The Issue #77 battery lifecycle adds simulated-time energy depletion, exclusive dock charging with
FIFO queues, and explicit off-dock `DEAD_BATTERY` failures while preserving the current behaviour,
arbitration, potential-field and discrete-event architecture. See
[`docs/battery-lifecycle.md`](docs/battery-lifecycle.md) for policy and test details.

The styled live simulation shows the actual runner-built swarm, selected robot's potential field,
infestation, docks, radio links and battery telemetry in one interactive window. It does not save
an animation by default:

```bash
uv run --with matplotlib python examples/field_simulation_demo.py
```

See [`docs/live-field-simulation-demo.md`](docs/live-field-simulation-demo.md) for controls and the
components the view observes.

## Demonstrations

> **Note:** The examples below are demonstration and visualization tools for the current world model and potential-field prototype. They are not the final swarm exploration or communication implementation.

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

Five explorers explore and three killers treat a field with a river and two rocks, while pests appear at random free cells over time. Every robot is a real `ComposedAgent` with a `UtilityArbiter`: explorers follow the EXPLORE field, and seeded killers treat the pests they know of with `TreatInfestation` on the TREAT field. `--hop` makes the killers hop killers, which follow the hop beacons every robot relays from the explorer that sees a pest. Robots share their memory and status only within radio range. The panel shows what each killer is doing and how many pests it knows of. The panel also shows the measures of `MetricsCollector`: infested cells, detection latency, coverage, idleness and redundancy. `--radio`, `--pests-per-hour`, `--spread`, `--explorers` and `--killers` change the setup, and `--save videos/pests.mp4` records a video. The killers' time per goal is weighted by simulated time, so pausing or changing the speed does not change it. `SwarmScenario` in `swarm_field.experiments` describes the run, and `swarm_field.runner` assembles and runs it.

### Role rebalancing (#64)

```bash
uv run python examples/scenario_viewer.py --scenario rebalancing
uv run python examples/scenario_viewer.py --scenario rebalancing-fixed
uv run python -m studies.rebalancing --seeds 4
```

Robots switch between explorer and killer as the pests need. `rebalancing` is the `REBALANCING` swarm (eight explorers and six seeded killers on the river field, a dock in every corner, 20 pests an hour) with every robot a `RebalancingPreset`; `rebalancing-fixed` is the same swarm with fixed roles. A rebalancing robot carries both roles' behaviours, each wrapped in a `BodyGate` (`"body_gates"` in its `BrainSpec`) so it explores only as a scout and treats only as a killer, plus `RebalanceRoles`. From its own memory it counts the explorers, killers and pests it knows of; the swarm needs `min_killers` plus one killer per `pests_per_killer` known pests, but always keeps the `min_monitors` explorers the field needs (`monitors_needed`). When the split stays off for `persistence`, only the robots best placed to switch bid `SwitchRole` (the explorers nearest a known pest, or the killers furthest from any), so the swarm does not switch all at once; a robot then keeps its new role for `role_dwell`. The world gives it the other body and that body's speed and keeps it busy for `RobotConfig.role_switch_time`. The viewer colours robots by body, so every switch shows. `studies.rebalancing` runs both swarms on the same seeds; see [`docs/research/rebalancing.md`](docs/research/rebalancing.md).

### Wall-following steering demonstration

```bash
uv run --with matplotlib python examples/steering_demo.py
```

A single land robot running the reactive A.3 `Steerer` towards a target across an obstacle wall, drawing the cells it visited. The steerer is the mapless policy: gradient alignment towards the goal, and a committed right-hand wall-follow whenever that heading is blocked. In the brain it is the fallback behind `Navigator.step_towards`, which prefers a potential field over the robot's believed terrain. See [`docs/navigation-module.md`](docs/navigation-module.md) §11.1 and [`docs/robot-memory-and-navigation.md`](docs/robot-memory-and-navigation.md).

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
| `--out DIR` | When the run ends, write its metrics series to `DIR`. | |

Each flag controls one thing, which is why there are several: `--scenario`, `--seed` and `--replay` choose the run; `--hours` sets how long it lasts and `--speed` how much of it one frame shows; and the run can produce three different outputs, which can be combined:

- `--save` is what the run looks like: a video or PNG frames.
- `--record` is what the user did: a small JSON `CommandLog` with the scenario, the seed, the time the run ended and every injected pest with its simulation time. A run is a function of its seed, so this is enough to replay it exactly, and to render it again later at another speed. Pause, step and speed changes are not recorded, because they change how the run is shown, not what happens in it.
- `--out` is what the run measured: the metrics series as `metrics.csv` and `metrics.json`.

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

Every scenario has a name, and every script that runs scenarios takes them by that name: the presets `S1` to `S5` of the scenario builder, and the hand-built experiments `explore` (`EXPLORE`, explorers monitoring a field) and `pests` (`PESTS`, explorers and seeded killers while pests keep appearing), both a `SwarmScenario`. `swarm_field.experiments.build_named_experiment(name, seed, until)` turns a name and a seed into the `ExperimentConfig` the runner runs, labelled `NAME/SEED`. Each scenario keeps its own horizon unless one is given: an hour for the presets, three hours for `explore` and `pests`. `scenario_names()` lists them, and `tests/test_catalogue.py` checks that every one of them reproduces its run exactly from the same seed.

### Running many configurations

A batch of runs, every configuration with every seed, goes through `swarm_field.experiments.Experiment`, which runs them in parallel and writes their results to a new folder in `output/`; the tutorial is [`docs/experiments.md`](docs/experiments.md). The robustness demo is one:

```bash
uv run python examples/robustness_demo.py --scenario pests --hours 1 --seeds 3
```

It runs every failure mode and position-error rate against one scenario of the catalogue, and `--render CONDITION` writes PNG frames of one condition instead:

```bash
uv run --with matplotlib python examples/robustness_demo.py --render failure-offline-0.005
```


## Current demonstration limitations

- An experiment builds every brain from a `BrainSpec`, whose parts the runner registers: explorers run `ExploreBehaviour`, and killers `TreatInfestation` or the hop method. The dock and battery behaviours, `ReturnToDock` and `SupplyBattery`, are not registered yet, so only a `ComposedAgent` built by hand uses them. These are the current simulation behaviours, not a guarantee of optimal coverage or pest-control performance.
- Goal-directed navigation uses potential fields over the robot's believed terrain, with a reactive wall-follow as its fallback. A wrong obstacle belief can therefore still lead to a wrong route; there is no general observation-based recovery from it yet.
- Coarse-grained infestation dynamics are implemented as a scheduled `WorldTick`; the current model supports severity growth, neighbour spread, and stochastic spawning.
- Sensing is local and body-specific. Robots do not receive the complete world state; observations beyond their sensing footprints depend on information being shared through communication.
- Battery usage, endurance, dock charging, treatment duration, and replacement-battery exchanges use coarse simulation assumptions. Idle and sensing energy costs are not yet modelled. Robot failure and position error can still be injected for robustness runs, and general recovery from a failure is not implemented.
- The body-capability model does not yet include a dedicated battery-request protocol or exclusive service reservation. See [`docs/body-capabilities.md`](docs/body-capabilities.md) for the model's current limitations.
- The visualization adapter renders read-only snapshots; interactive time control, infestation injection, robot inspection, and recording and replaying a run live in `swarm_field.console` and `experiments.LiveRun`, with `examples/scenario_viewer.py` as their command line (C.1).

## Project layout

```text
src/swarm_field/   # contracts, swarm brain, simulation, runner and visualization adapter
tests/             # pytest suite
examples/          # executable demonstrations
studies/           # experiments behind the report, one per research question
configs/           # experiments described in JSON, for swarm_field.experiments.from_json
docs/              # design documents, research notes and source material
```
