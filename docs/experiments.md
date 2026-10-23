# Running experiments

An experiment compares a few configurations, the *arms*, over the same seeds. Each run of an arm with a seed is simulated, measured, and its result kept. This page shows how to set up the arms for the main question, the seeded pest response against the hop-based one, how to measure them, how to run every arm with every seed in parallel, where the results end up, and how to watch a run as a video.

Everything here is imported from `swarm_field.experiments`. A complete example is [`tests/benchmark_obstacles.py`](../tests/benchmark_obstacles.py).

## 1. Configure a run

A run is a `SwarmScenario` (`src/swarm_field/experiments/scenario.py`). It is built rather than spelled out: a field, pests, settings every robot shares, and a swarm added preset by preset. Start from a ready scenario, or from `SwarmScenario()`, an empty swarm on the small river field, and add robots:

```python
from swarm_field.experiments import EXPLORER, HOP_KILLER, RIVER_SMALL, SEEDED_KILLER

seeded = RIVER_SMALL.without_robots().add_robots(EXPLORER, 10).add_robots(SEEDED_KILLER, 4)
hop = RIVER_SMALL.without_robots().add_robots(EXPLORER, 10).add_robots(HOP_KILLER, 4)
```

Every method returns a new scenario, so the calls chain and a ready scenario is never changed. Robots get their ids in the order they are added.

| Method | Returns the scenario with |
| --- | --- |
| `add_robots(preset, count)` | `count` more robots of `preset`, after the others. |
| `without_robots(body)` | its robots of `body` (e.g. `BodyType.KILLER`) left out, or all robots without an argument. |
| `with_map(map_config)` | another field (`MapConfig`). |
| `with_pests(pests)` | other pests (`PestConfig`). |
| `with_robot_config(robots)` | other robot settings (`RobotConfig`). |
| `with_metrics(metrics)` | another metrics cadence (`MetricsConfig`). |

`count(name)` says how many robots of a preset a scenario has, `killers` how many killers of every preset. `until` (3 h) and `seed` are plain fields; set them with `dataclasses.replace(scenario, seed=3)`.

| Scenario | Field | Robots | Pests per hour |
| --- | --- | --- | --- |
| `RIVER_SMALL` | 32 × 32, one river with a bridge, two rocks | 64 relays, 6 seeded killers | 18 |
| `RIVERS_LARGE` | 64 × 64, two rivers with three bridges, eight rocks | 256 relays, 18 seeded killers | 48 |
| `RELAYED_KILLERS` | 32 × 32, no obstacles | 400 relays, 8 seeded killers | set per run (15 in the benchmark) |
| `PESTS` | 32 × 32, one river with a bridge, two rocks | 5 explorers, 3 seeded killers | 15 |
| `EXPLORE` | 32 × 32, no obstacles | 5 explorers | none |

The first three have 10 m cells, a decision every 15 s, ground killers at 0.5 m/s, relays that stand still on a jittered lattice, and a mock source on every pest that stands in for the explorer that found it. The numbers of killers and the pest rates were chosen when killers drove at 0.1 m/s, to overload them slightly; at 0.5 m/s the same swarm keeps up more easily, so raise the pest rate or lower the killers if a question needs an overloaded swarm.

### The robots: presets

A preset is a kind of robot: its name, its body, and its brain (`src/swarm_field/experiments/presets.py`). Four are ready:

| Preset | Class | Body | What it does | Parameters |
| --- | --- | --- | --- | --- |
| `EXPLORER` | `ExplorerPreset` | scout | Explores the field on the EXPLORE field and shares everything it knows. | `explore`: its `ExploreConfig` |
| `RELAY` | `RelayPreset` | scout | Stands still and only passes on what the swarm's killers read. | — |
| `SEEDED_KILLER` | `SeededKillerPreset` | killer | Plans a route to a pest it knows of over its map, and treats it. | `claims` (on), `window_blocks` |
| `HOP_KILLER` | `HopKillerPreset` | killer | Walks down the hop beacons to a pest, without a map. | `beacons`: its `BeaconConfig`, `hop-tuned` by default |

A variant is the class with other parameters, and a name of its own keeps it apart in tables:

```python
from swarm_field.experiments import ExplorerPreset, SeededKillerPreset
from swarm_field.swarm import ExploreConfig

plain_explorer = ExplorerPreset(explore=ExploreConfig(spread_peers=False), name="plain-explorer")
scenario = RIVER_SMALL.without_robots().add_robots(plain_explorer, 10).add_robots(
    SeededKillerPreset(claims=False), 4
)
```

`CustomPreset(name, body, brain_spec)` adds a robot with any `BrainSpec`, e.g. one with a new behaviour under test. Its brain is used as it is.

What a robot broadcasts follows from who listens. Every preset says what the swarm has to relay for it (`Relaying`), and the scenario merges that for all its presets. In a swarm with hop killers, or a custom preset whose brain has beacons, every robot relays hop beacons, so an explorer that sees a pest is its source. In a swarm with seeded killers, the relays keep the pests they hear and pass them on, with the killers' claims if they claim. A custom preset that needs the pests passed on says so with `pest_facts`.

### Settings every robot shares: `RobotConfig`

| Field | Meaning |
| --- | --- |
| `speed_mps_drone`, `speed_mps_ground` | The scouts' speed (2 m/s) and the killers' (0.5 m/s). |
| `radio_range_m` | How far every robot's broadcasts reach. |
| `decision_interval` | The time between two decisions, 15 s. |
| `knows_terrain` | Whether robots start with the map. In the obstacle scenarios they do not, and learn obstacles by sensing. |
| `lattice_starts` | Each group of robots starts on a jittered lattice of its own over the field, from the seed. Otherwise all robots start side by side along the bottom row. |

```python
from dataclasses import replace

scenario = RIVER_SMALL.with_robot_config(replace(RIVER_SMALL.robots, radio_range_m=100.0))
```

The sparsest relay lattice that keeps nearly all relays in one network is one relay every 4 cells at a 50 m radio: 64 relays on 32 × 32, 256 on 64 × 64.

### The field: `MapConfig`

| Field | Meaning |
| --- | --- |
| `size` | The field's side in cells. |
| `obstacles` | The blocked cells, as a tuple of `(x, y)`. `river_and_rocks(size)` and `two_rivers_and_rocks(size)` build the two ready maps; `()` is an open field, and `None`, the default, the river. |
| `cell_size` | The side of a cell in metres. |

```python
from swarm_field.experiments import RIVERS_LARGE, MapConfig, two_rivers_and_rocks

open_field = RIVER_SMALL.with_map(MapConfig(obstacles=()))
larger = RIVERS_LARGE.with_map(MapConfig(size=48, obstacles=two_rivers_and_rocks(48)))
```

### Pests and treatment: `PestConfig`

| Field | Meaning |
| --- | --- |
| `pests_per_hour` | How many pests appear per hour, on average, on the free cells. |
| `spawn_severity` | The severity of a new pest, 0.1 by default. |
| `growth_rate` | How much a pest's severity grows per `period` (one minute), 0.05 by default. |
| `treatment_rate` | How much severity a treating killer removes per hour. The ready pest-response scenarios use `FAST_TREATMENT` (60, a full pest in one minute). It must beat the growth, or no pest ever clears. |
| `treatment_doses` | The doses a killer carries, 10 by default; every treatment uses one. |
| `source_range_m` | How far the mock source on a pest reaches, or `None` for no sources; keep it equal to the radio. It sends what the killers read: pest facts for seeded killers, a beacon of 0 for hop killers. |

### The pest response: seeded or hop-based

`pest_response_configurations(scenario)` turns a scenario into the compared responses, by name: the same scenario with its killers, as many as it has of every preset, replaced by seeded or by hop killers. Use the two best ones:

```python
from swarm_field.experiments import pest_response_configurations

configurations = pest_response_configurations(scenario)
seeded = configurations["seeded"]
hop = configurations["hop-tuned"]
```

- **`seeded`** shares pest cells. Relays gossip the pests they know and the killers' claims; a killer plans a route to a remembered pest over the terrain it sensed, leaves a pest to a closer killer that claimed it, and does not join a pest another killer already treats.
- **`hop-tuned`** shares hop numbers only. A source beacons 0, every relay broadcasts its distance in hops, and a killer walks down the hop values to the pest; killers repel each other, pulls are weighted by the square root of the distance, and a killer stuck in place pushes itself out. See [`research/hop-field.md`](research/hop-field.md).

The dictionary also has `hop` and `hop-repel`, the hop method without its later improvements, and with `seeded_no_yield=True` it adds `seeded-no-yield`, the seeded method without the claims. Use them only when a question is about those differences.

### Time and seeds

How long a run lasts and what is measured is a `PestRunSpec`:

```python
from swarm_field.experiments import PestRunSpec

spec = PestRunSpec(seeded, hours=3, pests_per_hour=scenario.pests.pests_per_hour, warmup_hours=1)
```

| Field | Meaning |
| --- | --- |
| `scenario` | The configured scenario. |
| `hours` | How long a run lasts, in whole simulated hours. |
| `pests_per_hour` | The pest rate of the runs; it replaces the scenario's own. |
| `warmup_hours` | How long at the start is left out of the main outcome, 1 by default; it must be shorter than the run, so a 1-hour run needs `warmup_hours=0`. |

The seed is not part of the spec: it comes with each run (section 3), and it sets the pests, the relay and killer positions, and when each robot first decides. Two arms run with the same seed see the same relays and the same pests.

## 2. Metrics

`measure_pest_response` runs one spec with one seed and returns a `PestResponse`. It advances the run one decision interval at a time and samples the ground truth and the killers between steps:

| `PestResponse` field | Table column | Meaning |
| --- | --- | --- |
| `infested` | infested | Infested cells, averaged over the run after the warm-up. The main outcome: it counts how many pests stay and for how long. |
| `treated` | treated | Pests cleared during the run. |
| `open` | open | Pests still infested at the end. |
| `response_minutes` | response | Minutes from a pest's appearance to its clearing, over the cleared pests. |
| `response_p90` | p90 | The 90th percentile of those minutes. |
| `cells_per_treatment` | cells/treat | Cells moved by all killers per treated pest. |
| `messages_per_robot_hour` | msgs | Messages received per robot per hour. |
| `stuck` | stuck | The share of the time a killer committed to treating stands idle. |
| `commit_hops` | commit hop | A killer's hop value each time it commits to treating, for hop arms; the table shows the median. |
| `empty_killers` | empty | Killers that used up their doses. |
| `consumable_used` | (not in the table) | Doses used by all killers together, in units of 1 per killer (`1 / treatment_doses` per dose). |
| `drift` | drift N h | The killers' mean distance in cells from their start, after each hour. |
| `wall` | wall (s) | Seconds the run took to compute. |

Every pest is either treated or still open at the end, so the share of the pests that were treated is

$$\text{TreatedShare} = \frac{T}{T + O}$$

| Symbol | Meaning |
| --- | --- |
| $T$ | `treated`, pests treated during the run |
| $O$ | `open`, pests still infested at the end |

To measure something else, add a field to `PestResponse` and fill it in `measure_pest_response` (`src/swarm_field/experiments/pest_response.py`). A run function can also be your own, as long as it takes a `Run` and returns a dataclass (section 3).

## 3. Run every arm with every seed, in parallel

A *run* (`Run`) is an arm name, a configuration and a seed. `runs_for` makes one run per arm and seed, and `Experiment.run` runs them all with a run function and returns their results in the same order:

```python
"""Compare the seeded and the hop-based pest response on the small river field."""

from swarm_field.experiments import (
    RIVER_SMALL,
    Experiment,
    PestRunSpec,
    measure_pest_response,
    pest_response_configurations,
    runs_for,
)


def main() -> None:
    configurations = pest_response_configurations(RIVER_SMALL)
    specs: dict[str, PestRunSpec] = {}
    for name in ("seeded", "hop-tuned"):
        specs[name] = PestRunSpec(configurations[name], hours=3, pests_per_hour=18.0)

    experiment = Experiment("seeded_vs_hop_small")
    runs = runs_for(specs, seeds=range(12), cost=PestRunSpec.cost)
    results = experiment.run(runs, measure_pest_response)


if __name__ == "__main__":
    main()
```

- **Arm names** are the keys of the dictionary you pass to `runs_for`, so every arm needs its own. When arms differ in a parameter, `arm_name` builds the name from it: `arm_name("river-small", "seeded", killers=8)` gives `river-small/seeded/killers=8`.
- **Seeds** are the same for every arm, so two arms are compared on the same relays and the same pests. Use at least 8; 12 is a good default.
- **`cost`** is optional. It is a rough relative cost of an arm, so the slowest runs start first and no process is left alone with a long run at the end. `PestRunSpec.cost` rates a seeded run five times a hop run, since its relays gossip whole pest lists.
- **`Experiment.run`** returns one `RunResult` per run, with `.run` (arm, seed), `.result` (the `PestResponse`) and `.wall` (seconds).

### How it runs in parallel

`Experiment.run` starts a pool of worker processes, by default one per CPU (`processes=None`), and hands every run to the next free process. A run is a function of its seed alone, so the results are the same on any computer and with any number of processes; only the time it takes changes. With fewer cores, pass `processes` or `--jobs` explicitly if you want to keep some free, and expect the time to grow with the number of runs divided by the cores. Each process holds one run in memory at a time, about a third of a gigabyte for a seeded run on the large field, so a machine with little memory may need fewer processes than cores.

Three rules follow from running in separate processes:

1. The run function must be a module-level function, such as `measure_pest_response`, not a lambda or a nested function.
2. Every configuration must pickle; the dataclasses of `swarm_field` do.
3. The script must start from `if __name__ == "__main__":`, so that worker processes that import it do not start an experiment of their own.

As a guide on 22 processes, measured with killers at 0.1 m/s: 12 seeds of `seeded` and `hop-tuned` on both ready fields, 3 h each, take about 6 minutes. A single 3-hour run takes about 30 s (`seeded`) or 11 s (`hop-tuned`) on the small field, and about 300 s or 100 s on the large one.

## 4. Averages, standard errors and the table

`by_arm` groups the results by arm, seeds in run order. The table helpers then give one row per arm:

```python
from swarm_field.experiments import (
    by_arm,
    markdown_table,
    paired_difference,
    pest_table_cells,
    pest_table_headers,
    treated_share,
)

grouped = by_arm(results)
baseline = [result.infested for result in grouped["seeded"]]
headers = pest_table_headers(hours=3)
headers.extend(["treated %", "Δ infested"])
rows: list[list[str]] = []
for arm, arm_results in grouped.items():
    infested = [result.infested for result in arm_results]
    cells = pest_table_cells(arm, arm_results, hours=3)
    cells.extend([treated_share(arm_results), paired_difference(infested, baseline)])
    rows.append(cells)
table = markdown_table(headers, rows)
print(table)
experiment.write_text("tables.md", table)
```

- `pest_table_headers(hours)` and `pest_table_cells(label, results, hours)` give the standard columns of section 2, as the mean over the seeds with its standard error where it matters.
- `mean_and_error(values)` formats any list of numbers as `mean ± error`; a value of `None`, e.g. the response time of a run that treated nothing, gives `-`.
- `paired_difference(values, baseline)` compares two arms seed by seed, which is much more precise than comparing their means, since both arms see the same pests.
- `markdown_table(headers, rows)` writes any list of rows as a Markdown table.

The mean of $n$ runs and its standard error, and the paired difference, are

$$\bar{x} = \frac{1}{n}\sum_{s=1}^{n} x_s, \qquad \text{SE} = \frac{\text{sd}(x)}{\sqrt{n}}, \qquad \bar{d} = \frac{1}{n}\sum_{s=1}^{n} \left(x_s - b_s\right) \pm \frac{\text{sd}(x - b)}{\sqrt{n}}$$

| Symbol | Meaning |
| --- | --- |
| $n$ | the number of seeds |
| $x_s$ | an arm's value with seed $s$, e.g. its infested cells |
| $b_s$ | the baseline arm's value with the same seed |
| $\text{sd}$ | the sample standard deviation over the seeds |

A difference of about two standard errors or more is unlikely to be chance.

## 5. Where the results are

Every `Experiment` makes a new folder in `output/`, at the root of the repository, which git ignores. Its name starts with the date and time, so a listing sorts by time:

```text
output/
  2026-10-22_22-19-02_seeded_vs_hop_small/
    experiment.json
    results.json
    tables.md
    video/
      hop-tuned_seed0.mp4
```

- **`experiment.json`** says what was run: the name, the start time, the command line, the git commit (with `+dirty` if there were uncommitted changes) and every arm with its seeds. The configurations themselves are not written yet, since they have no JSON form so far; the commit and the command reproduce them.
- **`results.json`** has one record per run: its `name` (`ARM/seedN`), `arm`, `seed`, `wall`, and `result`, the `PestResponse` as an object.
- **`video/`** holds the videos of section 6, one per filmed run, named after it.
- **Anything else** goes next to them with `experiment.write_text(name, text)`, `experiment.write_json(name, data)` or `experiment.path(name)`, which gives a path for a video or a figure.

To use the results again without running anything, read `results.json`:

```python
import json
from pathlib import Path
from statistics import mean

records = json.loads(Path("output/2026-10-22_22-19-02_seeded_vs_hop_small/results.json").read_text())
infested: dict[str, list[float]] = {}
for record in records:
    infested.setdefault(record["arm"], []).append(record["result"]["infested"])
for arm, values in infested.items():
    print(arm, round(mean(values), 2))
```

## 6. Watch a run: videos

The table says how well an arm did, not why. To see what its robots actually did, film a few of its runs. `record_videos` runs them again and renders each into a video in the experiment's folder. A run is a function of its arm and seed, so the video shows exactly the run that `results.json` describes.

Pass it the folder and the runs to film. `runs_for` builds those runs from the same specs as the batch, here the `hop-tuned` arm with seed 0:

```python
from swarm_field.experiments import record_videos

experiment = Experiment("seeded_vs_hop_small")
runs = runs_for(specs, seeds=range(12), cost=PestRunSpec.cost)
results = experiment.run(runs, measure_pest_response)
record_videos(experiment.directory, runs_for({"hop-tuned": specs["hop-tuned"]}, seeds=[0]))
```

The video goes to `video/hop-tuned_seed0.mp4` in the experiment's folder. Its name is the run's name, `ARM/seedN`, with every `/` turned into `_`; `video_path(directory, run)` gives it.

- **Several runs** are filmed in parallel, one per process, as in `Experiment.run`: `runs_for({"seeded": specs["seeded"], "hop-tuned": specs["hop-tuned"]}, seeds=[0, 1])` gives four videos. `processes` sets the number of processes.
- **An experiment run earlier** works too. Pass its folder, e.g. `Path("output/2026-10-22_22-19-02_seeded_vs_hop_small")`, and the same arm name and spec as in the batch, so the run is the same one. The arm name matters: it is the run's label.
- **Only pest runs** can be filmed: `record_videos` takes runs of a `PestRunSpec`, which it steps like `measure_pest_response`.

### What a video shows

One frame is one decision interval (15 s), and a video plays at 16 frames per second (`fps`), so a 3-hour run is 720 frames, a 45-second video. The frames are drawn with `MatplotlibRenderer`:

| Mark | Meaning |
| --- | --- |
| red cells | Pests, light red at a new one and dark at a full one. |
| big red dots | Killers, with a line along their last 20 cells and a dashed line to their target. |
| small grey dots | Relays. |
| dark grey cells | Obstacles. |
| side panel | The run's setup, the infested cells now, the pests treated so far and their mean response, the killers by goal, and the doses used. |

The panel's measures are those of `measure_pest_response` (`MeasuredPestRun`), so at the last frame they agree with the run's record in `results.json`.

### Requirements and time

- An `.mp4` needs [ffmpeg](https://ffmpeg.org/) on the `PATH` (`sudo apt install ffmpeg`).
- Drawing a frame takes about 0.3 s, on top of the simulation. One 3-hour video on the small field takes about 4 minutes, and the large field is slower. Film a few runs, not every seed.
- To try a video on its own, without a batch, `record_video(run, path)` films one run to any path: `.mp4`, `.gif`, or a directory of PNG frames. It returns the run's `PestResponse`.

## Example: the obstacle benchmark

[`tests/benchmark_obstacles.py`](../tests/benchmark_obstacles.py) is this page as a script: the four configurations on both ready fields, one table per field, the treated share and the paired difference to `seeded`.

```bash
uv run python -m tests.benchmark_obstacles --seeds 12
```

| Option | Meaning |
| --- | --- |
| `--scenario river-small` | Run only one field (`river-small`, `rivers-large` or `both`). |
| `--seeds` | Seeds per arm. |
| `--hours`, `--warmup-hours` | How long a run lasts, and how much of it is left out of `infested`. |
| `--killers` | Override the number of killers. |
| `--seeded-no-yield` | Add the seeded arm without the claims. |
| `--jobs` | Processes; one per CPU if not given. |

For a quick check that everything works, a run of a few seconds:

```bash
uv run python -m tests.benchmark_obstacles --scenario river-small --seeds 1 --hours 1 --warmup-hours 0
```
