# Running experiments

An experiment compares a few configurations, the *arms*, over the same seeds. Each run of an arm with a seed is simulated, measured, and its result kept. This page shows how to set up the arms from the default scenarios, measure them, run every arm with every seed in parallel, read the results, and watch a run as a video. What each experiment asks is in [`experiment-plan.md`](experiment-plan.md).

Everything here is imported from `swarm_field.experiments`. An experiment is a script in [`studies/`](../studies/README.md), run from the repository root as a module, e.g. `uv run python -m studies.<name>`.

## 1. Configure a run

A run is a `SwarmScenario` (`src/swarm_field/experiments/scenario.py`): a field (`MapConfig`), pests (`PestConfig`), settings every robot shares (`RobotConfig`), and a swarm added preset by preset. Start from a default scenario and change only what the question is about.

### The default scenarios

| Scenario | Field | Pests | Run length |
| --- | --- | --- | --- |
| `RIVER_SMALL` | `SMALL_RIVER_MAP`: 32 × 32 cells, one river with a bridge, rocks | `SMALL_FIELD_PESTS` | 2 h |
| `OPEN_SMALL` | `SMALL_OPEN_MAP`: 32 × 32 cells without obstacles | `SMALL_FIELD_PESTS` | 2 h |
| `RIVERS_LARGE` | `LARGE_RIVERS_MAP`: 64 × 64 cells, two rivers with three bridges, rocks | `LARGE_FIELD_PESTS` | 4 h |
| `OPEN_LARGE` | `LARGE_OPEN_MAP`: 64 × 64 cells without obstacles | `LARGE_FIELD_PESTS` | 4 h |

Every default scenario has the default swarm, 10 explorers (`EXPLORER`) and 4 seeded killers (`SEEDED_BEST`), with the robot settings `DEFAULT_ROBOTS`: explorers fly at 8 m/s, killers drive at 1 m/s, the radio reaches 100 m, every preset starts on a jittered lattice of its own, no robot knows the terrain in advance, and no battery runs empty. The pests are static (`STATIC_PESTS`): they neither grow nor spread, a killer treats one in a few seconds, and it never runs out of doses. `SMALL_FIELD_PESTS` puts 20 pests on the field at the start and lets 20 appear per hour, `LARGE_FIELD_PESTS` 8 and 8; these are a starting point, and an experiment sets the load its question needs. The runs are measured from the start, with the metrics sampled once a minute. The constants `DEFAULT_EXPLORERS`, `DEFAULT_KILLERS`, `SMALL_FIELD_HOURS` and `LARGE_FIELD_HOURS` hold the numbers.

### Change what the question is about

Every method returns a new scenario, so the calls chain and a default scenario is never changed:

```python
from dataclasses import replace

from swarm_field.contracts import BodyType
from swarm_field.experiments import (
    DEFAULT_KILLERS,
    DEFAULT_ROBOTS,
    EXPLORER,
    HOP_BEST,
    RIVER_SMALL,
    SMALL_FIELD_PESTS,
)

# The hop arm: the same scenario with HOP_BEST killers instead of SEEDED_BEST.
hop = RIVER_SMALL.without_robots(BodyType.KILLER).add_robots(HOP_BEST, DEFAULT_KILLERS)

# 20 explorers instead of 10.
explorers = RIVER_SMALL.without_robots(BodyType.SCOUT).add_robots(EXPLORER, 20)

# A 50 m radio.
radio = RIVER_SMALL.with_robot_config(replace(DEFAULT_ROBOTS, radio_range_m=50.0))

# Another pest load: 30 at the start and 30 per hour.
load = RIVER_SMALL.with_pests(replace(SMALL_FIELD_PESTS, pests_per_hour=30.0, initial_pests=30))

# A 3-hour run whose first hour is left out of the averages.
longer = RIVER_SMALL.with_hours(3, warmup_hours=1)
```

| Method | Returns the scenario with |
| --- | --- |
| `add_robots(preset, count)` | `count` more robots of `preset`, after the others. Robots get their ids in the order they are added. |
| `without_robots(body)` | its robots of `body` (`BodyType.SCOUT` or `BodyType.KILLER`) left out, or all robots without an argument. |
| `with_map(map_config)` | another field (`MapConfig`). |
| `with_pests(pests)` | other pests (`PestConfig`). |
| `with_pests_per_hour(rate)` | another pest rate. |
| `with_robot_config(robots)` | other robot settings (`RobotConfig`). |
| `with_hours(hours, warmup_hours)` | a run of `hours` whole hours, measured after `warmup_hours`, which must be shorter. It also samples the metrics once a minute, which the measurement needs. |
| `with_seed(seed)` | another seed. A batch sets each run's seed itself (section 3). |
| `with_failures(failures)` | robots that fail during the run (`FailureSpec`): how often and how. The failures are drawn from the run's fault-injection randomness, apart from the pests, so two arms with the same seed and the same robots lose the same robots at the same times. |

`count(name)` says how many robots of a preset a scenario has, and `killers` how many killers of every preset. The seed sets the pests, the starting positions and when each robot first decides, so two arms run with the same seed see the same pests.

### The robots: presets

A preset is a kind of robot: its name, its body, and its brain (`src/swarm_field/experiments/presets.py`).

| Preset | Body | What it does |
| --- | --- | --- |
| `EXPLORER` | scout | Explores the field and shares everything it knows. A scout flies, so it plans over the rivers and the rocks. |
| `SEEDED_BEST` | killer | The seeded method: plans a route over its map to a pest it knows of, and treats it. The swarm gossips the pests it knows and the killers' claims, so a killer leaves a pest to a closer killer that claimed it. |
| `HOP_BEST` | killer | The hop method: walks down the hop beacons to a pest, without a map. A pest's source beacons 0 and every robot broadcasts its distance in hops; a pest's source lasts 1.5 sweep times of the swarm; killers repel each other, and a killer stuck in place pushes itself out. It finds a dock down the DOCK beacons the docks broadcast. See [`research/hop-field.md`](research/hop-field.md). |

`SEEDED_BEST` and `HOP_BEST` are each method's best configuration, tuned one parameter at a time (`studies/tune_arms.py`). Compare the methods with these two, never with an untuned one. Their classes, `SeededKillerPreset` and `HopKillerPreset`, make variants with other parameters, as `ExplorerPreset` does for explorers; give a variant a `name` of its own to keep it apart in tables. `CustomPreset(name, body, brain_spec)` adds a robot with any `BrainSpec`, e.g. one with a new behaviour under test.

What a robot broadcasts follows from who listens. Every preset says what the swarm has to relay for it, and the scenario merges that for all its presets. In a swarm with hop killers every robot relays hop beacons, so an explorer that sees a pest is its source. In a swarm with seeded killers the robots pass on the pests they know, with the killers' claims.

### The settings

The fields of the three parts, for a question that changes one of them:

| `RobotConfig` field | Meaning |
| --- | --- |
| `speed_mps_drone`, `speed_mps_ground` | The explorers' speed and the killers' speed. |
| `radio_range_m` | How far every robot's broadcasts reach. |
| `decision_interval` | The time between two decisions of a robot that stands still, 15 s. A moving robot decides again as soon as its move arrives. |
| `knows_terrain` | Whether robots start with the map. By default they do not, and learn obstacles by sensing. |
| `lattice_starts` | Each preset's robots start on a jittered lattice of their own over the field. Otherwise all robots start along the bottom row. |
| `battery_capacity` | How many standard charges every battery holds. A standard charge lasts a body's endurance in travel, 30 minutes for a scout. `UNLIMITED_BATTERY_CAPACITY`, the default's, keeps robots working without trips to a dock. |

| `MapConfig` field | Meaning |
| --- | --- |
| `size` | The field's side in cells. |
| `obstacles` | The blocked cells, as a tuple of `(x, y)`. `river_and_rocks(size)` and `two_rivers_and_rocks(size)` build the river maps for any size; `()` is an open field. |
| `cell_size` | The side of a cell in metres, 10. |
| `docks` | The cells where robots recharge and killers restock, any number at once. One on each bank of the river on `SMALL_RIVER_MAP`, at `(0, 0)` and `(31, 31)`, and two opposite corners on the large maps, `(0, 0)` and `(63, 63)`; `SMALL_OPEN_MAP` has one at `(0, 0)`. Robots return to one when their battery runs low, killers also when their doses do ([`battery-lifecycle.md`](battery-lifecycle.md)). |
| `dock_range_m` | How far a dock's own DOCK beacon reaches, for hop killers; the robots' radio range by default. |

| `PestConfig` field | Meaning |
| --- | --- |
| `pests_per_hour` | How many pests appear per hour, on average, on the free cells. |
| `initial_pests` | How many pests are on the field when the run starts, each on a free cell of its own drawn from the run's seed. |
| `growth_rate` | How much a pest's severity grows per minute; 0 for static pests. |
| `spread_probability` | The chance that a pest spreads to a neighbour; 0 for static pests. |
| `spawn_severity` | The severity of a new pest, 0.1 by default. |
| `treatment_rate` | How much severity a treating killer removes per hour. `FAST_TREATMENT`, the default's, clears a full pest in a minute. It must beat the growth, or no pest ever clears. |
| `treatment_doses` | The doses a killer carries; every treatment uses one. `UNLIMITED_DOSES` is the default's. |

A larger field of the same kind is `MapConfig(size=48, obstacles=two_rivers_and_rocks(48))`.

The scenarios of the earlier studies stay for those studies: `RELAYED_RIVER_SMALL`, `RELAYED_RIVERS_LARGE` and `RELAYED_KILLERS` (standing relays and mock pest sources), `PESTS`, `EXPLORE`, `SMOKE` (the base demo: five explorers and five seeded killers on `PESTS`' field), and `moving_explorer_scenario`. They use the earlier parts `FAST_PESTS`, `NO_PESTS`, `LATTICE_ROBOTS` and `EXPLORE_ROBOTS`, with slower robots and a shorter radio.

## 2. Metrics

`measure_pest_response` runs one scenario with one seed and returns a `PestResponse`. It advances the run one decision interval at a time and samples the ground truth and the killers between steps. Coverage, idleness and redundancy come from the run's metrics, sampled once a minute, so the same runs measure how the explorers cover the field. A run without killers or without pests is measured too, e.g. explorers alone.

The main outcomes:

| `PestResponse` field | Table column | Meaning |
| --- | --- | --- |
| `infested` | infested | Infested cells, averaged over the run after the warm-up. The main outcome: it counts how many pests stay and for how long. |
| `treated` | treated | Pests cleared during the run. |
| `open` | open | Pests still infested at the end. |
| `response_minutes` | response | Minutes from a pest's appearance to its clearing, over the cleared pests. |
| `response_p90` | p90 | The 90th percentile of those minutes. |
| `cells_per_treatment` | cells/treat | Cells moved by all killers per cleared pest. |
| `messages_per_robot_hour` | msgs | Messages received per robot per hour. A hop robot sends its beacons in a message of their own next to its status, so its broadcast counts as two. |
| `items_per_robot_hour` | items | The items those messages carried, per robot per hour: a status, every fact of a knowledge snapshot and every beacon is one. It compares what the methods communicate by size. |
| `coverage` | coverage | The share of the crop cells some robot sensed by the end of the run. |
| `idleness_minutes` | idle | Minutes since a robot last sensed a crop cell, averaged over the cells and the samples after the warm-up: the expected wait of a new pest for its first sighting. |
| `redundancy` | redund | The share of the robots' visits to a cell sensed less than 30 minutes before. |
| `explored_minutes` | (not in the table) | Minutes until every crop cell had been sensed at least once, or `None` if some never was; to the minute. |
| `clean_minutes` | (not in the table) | Minutes until the field was first free of pests after a pest was treated, or `None`; e.g. the time to clear a field of `initial_pests` when no pest appears or spreads. |
| `wall` | wall (s) | Seconds the run took to compute. |

Every pest is either treated or still open at the end, so the share of the pests that were treated is

$$\text{TreatedShare} = \frac{T}{T + O}$$

| Symbol | Meaning |
| --- | --- |
| $T$ | `treated`, pests cleared during the run |
| $O$ | `open`, pests still infested at the end |

The other fields diagnose why an arm did as it did:

| `PestResponse` field | Meaning |
| --- | --- |
| `status_items_per_robot_hour`, `fact_items_per_robot_hour`, `beacon_items_per_robot_hour` | The items by kind: statuses, facts of knowledge snapshots, and beacons. |
| `stuck` | The share of the time a killer committed to treating stands idle. |
| `commit_hops` | A hop killer's hop value each time it commits to treating. |
| `no_doses` | The share of the killers' time without doses. |
| `shared_target` | Of the samples with two or more killers committed to treating, the share in which two report the same target: a duplicate trip. |
| `co_treating` | Of the samples with a treatment under way, the share in which two killers treat the same cell. |
| `isolated` | The share of the killers' time with no other robot in radio range. |
| `doses_used` | Doses used by all killers together. |
| `drift` | The killers' mean distance in cells from their start, after each hour. |
| `worst_idleness_minutes` | The longest idleness, over the same samples as `idleness_minutes`. |

The run's metrics have more standard measures in their last sample (`MetricsSample`, from `measured.live.metrics.latest()` of a `MeasuredPestRun`): the pests that appeared and were cleared, the knowledge latency from a pest's detection to the first killer knowing of it, the response time from detection to clearing, the shares of the killers' decisions spent idle, exploring, driving to a pest and treating, and the share of duplicate trips. `studies/smoke.py` reads them.

To measure something else, write a run function of your own: it takes a `Run` and returns a dataclass. `MeasuredPestRun` steps a run as `measure_pest_response` does and gives access to its simulation and its `PestTracker` between steps; `studies/hop_dock_return.py` measures the killers' trips to a dock this way.

## 3. Run every arm with every seed, in parallel

A *run* (`Run`) is an arm name, a scenario and a seed. `runs_for` makes one run per arm and seed, and `Experiment.run` runs them all with a run function and returns their results in the same order:

```python
"""Compare the seeded and the hop-based pest response on the small river field."""

from swarm_field.contracts import BodyType
from swarm_field.experiments import (
    DEFAULT_KILLERS,
    HOP_BEST,
    RIVER_SMALL,
    Experiment,
    SwarmScenario,
    measure_pest_response,
    runs_for,
)


def main() -> None:
    hop = RIVER_SMALL.without_robots(BodyType.KILLER).add_robots(HOP_BEST, DEFAULT_KILLERS)
    arms: dict[str, SwarmScenario] = {"seeded": RIVER_SMALL, "hop": hop}

    experiment = Experiment("seeded_vs_hop_small")
    runs = runs_for(arms, seeds=range(8), cost=SwarmScenario.run_cost)
    results = experiment.run(runs, measure_pest_response)


if __name__ == "__main__":
    main()
```

- **Arm names** are the keys of the dictionary passed to `runs_for`, so every arm needs its own. When arms differ in a parameter, `arm_name` builds the name from it: `arm_name("river-small", "seeded", explorers=20)` gives `river-small/seeded/explorers=20`.
- **Seeds** are the same for every arm, so two arms are compared on the same pests. Use 3 to 8.
- **`cost`** is optional: a rough relative cost of an arm, so the slowest runs start first and no process is left alone with a long run at the end.
- **`Experiment.run`** returns one `RunResult` per run, with `.run` (arm, seed), `.result` (the `PestResponse`) and `.wall` (seconds).

Collect every configuration of an experiment into one `Experiment`, so its runs share the processors and its results end up in one folder.

### How it runs in parallel

`Experiment.run` starts a pool of worker processes, by default one per CPU, and hands every run to the next free process. A run is a function of its seed alone, so the results are the same on any computer and with any number of processes; only the time changes. Pass `processes`, or `--jobs` in a study, to keep some cores free. Three rules follow from running in separate processes:

1. The run function must be a module-level function, such as `measure_pest_response`, not a lambda or a nested function.
2. Every configuration must pickle; the dataclasses of `swarm_field` do.
3. The script must start from `if __name__ == "__main__":`, so that worker processes that import it do not start an experiment of their own.

As a guide, one run of a default scenario at its default load takes about 45 s on a small field and about 170 s on a large one. A run with more pests takes longer, up to a few times as long when the killers fall behind, and the batch logs nothing until it ends.

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
headers = pest_table_headers(hours=2)
headers.extend(["treated", "Δ infested"])
rows: list[list[str]] = []
for arm, arm_results in grouped.items():
    infested = [result.infested for result in arm_results]
    cells = pest_table_cells(arm, arm_results, hours=2)
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

A difference smaller than one or two standard errors supports no conclusion.

## 5. Where the results are

Every `Experiment` makes a new folder in `output/`, at the root of the repository, which git ignores. Its name starts with the date and time, so a listing sorts by time, and ends in four random digits, so two experiments started in the same second get folders of their own:

```text
output/
  2026-10-22_22-19-02_seeded_vs_hop_small_4821/
    experiment.json
    results.json
    tables.md
    video/
      hop_seed0.mp4
```

- **`experiment.json`** says what was run: the name, the start time, the command line, the git commit (with `+dirty` if there were uncommitted changes) and every arm with its seeds. The configurations themselves are not written; the commit and the command reproduce them.
- **`results.json`** has one record per run: its `name` (`ARM/seedN`), `arm`, `seed`, `wall`, and `result`, the `PestResponse` as an object.
- **Anything else**, such as tables and figures, goes next to them with `experiment.write_text(name, text)`, `experiment.write_json(name, data)` or `experiment.path(name)`, which gives a path to write to.

To use the results again without running anything, e.g. to plot them, read `results.json`:

```python
import json
from pathlib import Path
from statistics import mean

records = json.loads(Path("output/2026-10-22_22-19-02_seeded_vs_hop_small_4821/results.json").read_text())
infested: dict[str, list[float]] = {}
for record in records:
    infested.setdefault(record["arm"], []).append(record["result"]["infested"])
for arm, values in infested.items():
    print(arm, round(mean(values), 2))
```

## 6. Watch a run: videos

The table says how well an arm did, not why. `record_videos` runs some runs of a batch again and renders each into a video in the experiment's folder. A run is a function of its arm and seed, so the video shows exactly the run that `results.json` describes:

```python
from swarm_field.experiments import record_videos

record_videos(experiment.directory, runs_for({"hop": arms["hop"]}, seeds=[0]))
```

The video goes to `video/hop_seed0.mp4`: the run's name, `ARM/seedN`, with every `/` turned into `_`. Several runs are filmed in parallel, one per process. A folder of an earlier experiment works too, with the same arm name and scenario as in its batch.

A video draws three frames per decision interval (`frames_per_step`) and plays at 16 frames per second (`fps`), so one second of video is 80 s of the run. It shows pests as red cells, light at a new one and dark at a full one, killers as big red dots with a dashed line to their target, explorers as blue dots, obstacles as dark grey cells, and a side panel with the time, the infested cells, the pests treated so far and the killers by goal. The panel's measures are those of `measure_pest_response`, so at the last frame they agree with `results.json`.

An `.mp4` needs [ffmpeg](https://ffmpeg.org/) on the `PATH`. Drawing a frame takes about 0.3 s on top of the simulation, so film a few runs, not every seed. `record_video(run, path)` films one run on its own, to an `.mp4`, a `.gif` or a directory of PNG frames.

## 7. An experiment as a JSON file

A simple sweep needs no script: a JSON file in [`configs/`](../configs/) names a scenario, the seeds and the conditions, and `uv run python -m swarm_field.experiments.from_json FILE` runs it with `measure_pest_response`. The default scenarios are named `river-small`, `rivers-large`, `open-small` and `open-large`. The file format is in the [README](../README.md).
