# Experiment plan

The experiments behind the report: the settings they share, how to run, plot and analyse them, and what each one asks. How to write an experiment as code is in [`experiments.md`](experiments.md). Every experiment is a script in [`studies/`](../studies/README.md).

## Default settings

An experiment uses one of the default scenarios below unless its question needs something else. Everything named here is imported from `swarm_field.experiments`.

### Fields

| Default scenario | Field | Map | Run length |
| --- | --- | --- | --- |
| `RIVER_SMALL` | 32 × 32 cells, one river with one bridge, and rocks | `SMALL_RIVER_MAP` | 2 h (`SMALL_FIELD_HOURS`) |
| `RIVERS_LARGE` | 64 × 64 cells, two rivers with three bridges, and rocks | `LARGE_RIVERS_MAP` | 4 h (`LARGE_FIELD_HOURS`) |
| `OPEN_SMALL` | 32 × 32 cells without obstacles | `SMALL_OPEN_MAP` | 2 h (`SMALL_FIELD_HOURS`) |
| `OPEN_LARGE` | 64 × 64 cells without obstacles | `LARGE_OPEN_MAP` | 4 h (`LARGE_FIELD_HOURS`) |

A cell is 10 m wide. Use a custom map only when the question needs one.

### Robots

- 10 explorers and 4 killers (`DEFAULT_EXPLORERS`, `DEFAULT_KILLERS`).
- An explorer (`EXPLORER`) is a drone: fast, with a smaller battery, and it flies over obstacles. A killer is a ground robot: slow, with a bigger battery.
- Explorers fly at 8 m/s, killers drive at 1 m/s.
- The radio reaches 10 cells, 100 m.
- A robot senses pests in its own cell, and obstacles in its direct neighbours.
- A robot always detects and treats perfectly.
- No battery drain, and unlimited killer doses.
- No rebalancing.

`DEFAULT_ROBOTS` holds the speeds, the radio and the unlimited battery. The default scenarios have `SEEDED_BEST` killers; the hop arm swaps them for `HOP_BEST`.

### Pests

- X pests on the field at the start, and X new pests every hour. Tune X per field size and swarm so that the killers are neither idle nor overworked. `SMALL_FIELD_PESTS` (20) and `LARGE_FIELD_PESTS` (8) are a starting point.
- Static: no growth and no spreading (`STATIC_PESTS`).

## Running guidelines

- Orchestrate the runs with `swarm_field.experiments`, as [`experiments.md`](experiments.md) describes, and put the experiment's code in `studies/`.
- Track the swarm's performance mostly with the standard metrics. Explain any custom metric.
- Use the four standard fields: small and large, with rivers and open. Use a custom map only when it is required.
- Collect all the run configurations at once, and run the whole experiment with all of them in one `Experiment`. Its runs then go in parallel over the processor, and all their results are written together to one folder, `output/<date>_<time>_<name>_<digits>/`.
- If it does not take too long, run each configuration with several seeds (3, 5 or 8) and compute standard errors. Prefer fewer configurations with a few seeds to many configurations. When two configurations differ by less than one or two standard errors, draw no conclusion.
- When comparing seeded with hop-based killers, always use the best parameters for both (`SEEDED_BEST`, `HOP_BEST`). A comparison with a sub-optimal configuration is meaningless.

## Plotting guidelines

- Use the same colours for the two methods throughout: blue or other cool colours for hop-based, orange or other warm colours for seeded.
- Give the configuration in the caption: radio range, number of robots, map type (rivers and rocks, or open) and map size.
- Export figures as PDF.
- If possible, generate the LaTeX fragments in Python, so that the figures can be rebuilt when the experiments run again.

## Analysis guidelines

- Before running an experiment, state a hypothesis of what will happen. It helps find bugs and verify the results.
- Plot the relations the hypothesis is about, as numerical verification, with standard errors over the seeds of a configuration where possible.
- Write a summary of the results: analyse them and explain what was unexpected, supported by the plots. The text goes into the report.

## Experiments

### BASELINE: baseline study

Assignee: Błażej. Fields: `RIVER_SMALL`, `RIVERS_LARGE`.

Which pest response performs better in a general scenario? Measure the standard metrics: latency, the infested cells at the end, the pests cleared per hour, and the cells driven per pest cleared.

| Configuration | Robots |
| --- | --- |
| 1 | 5 explorers and 5 seeded killers |
| 2 | 5 explorers and 5 hop-based killers |
| 3 | 5 baseline killers: a simple implementation of robots that do not communicate |

Status: `studies/explorer_hop_killers.py` runs explorers with seeded and with hop-based killers. It needs plots of the results. The baseline arm (3) is not implemented yet.

### EXPLORE: exploration study

Assignee: Tycho. Fields: `RIVER_SMALL`, `RIVERS_LARGE`.

How long does a swarm of explorers take to explore the whole field? Compare a grid search with the seeder, for 1, 5, 10 and 20 explorers.

Status: most parts exist. It needs a metric for the time at which the exploration is complete.

### RADIO: radio range

Assignee: Tycho. Fields: with rivers and rocks, and open.

How does the radio range affect the performance of seeded and hop-based killers? Measure the standard metrics against the radio range, and plot them for both killer types.

- Radio ranges of 50 m, 100 m and 300 m. These are flexible and can be tuned to the field size.
- One number of explorers and killers per field: the number of robots does not vary on a field.
- Hop-based against seeded killers.

Status: `configs/radio_range.json` sweeps the radio range, but only with seeded killers.

### NUM_EXPLORERS: number of explorers

Assignee: Obi. Fields: `RIVER_SMALL`, `RIVERS_LARGE`.

How many explorers does a number of killers need, on a field of a given size, at a given infestation rate? Measure the standard metrics (latency, pests cleared and others) against the number of explorers.

- 5 hop-based killers, or 5 seeded killers; more on the large field.
- A sweep of the number of explorers, e.g. 10, 20, 30.
- One infestation rate and growth rate, or two sets of them: low and high infestation.
- A radio range small enough to tell a sparse swarm with few explorers from a dense one with many.

That makes (2 killer types) × (fields) × (explorer counts) × (infestation modes) configurations, each with its seeds.

Status: not implemented, and not specified in detail.

### OUTAGE: robustness

Assignee: Tycho.

Remove robots from the swarm during a run, and see how its effectiveness drops with the number of robots. Measure the standard metrics against the robots dropping out over time, and plot performance against time or the robots left, in series for hop-based and seeded killers.

- Explorers with hop-based killers, and explorers with seeded killers.
- A failure rate at which 30 % to 50 % of the swarm is gone by the end of the run.

To decide:

- Which robots are removed: explorers, or all types.
- Whether outages are random at an interval, as the code does now (`FailureSpec`), or whether a specific explorer or killer is disabled at specific times in both arms. The second removes some randomness from the results.

A possible extension simulates several failure types: for example only the motors fail and the robot still communicates, or only its treatment fails. With several failure modes it matters even more that the compared configurations see the same sequence of failures.

Status: not implemented, and not specified in detail.

### SCALE: scalability

Assignee: Błażej.

Set a required latency for clearing pests, and find the smallest swarm that meets it. Find how that minimal swarm grows with the field.

- One required latency and one pest rate.
- Go from small fields to large ones, and for each find the smallest swarm that meets the latency. Alternatively go over swarm sizes, and for each find the largest field it can handle.
- Both hop-based and seeded killers.
- If the requirement turns out too rigid and the results are odd, relax it, e.g. to "90 % of the pests are cleared within the latency".

Plot and explain the field size against the required swarm size, and put it into one number with the adaptability or scalability metric. The question: when the field doubles in cells (its side grows by √2), does the swarm need to double too, or grow less, or more?

### SPEED: speed of the swarm

Assignee: Tycho. Field: `RIVERS_LARGE`.

How long does a swarm take to clear the whole field, when 1 % of the plants are infested at the start and pests neither appear nor spread? Measure the time until the field is clear.

- No new pests and no spreading; 1 % of the plants infested at the start (`PestConfig.initial_pests`).
- Explorer and killer speeds: to ask the other group.
- Compare hop-based, grid search and seeded.
- With 1, 5, 10, 20 and 40 robots.

### RESUPPLY: charging and resupply

Assignee: Błażej.

How do a finite battery and finite killer doses affect the swarm? Measure the standard metrics with charging and resupply. Also judge how many docks are needed: whether one is enough or one per bank of the river is needed. And how fast charging has to be to support the swarm: charging (30 minutes to 90 %) or a battery swap (5 minutes to 100 %).

- Seeded against hop-based killers.
- Different numbers and places of docks (`MapConfig.docks`).
- Battery swap against charging.
- Realistic battery capacities for every robot body.
- Compared with the baseline of an unlimited battery and unlimited doses.

Plot the runs with a finite battery and doses against the unlimited baseline. Give a qualitative review of how many docks are needed and where. In earlier runs, hop-based killers on a field with a river did badly with a dock on one bank only, so this can be presented as a flaw of the hop-based method.

### REBALANCE

Assignee: Obi. Not specified yet.
