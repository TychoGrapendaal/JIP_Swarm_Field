# EXPLORE: drawing robots to stale information (#41)

This document describes the EXPLORE field seeder, how it is measured, and what the measurements say. The seeder is `ExploreSeeder` in `swarm/explore.py`; the measurements come from `MetricsCollector` (`simulation/metrics.py`), the experiment `EXPLORE`, a `SwarmScenario` of explorers alone, in `swarm_field.experiments`, run by `swarm_field.runner`, and `studies/explore_variants.py`. How a seeder fits into the brain is described in [`robot-memory-and-navigation.md`](./robot-memory-and-navigation.md) under "Writing a behaviour".

Exploring a new field and monitoring a known one are one objective. A cell is worth visiting again by how long ago anyone observed it, and a never-seen cell is simply the oldest information there is. A robot observes only the cell it stands on, so visiting a cell is observing it.

## Value of a cell

A cell is worth nothing for a while after it was observed, and from then on it is *stale* and worth more the older its information is:

$$v(c) = \begin{cases} 0 & \text{if } a(c) < A \\ 1 - 2^{-a(c) / A} & \text{otherwise} \end{cases} \qquad a(c) = t_\text{now} - t_\text{obs}(c)$$

| Symbol | Meaning |
| --- | --- |
| $v(c)$ | What observing cell $c$ again is worth. |
| $a(c)$ | The age of the cell's information. |
| $t_\text{now}$ | The current time. |
| $t_\text{obs}(c)$ | When the cell was last observed, by the robot or a peer; $-\infty$ if never. |
| $A$ | The revisit age (`ExploreConfig.revisit_age`): how long a cell is worth nothing after an observation. The one parameter of the value rule. |

A stale cell is worth 0.5 at the revisit age, 0.75 at twice that, and approaches 1; a cell never observed is worth 1. This is $1 - e^{-a/\tau}$ with $\tau = A / \ln 2$, the rule of the first version, which had a separate $\tau$ and a stale threshold $\theta$ = 0.5; $A = \tau \ln \frac{1}{1 - \theta}$ maps one onto the other. Cells memory knows as obstacles are worth 0. Memory keeps $t_\text{obs}$ for every cell in one array (`AgentMemory.last_observed_times`), so the values of all cells are one vectorised expression.

## Seeds: the robot's block cell by cell, other blocks one seed each

The field is split into aligned square blocks. The robot's own block gets one seed per stale cell. Every other block gets one seed, on its stalest cell, worth that cell's value, so a single forgotten cell is never hidden among fresh ones. A seed starts at $-\beta$ times its worth.

$$w_b = \max_{c \in b} v(c)$$

$$\phi(x) = \min\Big( \min_{c \in \text{own block},\ v(c) > 0} \big( -\beta \, v'(c) + d(x, c) \big),\ \min_{b \neq \text{own}} \big( -\beta \, w'_b + d(x, r_b) \big) \Big)$$

| Symbol | Meaning |
| --- | --- |
| $\phi(x)$ | The potential at cell $x$; the robot steps downhill. |
| $w_b$ | The worth of block $b$: the value of its stalest cell. |
| $v'$, $w'_b$ | The values after the peer discount below; the plain values when it is off. |
| $r_b$ | The stalest cell of block $b$, which only serves to route the robot to the block. |
| $d(x, c)$ | The route length from $x$ to $c$ over the quadtree, in cells. |
| $\beta$ | How many cells of travel a fully stale cell or block is worth (`ExploreConfig.beta`). |

One field thereby does two jobs. Inside its block the robot heads for the nearest stale cell. Its own stale cells are close, so it usually finishes its block before leaving it; only a much staler cell elsewhere pulls it away first. On an open block this produces a back-and-forth sweep without being told to. A single robot on a 16 × 16 field sweeps each block column by column and covers the field in 42.5 minutes with 16 × 16 blocks and 45.5 minutes with 8 × 8 blocks, where a straight move takes 10 seconds, so one new cell per move would take 42.7 minutes. Even counting its first revisits, only 3 to 5 % of its moves land on a cell it has already visited. Once the block has no stale cell, only the other blocks are left, and the field leads to the best of them; on entering that block, its cells become the per-cell seeds.

**Block size.** `ExploreConfig.block_side` is the side of a block, 8 cells by default. Blocks that grew with the distance from the robot (8 × 8 near it, 16 × 16 farther away) were also tried; they did no better than 8 × 8 blocks everywhere (rows "8/16" in the results) and were removed.

## When the field is rebuilt

A field leads to one target, and a robot that observes only its own cell uses a target up by reaching it. So the field has to be rebuilt once per swept cell, and the key (`ExploreSeeder.field_key`) is made to change exactly then: it holds which cells of the robot's block are stale, the robot's block, a rebuild period (`ExploreConfig.rebuild_period`, which bounds how long the other blocks' worths may drift), and the peers' blocks when the discount is on. Travelling over fresh ground and peers sweeping other blocks cause no rebuild.

Rebuilding once per swept cell makes the window of the field matter. `FieldSeeder.window_blocks` says how many 16 × 16 blocks around the robot's block are searched cell by cell; beyond them the quadtree's leaves are used. EXPLORE uses 1. A window of 0 builds about three times faster. With 16 × 16 blocks, though, every neighbouring block is then one leaf, measured from its centre, so it looks up to about 17 cells farther than it is, and robots that should move on to it stay; in a first, single-seed comparison redundancy went from 0.24 to 0.39. With 8 × 8 blocks, a window of 0 is nearly as good as 1 (see the results).

## Spreading the robots: the peer discount

$$v'(c) = \frac{v(c)}{1 + n_{b(c)}}, \qquad w'_b = \frac{w_b}{1 + n_b}$$

| Symbol | Meaning |
| --- | --- |
| $n_b$ | The number of peers whose reported cell or reported target lies in block $b$. |
| $b(c)$ | The block containing cell $c$. |

Every robot broadcasts its cell and its target at every decision point (`MessageKind.STATUS`), and memory keeps the newest report per peer; a newer sighting of the peer by another robot updates its cell but keeps its reported target. With `ExploreConfig.spread_peers`, a block's worths are divided by one plus the number of peers in it or heading to it (`PeerDiscount`), counting only reports younger than `peer_max_age`. The discount applies to the robot's own block too, so a robot sharing its block with a peer is drawn to a block nobody covers. There is no kernel and no further parameter. Robots are thus made to spread explicitly, not only through shared observations making each other's ground look fresh.

## Parameters

The defaults are the ones that did best in the benchmark below.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `revisit_age` | 28 min | How long a cell is worth nothing after an observation. |
| `beta` | 64 | Cells of travel a fully stale cell or block is worth. |
| `rebuild_period` | 5 min | Longest drift of the other blocks' worths between rebuilds. |
| `block_side` | 8 | The side of a block in cells. |
| `spread_peers` | on | The peer discount. |
| `peer_max_age` | 1 min | Older peer reports are left out of the discount. |
| `window_blocks` | 1 | Blocks around the robot's searched cell by cell. |

## Measuring exploration

`MetricsCollector` measures a run from ground truth, independently of what the robots believe. A robot senses the cell it stands on at every decision, and every cell it senses is recorded, so no visit is missed; the ages are sampled once a minute. Over the cropped cells, which in the experiments are all cells that are not obstacles:

| Measure | Definition | What it shows |
| --- | --- | --- |
| Coverage, `t90`, `full` | The share of cells sensed at least once, and when 90 % and all of them were. | How fast the swarm explores a new field. |
| Mean idleness | The age $t - t_\text{last}(c)$ of a cell, averaged over cells and over the samples from `measure_from` on. A cell never sensed is as old as the run. | How fresh the swarm keeps its information. |
| Worst idleness | The largest age in those samples. | The worst-case detection latency. |
| Redundancy | The share of visits to a cell sensed less than 30 minutes before. | Wasted travel and robots getting in each other's way. |

The mean idleness is also the expected detection latency. A pest appearing at a uniformly random time in a cell with visit gaps $g_i$ waits $\sum_i g_i^2 / (2 \sum_i g_i)$ on average, and the time-averaged age of that cell is the same expression. So the benchmark measures how fast new pests are found without injecting any.

## The experiment and the benchmark

`SwarmScenario.into_experiment` describes one run of `EXPLORE` as an `ExperimentConfig`, which `swarm_field.runner` assembles and runs: a field with optional obstacles, explorers starting side by side in a corner, each a real `ComposedAgent` on the real simulation loop, and a `MetricsCollector`. Every explorer has one behaviour, the explore behaviour on its seeder's field, so the arbiter has nothing to choose. Robots start with the terrain map, so they plan around obstacles they have not seen yet. They share their memory and status over the channel, with an unlimited radio range unless `radio_range_m` says otherwise. The scenario's `seed` staggers the robots' first decisions, so repeated runs interleave them differently. `RandomWaypointSeeder` in `tests/baselines.py` is the baseline: one random target cell, and a new one once the robot stands on it.

`studies/explore_variants.py` runs every variant with several seeds and prints the mean of each measure; `tests/test_explore_swarm.py` checks the main findings on a small field.

## Results

Five robots start side by side in a corner of a 32 × 32 field of 10 m cells and run for three simulated hours; each row is the mean over four seeds. A straight move takes 10 s, so visiting one new cell per move, the swarm would need about 34 minutes to cover the field. `t90` and `full` are the minutes until 90 % and all of the field had been visited (`-` when a seed never got there). `idle` and `worst` are the mean and worst idleness in minutes over the second half of the run, and `redund` the share of redundant visits. Unless a row says otherwise, the parameters are the defaults: 8 × 8 blocks, the discount on, a revisit age of 28 min and $\beta$ = 64. "Obstacles" is the river with one bridge and two rocks of `river_and_rocks`.

| variant | t90 | full | idle | worst | redund | wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| random waypoint | - | - | 50.4 | 180.0 | 0.550 | 13 |
| explore 16 | 100.5 | - | 77.0 | 180.0 | 0.734 | 145 |
| explore 16 + discount | 39.6 | 46.6 | 22.6 | 62.9 | 0.235 | 190 |
| explore 8 | 125.7 | - | 79.9 | 180.0 | 0.768 | 101 |
| explore 8 + discount | 34.8 | 44.9 | 20.4 | 57.5 | 0.146 | 118 |
| explore 8/16 + discount (removed) | 36.5 | 47.8 | 20.4 | 57.5 | 0.140 | 124 |
| explore 8 + discount, window 0 | 35.3 | 43.3 | 20.8 | 63.1 | 0.160 | 42 |
| explore 8, no sharing | 140.4 | 159.0 | 76.2 | 179.3 | 0.773 | 93 |
| explore 8 + discount, no sharing | 84.6 | - | 38.9 | 180.0 | 0.594 | 101 |
| random waypoint, obstacles | - | - | 58.0 | 180.0 | 0.647 | 12 |
| explore 8, obstacles | - | - | 81.6 | 180.0 | 0.791 | 78 |
| explore 8 + discount, obstacles | 35.5 | 45.3 | 26.6 | 112.4 | 0.287 | 80 |

**The peer discount is what makes EXPLORE work.** Without it, the robots move as a herd: they all head for the same next block and sweep it together, so three quarters of their visits are redundant and they keep the field less fresh than random waypoints (mean idleness 77 or 80 minutes against 50). With it, mean idleness drops to about 20 minutes and the worst to about 60 minutes, against 180 for the baseline, and the field is covered in about 45 minutes.

**Block size matters little once the discount is on.** 8 × 8 blocks do a little better than 16 × 16, and blocks that grow from 8 to 16 cells with the distance did as well as 8 × 8, so they were removed. **Sharing observations matters:** without it, every robot only knows the cells it visited itself, and even with the discount mean idleness doubles. **A window of 0 is nearly as good and about 3 times faster** with 8 × 8 blocks, since it still searches the robot's whole 16 × 16 quadtree block cell by cell, which holds some of the neighbouring 8 × 8 blocks.

### Parameters

| variant | t90 | full | idle | worst | redund | wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| defaults (revisit age 28 min, beta 64) | 34.8 | 44.9 | 20.4 | 57.5 | 0.146 | 112 |
| revisit age 14 min | 35.0 | 45.0 | 20.2 | 73.0 | 0.168 | 121 |
| revisit age 42 min | 34.8 | 43.0 | 22.2 | 57.6 | 0.191 | 133 |
| revisit age 84 min | 34.8 | 43.0 | 43.0 | 119.5 | 0.185 | 78 |
| beta 16 | 36.8 | 48.9 | 24.2 | 125.8 | 0.255 | 129 |
| beta 32 | 35.9 | 44.9 | 20.1 | 73.0 | 0.165 | 104 |
| beta 128 | 34.8 | 45.1 | 21.2 | 52.7 | 0.165 | 78 |

A revisit age of 14 to 42 minutes works; 84 minutes is too long, since cells then wait more than twice the sweep time to become targets again, and 14 minutes raises the worst idleness. $\beta$ needs to be 64 cells or more, for a stale cell across the field to beat a barely stale one next door.

### A larger field

The same variants on a 64 × 64 field for six hours, mean over two seeds. Visiting one new cell per move, five robots would need about 136 minutes to cover it.

| variant | t90 | full | idle | worst | redund | wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| random waypoint | - | - | 147.2 | 360.0 | 0.216 | 44 |
| explore 16 | - | - | 207.1 | 360.0 | 0.741 | 364 |
| explore 16 + discount | - | - | 128.0 | 360.0 | 0.029 | 444 |
| explore 8 | - | - | 216.2 | 360.0 | 0.782 | 246 |
| explore 8 + discount | 146.8 | 183.4 | 88.4 | 332.7 | 0.088 | 331 |
| explore 8/16 + discount (removed) | 160.1 | - | 93.8 | 360.0 | 0.070 | 354 |
| explore 8 + discount, window 0 | 145.9 | 182.4 | 90.2 | 307.2 | 0.099 | 106 |
| explore 8, no sharing | - | - | 213.6 | 360.0 | 0.773 | 247 |
| explore 8 + discount, no sharing | - | - | 173.0 | 360.0 | 0.506 | 330 |
| random waypoint, obstacles | - | - | 162.2 | 360.0 | 0.356 | 33 |
| explore 8, obstacles | - | - | 228.4 | 360.0 | 0.793 | 318 |
| explore 8 + discount, obstacles | 152.1 | - | 115.8 | 360.0 | 0.085 | 269 |

The discount again cuts the mean idleness to well under half of no discount and to 60 % of random waypoints, with little redundancy, and with 8 × 8 blocks the whole field is covered at about 183 minutes. The worst idleness stays high, around 300 to 360 minutes: on a field that takes more than two hours to sweep, a few cells are still left for a long time, and with obstacles or 16 × 16 blocks some are never reached in six hours. The revisit age should grow with the time the swarm needs to sweep its field: in the comparison below, 56 minutes did a little better than 28 on this field (full coverage at 168 minutes, mean idleness 85 minutes), and 14 minutes never covered it.

### Earlier value rules

The first version valued a block by the mean value of its cells. On large fields it left holes: a single forgotten cell adds only $1/64$ to the mean of an 8 × 8 block, so its block looks almost done. Two fixes were compared, alone and together, with the same seeds as above: a block worth $\max(\text{mean}, \lambda \max)$, and a value that grows without bound, linearly as $a / 2A$ (0.5 at $A$, like the current rule), instead of saturating at 1.

| Value rule | 32 × 32: full | idle | worst | 64 × 64: full | idle | worst |
| --- | --- | --- | --- | --- | --- | --- |
| saturating, block mean (first version) | 46.3 | 21.4 | 71.2 | - | 91.1 | 360.0 |
| saturating, $\max(\text{mean}, 0.5 \max)$ | 48.1 | 22.1 | 78.1 | - | 91.1 | 360.0 |
| saturating, $\max(\text{mean}, 0.75 \max)$ | 45.1 | 24.4 | 87.2 | - | 106.5 | 341.2 |
| **saturating, block max (current)** | 44.9 | 20.4 | 57.5 | 183.4 | 88.4 | 332.7 |
| linear, block mean | 45.4 | 22.1 | 71.8 | 202.1 | 86.2 | 325.1 |
| linear, block max | 45.1 | 21.7 | 53.1 | 168.0 | 83.6 | 178.4 |

A block worth $\max(\text{mean}, \lambda \max)$ only helps at $\lambda$ = 1: with 0.25 or 0.5 nothing changes on the large field, since a block whose only stale cell is worth at most 0.5 still loses to the old blocks around it, worth close to 1, and 0.75 is worse than the plain mean. At $\lambda$ = 1 the mean never wins, so a block is simply worth its stalest cell. Linear growth on top of that nearly halved the worst idleness on the large field (333 to 178 minutes), since a forgotten cell keeps gaining value until it beats every other. It was removed all the same to keep the rule simple and every value between 0 and 1.

The revisit age with saturating values and the block max (32 × 32, four seeds; 64 × 64, two seeds):

| Revisit age | 32 × 32: full | idle | worst | 64 × 64: full | idle | worst |
| --- | --- | --- | --- | --- | --- | --- |
| 14 min | 45.0 | 20.2 | 73.0 | - | 138.6 | 360.0 |
| 28 min | 44.9 | 20.4 | 57.5 | 183.4 | 88.4 | 332.7 |
| 56 min | 43.0 | 27.7 | 73.6 | 167.8 | 85.4 | 324.7 |

## What is not done

- **The behaviour's bid.** `ExploreBehaviour` (see [`explore-behaviour.md`](./explore-behaviour.md)) scores EXPLORE from memory, but an explorer has no other behaviour, so the bid never competes with anything yet. The random-waypoint baseline of `tests/baselines.py` follows its seeder with `FollowSeederField`, a fixed bid of 0.5.
- **Results recorded before the move to the runner.** The tables above were measured with `CoverageTracker`, which no longer exists, and with a perfect pool that shared every sensed fact instantly. The experiments now share over the channel, and the "no sharing" variants are a radio range of 0, which also turns off the peer discount. The tables need a re-run with `studies/explore_variants.py`.
- **Holes on large fields.** On 64 × 64 cells the worst idleness stays above 300 minutes. Uncapped linear growth fixed most of it (see "Earlier value rules") but was left out to keep the rule simple.
- **Choosing the revisit age.** It should grow with the time the swarm needs to sweep its field, which depends on the field and the number of robots; it is a fixed parameter for now, and the swarm-size experiment is still to do.
- **Decisions while moving.** Robots decide every 2.5 s while a move takes 10 to 14 s. The navigator then sees no progress for three ticks and hands over to the steerer, whose moves the simulation ignores while the robot is moving. This is harmless but wasted work.
