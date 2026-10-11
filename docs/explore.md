# EXPLORE: drawing robots to stale information (#41)

This document describes the EXPLORE field seeder, how it is measured, and what the measurements say. The seeder is `ExploreSeeder` in `swarm/explore.py`; the measurements come from `CoverageTracker` (`simulation/coverage.py`), the runnable experiment in `swarm_field.experiments`, and `tests/benchmark_explore.py`. How a seeder fits into the brain is described in [`robot-memory-and-navigation.md`](./robot-memory-and-navigation.md) under "Writing a behaviour".

Exploring a new field and monitoring a known one are one objective. A cell is worth visiting again by how long ago anyone observed it, and a never-seen cell is simply the oldest information there is. A robot observes only the cell it stands on, so visiting a cell is observing it.

## Value of a cell

$$v(c) = 1 - e^{-(t_\text{now} - t_\text{obs}(c)) / \tau}$$

| Symbol | Meaning |
| --- | --- |
| $v(c)$ | What observing cell $c$ again is worth, from 0 (just observed) to 1 (long ago, or never). |
| $t_\text{now}$ | The current time. |
| $t_\text{obs}(c)$ | When the cell was last observed, by the robot or a peer; $-\infty$ if never, which gives $v = 1$. |
| $\tau$ | How fast information goes stale (`ExploreConfig.tau`). |
| $\theta$ | The stale threshold (`ExploreConfig.stale_threshold`): a cell is stale when $v(c) > \theta$. |

Cells memory knows as obstacles are worth 0. Memory keeps $t_\text{obs}$ for every cell in one array (`AgentMemory.last_observed_times`), so the values of all cells are one vectorised expression.

## Seeds: the robot's block cell by cell, other blocks one seed each

The field is split into aligned square blocks. The robot's own block gets one seed per stale cell. Every other block gets one seed, on its stalest cell, worth the block's mean value over its cells that are not obstacles. A seed starts at $-\beta$ times its worth.

$$\phi(x) = \min\Big( \min_{c \in \text{own block},\ v(c) > \theta} \big( -\beta \, v'(c) + d(x, c) \big),\ \min_{b \neq \text{own}} \big( -\beta \, \bar v'_b + d(x, r_b) \big) \Big)$$

| Symbol | Meaning |
| --- | --- |
| $\phi(x)$ | The potential at cell $x$; the robot steps downhill. |
| $\bar v_b$ | The mean value of block $b$. |
| $v'$, $\bar v'_b$ | The values after the peer discount below; the plain values when it is off. |
| $r_b$ | The stalest cell of block $b$, which only serves to route the robot to the block. |
| $d(x, c)$ | The route length from $x$ to $c$ over the quadtree, in cells. |
| $\beta$ | How many cells of travel a fully stale cell or block is worth (`ExploreConfig.beta`). |

One field thereby does two jobs. Inside its block the robot heads for the nearest stale cell, and a stale cell is worth at least as much as an averagely stale block and is much closer, so the robot finishes its block before leaving it. On an open block this produces a back-and-forth sweep without being told to. A single robot on a 16 × 16 field sweeps each block column by column and covers the field in 42.5 minutes with 16 × 16 blocks and 45.5 minutes with 8 × 8 blocks, where a straight move takes 10 seconds, so one new cell per move would take 42.7 minutes. Even counting its first revisits, only 3 to 5 % of its moves land on a cell it has already visited. Once the block has no stale cell, only the other blocks are left, and the field leads to the best of them; on entering that block, its cells become the per-cell seeds.

**Blocks that grow with the distance.** `ExploreConfig.block_sides` lists the block sides, finest first. The robot's own block has the first side. The blocks of each side within `near_blocks` blocks of the robot's are seeded one per block, and the next side covers what lies beyond; the last side covers everything left. A block is valued over the cells no finer block has claimed. `(16,)` uses 16 × 16 blocks everywhere, `(8,)` 8 × 8 blocks, and `(8, 16)` 8 × 8 blocks around the robot and 16 × 16 ones farther away.

## When the field is rebuilt

A field leads to one target, and a robot that observes only its own cell uses a target up by reaching it. So the field has to be rebuilt once per swept cell, and the key (`ExploreSeeder.field_key`) is made to change exactly then: it holds which cells of the robot's block are stale, the robot's block, a time bucket (`ExploreConfig.bucket`, which bounds how long the other blocks' means may drift), and the peers' blocks when the discount is on. Travelling over fresh ground and peers sweeping other blocks cause no rebuild.

Rebuilding once per swept cell makes the window of the field matter. `FieldSeeder.window_blocks` says how many 16 × 16 blocks around the robot's block are searched cell by cell; beyond them the quadtree's leaves are used. EXPLORE uses 1. A window of 0 builds about three times faster. With 16 × 16 blocks, though, every neighbouring block is then one leaf, measured from its centre, so it looks up to about 17 cells farther than it is, and robots that should move on to it stay; in a first, single-seed comparison redundancy went from 0.24 to 0.39. With 8 × 8 blocks, a window of 0 is nearly as good as 1 (see the results).

## Spreading the robots: the peer discount

$$v'(c) = \frac{v(c)}{1 + n_{b(c)}}, \qquad \bar v'_b = \frac{\bar v_b}{1 + n_b}$$

| Symbol | Meaning |
| --- | --- |
| $n_b$ | The number of peers whose reported cell or reported target lies in block $b$. |
| $b(c)$ | The block containing cell $c$. |

Every robot broadcasts its cell and its target at every decision point (`MessageKind.STATUS`), and memory keeps the newest report per peer. With `ExploreConfig.spread_peers`, a block's values are divided by one plus the number of peers in it or heading to it, counting only reports younger than `peer_max_age`. The discount applies to the robot's own block too, so a robot sharing its block with a peer is drawn to a block nobody covers. There is no kernel and no further parameter. Robots are thus made to spread explicitly, not only through shared observations making each other's ground look fresh.

## Parameters

The defaults are the ones that did best in the benchmark below.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `tau` | 40 min | How fast information goes stale. About 0.6 times the time the swarm needs to sweep its field once does best (see the results); 40 min suits five robots on 32 × 32 cells. |
| `stale_threshold` | 0.5 | A cell worth more than this is stale. |
| `beta` | 64 | Cells of travel a fully stale cell or block is worth. |
| `bucket` | 5 min | Longest drift of the other blocks' means between rebuilds. |
| `block_sides` | `(8,)` | Block sides, finest first. |
| `near_blocks` | 1 | How far each block side but the last reaches. |
| `spread_peers` | on | The peer discount. |
| `peer_max_age` | 1 min | Older peer reports are left out of the discount. |
| `window_blocks` | 1 | Blocks around the robot's searched cell by cell. |

## Measuring exploration

`CoverageTracker` measures a run from ground truth, independently of what the robots believe. A `CoverageSample` event records the robots' cells every second, which is shorter than any move, so no visit is missed. Over the cells that are not obstacles:

| Measure | Definition | What it shows |
| --- | --- | --- |
| Coverage, `t90`, `full` | The share of cells visited at least once, and when 90 % and all of them were. | How fast the swarm explores a new field. |
| Mean idleness | The age $t - t_\text{last}(c)$ of a cell, averaged over cells and over the samples from `measure_from` on. A cell never visited is as old as the run. | How fresh the swarm keeps its information. |
| Worst idleness | The largest age in those samples. | The worst-case detection latency. |
| Redundancy | The share of visits to a cell visited less than 30 minutes before. | Wasted travel and robots getting in each other's way. |

The mean idleness is also the expected detection latency. A pest appearing at a uniformly random time in a cell with visit gaps $g_i$ waits $\sum_i g_i^2 / (2 \sum_i g_i)$ on average, and the time-averaged age of that cell is the same expression. So the benchmark measures how fast new pests are found without injecting any.

## The experiment and the benchmark

`swarm_field.experiments.build_explore_run` assembles one run: a field with optional obstacles, robots starting side by side in a corner, each a real `ComposedAgent` on the real simulation loop, and a `CoverageTracker`. Every robot has one behaviour, `FollowSeederField`, which follows the field of its seeder and bids the negated potential, so the arbiter has nothing to choose and needs no mock. Robots start with the terrain map, since they never sense an obstacle they cannot enter. Their status goes over the real channel. Sharing observations is not built yet (#34), so `share_observations` stands in for it: every infestation fact a robot senses reaches every other robot at its next decision, as with perfect communication. `ExploreScenario.seed` staggers the robots' first decisions, so repeated runs interleave them differently. `RandomWaypointSeeder` is the baseline: one random target cell, and a new one once the robot stands on it.

`tests/benchmark_explore.py` runs every variant with several seeds and prints the mean of each measure; `tests/test_explore_swarm.py` checks the main findings on a small field.

## Results

Five robots start side by side in a corner of a 32 × 32 field of 10 m cells and run for three simulated hours; each row is the mean over four seeds. A straight move takes 10 s, so visiting one new cell per move, the swarm would need about 34 minutes to cover the field. `t90` and `full` are the minutes until 90 % and all of the field had been visited (`-` when a seed never got there). `idle` and `worst` are the mean and worst idleness in minutes over the second half of the run, and `redund` the share of redundant visits. Every row but the parameter sweep uses $\tau$ = 40 min and $\beta$ = 64, and "obstacles" is the river with one bridge and two rocks of `river_and_rocks`.

| variant | t90 | full | idle | worst | redund | wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| random waypoint | - | - | 50.4 | 180.0 | 0.550 | 13 |
| explore 16 | 97.1 | - | 63.7 | 163.8 | 0.672 | 125 |
| explore 16 + discount | 39.6 | 46.8 | 21.9 | 51.5 | 0.223 | 182 |
| explore 8 | 122.9 | - | 79.1 | 180.0 | 0.756 | 93 |
| explore 8 + discount | 34.2 | 47.5 | 20.8 | 67.6 | 0.166 | 117 |
| explore 8/16 + discount | 35.2 | 46.8 | 20.3 | 67.0 | 0.155 | 110 |
| explore 8 + discount, window 0 | 35.6 | 48.9 | 21.5 | 72.4 | 0.185 | 40 |
| explore 8, no sharing | 140.4 | 159.0 | 76.2 | 179.3 | 0.773 | 80 |
| explore 8 + discount, no sharing | 83.1 | - | 35.1 | 180.0 | 0.577 | 78 |
| random waypoint, obstacles | - | - | 58.0 | 180.0 | 0.647 | 10 |
| explore 8, obstacles | - | - | 81.2 | 180.0 | 0.772 | 81 |
| explore 8 + discount, obstacles | 35.0 | 46.8 | 20.1 | 77.5 | 0.242 | 82 |

**The peer discount is what makes EXPLORE work.** Without it, the robots move as a herd: they all head for the same next block and sweep it together, so two thirds of their visits are redundant and they keep the field less fresh than random waypoints (mean idleness 63.7 or 79.1 minutes against 50.4). With it, mean idleness drops to about 21 minutes and the worst to 52 to 78 minutes, less than half of the baseline's, and the field is covered in about 47 minutes. The same holds with obstacles.

**Block size matters little once the discount is on.** 8 × 8 blocks give the lowest redundancy and mean idleness, 16 × 16 blocks the lowest worst idleness, and blocks that grow from 8 to 16 cells with the distance do as well as 8 × 8 blocks. The defaults use 8 × 8 blocks.

**Sharing observations matters.** Without it, every robot only knows the cells it visited itself, and even with the discount mean idleness rises to 35 minutes and some seeds never cover the field.

**A window of 0 is nearly as good and 3 times faster** with 8 × 8 blocks. A window of 0 still searches the robot's whole 16 × 16 quadtree block cell by cell, which holds some of the 8 × 8 blocks next to the robot's, so their distances stay exact. With 16 × 16 blocks every neighbouring block lies outside it.

| variant | t90 | full | idle | worst | redund | wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| defaults (tau 40 min, beta 64, threshold 0.5) | 34.2 | 47.5 | 20.8 | 67.6 | 0.166 | 121 |
| tau 20 min | 34.6 | 56.1 | 20.4 | 72.2 | 0.188 | 100 |
| tau 60 min | 34.5 | 44.2 | 22.2 | 76.6 | 0.198 | 116 |
| tau 120 min | 34.2 | 44.5 | 43.4 | 140.0 | 0.525 | 125 |
| beta 16 | 41.1 | 84.7 | 26.7 | 133.0 | 0.332 | 119 |
| beta 32 | 34.4 | 68.9 | 20.8 | 91.0 | 0.238 | 82 |
| beta 128 | 34.4 | 46.0 | 21.7 | 74.1 | 0.158 | 84 |
| stale threshold 0.3 | 34.2 | 52.3 | 20.6 | 67.7 | 0.177 | 86 |
| stale threshold 0.7 | 34.2 | 43.7 | 24.7 | 78.6 | 0.256 | 91 |

$\tau$ should be of the order of the time the swarm needs to sweep its field, about 34 minutes here. Much longer (120 min), and a cell swept half an hour ago looks almost as fresh as one swept a minute ago, so the robots stop telling recently swept blocks from old ones. $\beta$ needs to be large enough (64 cells or more here) for a stale block across the field to beat a half-stale one next door.

### A larger field

The same variants on a 64 × 64 field for six hours, mean over two seeds. Visiting one new cell per move, five robots would need about 136 minutes to cover it.

| variant | t90 | full | idle | worst | redund | wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| random waypoint | - | - | 147.2 | 360.0 | 0.216 | 45 |
| explore 16 | - | - | 201.9 | 360.0 | 0.699 | 338 |
| explore 16 + discount | 172.8 | - | 104.8 | 360.0 | 0.026 | 438 |
| explore 8 | - | - | 215.9 | 360.0 | 0.781 | 319 |
| explore 8 + discount | 142.6 | - | 96.9 | 344.9 | 0.088 | 356 |
| explore 8/16 + discount | 197.0 | - | 96.8 | 360.0 | 0.063 | 370 |
| explore 8 + discount, window 0 | 140.1 | - | 90.6 | 330.4 | 0.078 | 104 |
| explore 8, no sharing | - | - | 213.6 | 360.0 | 0.773 | 296 |
| explore 8 + discount, no sharing | - | - | 168.9 | 360.0 | 0.454 | 320 |
| random waypoint, obstacles | - | - | 162.2 | 360.0 | 0.356 | 45 |
| explore 8, obstacles | - | - | 223.3 | 360.0 | 0.795 | 263 |
| explore 8 + discount, obstacles | 149.4 | - | 110.1 | 360.0 | 0.095 | 295 |

The discount again halves the mean idleness against no discount and brings it well below random waypoints, with very little redundancy. 90 % of the field is covered at about 140 minutes, close to the best possible, but no run covers all of it: a few cells are left behind as holes, and they keep the worst idleness at the length of the run. The cause is $\tau$. With 40 minutes against a sweep of more than two hours, every cell swept once is almost fully stale again by the time the swarm could come back, so a never-visited hole is worth hardly more than any other cell. Scaling $\tau$ with the field fixes most of it (defaults otherwise, two seeds):

| variant | t90 | full | idle | worst | redund | wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| tau 40 min | 142.6 | - | 96.9 | 344.9 | 0.088 | 245 |
| tau 80 min | 141.3 | 244.6 | 82.9 | 346.9 | 0.089 | 228 |
| tau 160 min | 140.9 | 262.1 | 90.0 | 317.9 | 0.126 | 243 |
| tau 160 min, beta 128 | 144.7 | 192.8 | 84.6 | 294.7 | 0.110 | 246 |

A rule of thumb from both fields: $\tau$ of about 0.6 times the time the swarm needs to sweep its field once, i.e. $\tau \approx 0.6 \, N \, t_\text{move} / k$ for $N$ cells, $k$ robots and $t_\text{move}$ per move (20 to 40 minutes on 32 × 32 cells, about 80 minutes on 64 × 64).

## What is not done

- **The behaviour.** `ExploreBehaviour` (Jin, branch `41-explore-objective`) scores EXPLORE against other objectives. The experiments use the stand-in `FollowSeederField`.
- **Sharing observations.** Belief sharing (#34) is replaced by a perfect, test-side pool in the experiments.
- **Holes.** A single stale cell left in a swept block only adds $1/64$ to its block's mean, so the swarm picks it up late, when a robot sweeps that block again. On large fields this keeps the worst idleness high; a never-observed cell is worth no more than one observed long ago. Scaling $\tau$ with the field helps; a bonus for never-observed cells, or the block's maximum next to its mean, are untested options.
- **Choosing $\tau$ automatically.** $\tau$ should follow the sweep time of the swarm, which depends on the field and the number of robots; it is a fixed parameter for now.
- **Decisions while moving.** Robots decide every 2.5 s while a move takes 10 to 14 s. The navigator then sees no progress for three ticks and hands over to the steerer, whose moves the simulation ignores while the robot is moving. This is harmless but wasted work.
