# EXPLORE: drawing robots to stale information (#41)

This document describes the EXPLORE field seeder, how it is measured, and what the measurements say. The seeder is `ExploreSeeder` in `swarm/explore.py`; the measurements come from `CoverageTracker` (`simulation/coverage.py`), the runnable experiment in `swarm_field.experiments`, and `tests/benchmark_explore.py`. How a seeder fits into the brain is described in [`robot-memory-and-navigation.md`](./robot-memory-and-navigation.md) under "Writing a behaviour".

Exploring a new field and monitoring a known one are one objective. A cell is worth visiting again by how long ago anyone observed it, and a never-seen cell is simply the oldest information there is. A robot observes only the cell it stands on, so visiting a cell is observing it.

## Value of a cell

A cell is worth nothing for a while after it was observed, and from then on it is *stale* and worth more the older its information is:

$$v(c) = \begin{cases} 0 & \text{if } a(c) < A \\ g\big(a(c)\big) & \text{otherwise} \end{cases} \qquad a(c) = t_\text{now} - t_\text{obs}(c)$$

| Symbol | Meaning |
| --- | --- |
| $v(c)$ | What observing cell $c$ again is worth. |
| $a(c)$ | The age of the cell's information. |
| $t_\text{now}$ | The current time. |
| $t_\text{obs}(c)$ | When the cell was last observed, by the robot or a peer; $-\infty$ if never. |
| $A$ | The revisit age (`ExploreConfig.revisit_age`): how long a cell is worth nothing after an observation. The one parameter of the value rule. |
| $g$ | How a stale cell's value grows (`ExploreConfig.growth`), 0.5 at $A$ in both forms below. |

The growth has two forms (`Growth`):

| Form | $g(a)$ | Never observed |
| --- | --- | --- |
| `SATURATING` | $1 - 2^{-a / A}$: 0.5 at $A$, 0.75 at $2A$, approaching 1 | 1 |
| `LINEAR` | $a / 2A$: 0.5 at $A$, 1 at $2A$, without bound | as if observed at the start of the run (time 0), but at least 1 |

The saturating form is $1 - e^{-a/\tau}$ with $\tau = A / \ln 2$, the rule of the first benchmarks below, which had a separate $\tau$ and a stale threshold $\theta$ = 0.5; $A = \tau \ln \frac{1}{1 - \theta}$ maps one onto the other. Cells memory knows as obstacles are worth 0. Memory keeps $t_\text{obs}$ for every cell in one array (`AgentMemory.last_observed_times`), so the values of all cells are one vectorised expression.

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

**Blocks that grow with the distance.** `ExploreConfig.block_sides` lists the block sides, finest first. The robot's own block has the first side. The blocks of each side within `near_blocks` blocks of the robot's are seeded one per block, and the next side covers what lies beyond; the last side covers everything left. A block is valued over the cells no finer block has claimed. `(16,)` uses 16 × 16 blocks everywhere, `(8,)` 8 × 8 blocks, and `(8, 16)` 8 × 8 blocks around the robot and 16 × 16 ones farther away.

## When the field is rebuilt

A field leads to one target, and a robot that observes only its own cell uses a target up by reaching it. So the field has to be rebuilt once per swept cell, and the key (`ExploreSeeder.field_key`) is made to change exactly then: it holds which cells of the robot's block are stale, the robot's block, a time bucket (`ExploreConfig.bucket`, which bounds how long the other blocks' means may drift), and the peers' blocks when the discount is on. Travelling over fresh ground and peers sweeping other blocks cause no rebuild.

Rebuilding once per swept cell makes the window of the field matter. `FieldSeeder.window_blocks` says how many 16 × 16 blocks around the robot's block are searched cell by cell; beyond them the quadtree's leaves are used. EXPLORE uses 1. A window of 0 builds about three times faster. With 16 × 16 blocks, though, every neighbouring block is then one leaf, measured from its centre, so it looks up to about 17 cells farther than it is, and robots that should move on to it stay; in a first, single-seed comparison redundancy went from 0.24 to 0.39. With 8 × 8 blocks, a window of 0 is nearly as good as 1 (see the results).

## Spreading the robots: the peer discount

$$v'(c) = \frac{v(c)}{1 + n_{b(c)}}, \qquad w'_b = \frac{w_b}{1 + n_b}$$

| Symbol | Meaning |
| --- | --- |
| $n_b$ | The number of peers whose reported cell or reported target lies in block $b$. |
| $b(c)$ | The block containing cell $c$. |

Every robot broadcasts its cell and its target at every decision point (`MessageKind.STATUS`), and memory keeps the newest report per peer. With `ExploreConfig.spread_peers`, a block's values are divided by one plus the number of peers in it or heading to it, counting only reports younger than `peer_max_age`. The discount applies to the robot's own block too, so a robot sharing its block with a peer is drawn to a block nobody covers. There is no kernel and no further parameter. Robots are thus made to spread explicitly, not only through shared observations making each other's ground look fresh.

## Parameters

The defaults are the ones that did best in the benchmark below.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `revisit_age` | 28 min | How long a cell is worth nothing after an observation. |
| `growth` | `LINEAR` | How a stale cell's value grows: linear without a cap, or saturating towards 1. |
| `beta` | 64 | Cells of travel a fully stale cell or block is worth. |
| `bucket` | 5 min | Longest drift of the other blocks' values between rebuilds. |
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

Five robots start side by side in a corner of a 32 × 32 field of 10 m cells and run for three simulated hours; each row is the mean over four seeds. A straight move takes 10 s, so visiting one new cell per move, the swarm would need about 34 minutes to cover the field. `t90` and `full` are the minutes until 90 % and all of the field had been visited (`-` when a seed never got there). `idle` and `worst` are the mean and worst idleness in minutes over the second half of the run, and `redund` the share of redundant visits. Unless a row says otherwise, the parameters are the defaults: 8 × 8 blocks, the discount on, a revisit age of 28 min, linear growth and $\beta$ = 64. "Obstacles" is the river with one bridge and two rocks of `river_and_rocks`.

| variant | t90 | full | idle | worst | redund | wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| random waypoint | - | - | 50.4 | 180.0 | 0.550 | 7 |
| explore 16 | 101.3 | - | 67.9 | 148.8 | 0.723 | 82 |
| explore 16 + discount | 39.6 | 46.7 | 22.9 | 55.0 | 0.241 | 99 |
| explore 8 | 125.7 | - | 79.5 | 167.0 | 0.770 | 57 |
| explore 8 + discount | 34.8 | 45.1 | 21.7 | 53.1 | 0.175 | 70 |
| explore 8/16 + discount | 36.9 | 48.9 | 21.3 | 56.1 | 0.187 | 68 |
| explore 8 + discount, window 0 | 35.3 | 43.3 | 22.1 | 56.0 | 0.190 | 27 |
| explore 8, no sharing | 140.4 | 159.0 | 74.6 | 178.0 | 0.757 | 54 |
| explore 8 + discount, no sharing | 85.7 | 131.7 | 35.0 | 155.8 | 0.595 | 55 |
| random waypoint, obstacles | - | - | 58.0 | 180.0 | 0.647 | 6 |
| explore 8, obstacles | - | - | 80.4 | 180.0 | 0.793 | 43 |
| explore 8 + discount, obstacles | 35.5 | 45.3 | 22.9 | 57.7 | 0.294 | 48 |

**The peer discount is what makes EXPLORE work.** Without it, the robots move as a herd: they all head for the same next block and sweep it together, so three quarters of their visits are redundant and they keep the field less fresh than random waypoints (mean idleness 68 or 80 minutes against 50). With it, mean idleness drops to about 22 minutes and the worst to about 55 minutes, against 180 for the baseline, and the field is covered in about 45 minutes. The same holds with obstacles.

**Block size matters little once the discount is on.** 8 × 8 blocks give the lowest redundancy, and blocks that grow from 8 to 16 cells with the distance do as well. **Sharing observations matters:** without it, every robot only knows the cells it visited itself, and even with the discount mean idleness rises to 35 minutes. **A window of 0 is as good and about 2.5 times faster** with 8 × 8 blocks, since it still searches the robot's whole 16 × 16 quadtree block cell by cell, which holds some of the neighbouring 8 × 8 blocks.

### Parameters

| variant | t90 | full | idle | worst | redund | wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| defaults (revisit age 28 min, beta 64) | 34.8 | 45.1 | 21.7 | 53.1 | 0.175 | 71 |
| revisit age 14 min | 36.5 | 46.2 | 22.0 | 52.6 | 0.194 | 67 |
| revisit age 42 min | 34.8 | 43.0 | 25.1 | 73.4 | 0.217 | 75 |
| revisit age 84 min | 34.8 | 43.0 | 43.2 | 120.9 | 0.213 | 50 |
| beta 16 | 36.8 | 48.5 | 21.6 | 85.1 | 0.219 | 72 |
| beta 32 | 36.4 | 45.8 | 22.2 | 71.8 | 0.174 | 57 |
| beta 128 | 34.8 | 45.1 | 21.9 | 52.3 | 0.192 | 40 |

| variant | t90 | full | idle | worst | redund | wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| A 14: linear | 36.5 | 46.2 | 22.0 | 52.6 | 0.194 | 69 |
| A 14: saturating | 35.0 | 45.0 | 20.2 | 73.0 | 0.168 | 69 |
| A 28: linear | 34.8 | 45.1 | 21.7 | 53.1 | 0.175 | 65 |
| A 28: saturating | 34.8 | 44.9 | 20.4 | 57.5 | 0.146 | 67 |
| A 56: linear | 34.8 | 43.0 | 28.0 | 79.6 | 0.222 | 62 |
| A 56: saturating | 34.8 | 43.0 | 27.7 | 73.6 | 0.213 | 46 |

A revisit age of 14 to 28 minutes does best; 84 minutes is too long, since cells then wait more than twice the sweep time to become targets again. $\beta$ needs to be 64 cells or more, for a stale cell across the field to beat a barely stale one next door. On this field linear and saturating growth are close: saturating has a slightly lower mean idleness, linear a lower worst idleness.

### A larger field

The same variants on a 64 × 64 field for six hours, mean over two seeds. Visiting one new cell per move, five robots would need about 136 minutes to cover it.

| variant | t90 | full | idle | worst | redund | wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| random waypoint | - | - | 147.2 | 360.0 | 0.216 | 24 |
| explore 16 | - | - | 201.5 | 360.0 | 0.740 | 235 |
| explore 16 + discount | 135.0 | 182.4 | 77.8 | 184.3 | 0.065 | 237 |
| explore 8 | - | - | 216.2 | 360.0 | 0.783 | 172 |
| explore 8 + discount | 143.7 | 168.0 | 83.6 | 178.4 | 0.131 | 204 |
| explore 8/16 + discount | 144.7 | 170.1 | 86.4 | 210.4 | 0.140 | 188 |
| explore 8 + discount, window 0 | 139.2 | 163.6 | 85.8 | 192.1 | 0.152 | 60 |
| explore 8, no sharing | - | - | 213.6 | 360.0 | 0.773 | 172 |
| explore 8 + discount, no sharing | - | - | 150.6 | 360.0 | 0.437 | 197 |
| random waypoint, obstacles | - | - | 162.2 | 360.0 | 0.356 | 20 |
| explore 8, obstacles | - | - | 216.6 | 360.0 | 0.798 | 183 |
| explore 8 + discount, obstacles | 150.0 | 177.3 | 81.7 | 189.5 | 0.129 | 209 |

| variant | t90 | full | idle | worst | redund | wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| A 14: linear | 144.6 | 167.5 | 87.5 | 187.2 | 0.158 | 164 |
| A 14: saturating | 203.6 | - | 138.6 | 360.0 | 0.076 | 144 |
| A 28: linear | 143.7 | 168.0 | 83.6 | 178.4 | 0.131 | 159 |
| A 28: saturating | 146.8 | 183.4 | 88.4 | 332.7 | 0.088 | 163 |
| A 56: linear | 143.2 | 171.4 | 98.1 | 265.0 | 0.171 | 135 |
| A 56: saturating | 143.2 | 167.8 | 85.4 | 324.7 | 0.098 | 123 |

The discount again cuts the mean idleness by more than half against no discount and to under 60 % of random waypoints, with little redundancy. The whole field is covered at about 170 minutes, and the worst idleness stays near 180 minutes. Here linear growth matters: with saturating growth, the worst idleness is about 330 minutes, and with a short revisit age (14 min) the field is never fully covered. The same revisit age of 28 minutes does best on both fields, so with linear growth it does not have to scale with the field.

### Fixing holes: before and after

The first version valued a block by the mean value of its cells and let a stale cell's value saturate at 1 ($1 - e^{-a/\tau}$ with $\tau$ = 40 min, which is $A$ = 28 min). On large fields it left holes: a single forgotten cell adds only $1/64$ to the mean of an 8 × 8 block, so its block looks almost done, and with saturating values a never-visited cell is worth hardly more than any cell visited a while ago. Two fixes were compared, alone and together (defaults otherwise, same seeds as above):

| Value rule | 32 × 32: full | idle | worst | 64 × 64: full | idle | worst |
| --- | --- | --- | --- | --- | --- | --- |
| saturating, block mean (first version) | 46.3 | 21.4 | 71.2 | - | 91.1 | 360.0 |
| saturating, $\max(\text{mean}, 0.5 \max)$ | 48.1 | 22.1 | 78.1 | - | 91.1 | 360.0 |
| saturating, $\max(\text{mean}, 0.75 \max)$ | 45.1 | 24.4 | 87.2 | - | 106.5 | 341.2 |
| saturating, block max | 44.9 | 20.4 | 57.5 | 183.4 | 88.4 | 332.7 |
| linear, block mean | 45.4 | 22.1 | 71.8 | 202.1 | 86.2 | 325.1 |
| **linear, block max (current)** | 45.1 | 21.7 | 53.1 | 168.0 | 83.6 | 178.4 |

A block worth $\max(\text{mean}, \lambda \max)$ only helps at $\lambda$ = 1: with 0.25 or 0.5 nothing changes on the large field, since a block whose only stale cell is worth at most 0.5 still loses to the old blocks around it, worth close to 1, and 0.75 is worse than the plain mean. At $\lambda$ = 1 the mean never wins, so $\lambda$ is bound to 1 and a block is simply worth its stalest cell. Linear growth then removes most of what is left: a forgotten cell keeps gaining value until it beats every other, and the worst idleness on the large field drops from 333 to 178 minutes.

## What is not done

- **The behaviour.** `ExploreBehaviour` (Jin, branch `41-explore-objective`) scores EXPLORE against other objectives. The experiments use the stand-in `FollowSeederField`.
- **Sharing observations.** Belief sharing (#34) is replaced by a perfect, test-side pool in the experiments.
- **Unbounded scores.** With linear growth, a seed's value has no upper bound, and so neither has the potential a behaviour reads to score its bid. The arbiter (#39) will need a cap or a normalisation of the score, while the seeds can stay uncapped.
- **The swarm-size experiment.** Whether the revisit age can stay fixed as the number of robots changes is untested; it did not need to change between the two field sizes.
- **Decisions while moving.** Robots decide every 2.5 s while a move takes 10 to 14 s. The navigator then sees no progress for three ticks and hands over to the steerer, whose moves the simulation ignores while the robot is moving. This is harmless but wasted work.
