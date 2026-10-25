# Research note — from which pest spread rate is a swarm needed? (#74)

Status: describes the grid-search baseline (`GridSweepSeeder` and `grid_sweep_brain` in `studies/baselines.py`), the study `studies/spread_threshold.py`, and its results. The results below were measured with the earlier `PestScenario.smoke`, before the body capabilities (#61), the battery lifecycle (#81), the move to `SwarmScenario` (#84) and the spreading of killers over pests (#60). A short check after those changes (one seed, one hour) already turned the conclusion around: the swarm kept spread 0.02 under control where the grid baseline did not. The results need a rerun with:

```bash
uv run python -m studies.spread_threshold --seeds 5 --hours 3 --jobs 8
```

Use at most about eight processes on a machine with 16 GB: with 18, a worker ran out of memory at the highest spread rates and the study was lost.

## The question

A swarm is only worth its complexity if it does something simpler robots cannot. The simplest alternative is a grid search: utility robots that each sweep a strip of the field in a fixed pattern and treat whatever they drive over, without communicating. Such a sweep revisits every cell at a fixed period, so it should keep slowly spreading pests under control but fall behind once pests spread faster than the sweep comes back. The swarm reacts to a pest as soon as a scout sees it, so it should hold out to a higher spread rate. The study looks for that crossover.

## Setup

The field of the base demo (`PestScenario.smoke`, see `smoke.md`): 32 × 32 cells of 10 m with a river and two rocks. Every run starts with ten pests of severity 0.1 on random cells off the bottom row, drawn from the seed, so all systems face the same pests for the same seed. No pest appears after that, so pests only multiply by spreading, and the spread rate is the only variable: the chance per minute that a pest of severity 1 infests each free neighbour (`WorldTick._spread`), over 0, 0.005, 0.01, 0.02 and 0.05. Severity grows 0.05 per minute. Five seeds per point, three simulated hours per run, sampled every minute.

| System | Robots | Behaviour |
|---|---|---|
| `grid-10` | 10 utility robots at 0.5 m/s | each sweeps its own strip of about three columns in a lawnmower pattern and treats the pest underfoot; no radio |
| `swarm-5+5` | 5 scouts (1.5 m/s) + 5 killers (0.5 m/s) | the base demo: scouts explore, killers treat what they hear of over a 300 m radio |
| `swarm-10+10` | 10 scouts + 10 killers | the same swarm with twice the robots |

The utility robots have the killers' speed and abilities. A robot moves one cell in 20 s but decides only every 15 s, so in this simulation every robot spends 30 s per cell; a full sweep of a strip takes 50 to 65 minutes.

A run is out of control when its infested cells still grow at the end: their mean over the last hour is more than 1.5 times, and more than 10 cells above, their mean over the hour before. A system's threshold is the lowest spread rate at which most of its seeds are out of control.

## Results

Mean infested cells over the run, with the standard deviation over the seeds and the seeds out of control:

| Spread | grid-10 | swarm-5+5 | swarm-10+10 |
| --- | --- | --- | --- |
| 0 | 2.1 (sd 0.3; 0/5 lost) | 4.2 (sd 1.8; 0/5 lost) | 2.7 (sd 0.6; 0/5 lost) |
| 0.005 | 13.8 (sd 6.4; 3/5 lost) | 36.0 (sd 21.1; 3/5 lost) | 27.1 (sd 28.5; 2/5 lost) |
| 0.01 | 83.9 (sd 28.1; 5/5 lost) | 138.9 (sd 22.9; 5/5 lost) | 83.8 (sd 45.0; 4/5 lost) |
| 0.02 | 277.5 (sd 42.1; 5/5 lost) | 360.8 (sd 37.3; 5/5 lost) | 286.2 (sd 51.6; 5/5 lost) |
| 0.05 | 570.7 (sd 54.3; 1/5 lost) | 619.3 (sd 14.2; 0/5 lost) | 570.1 (sd 39.3; 0/5 lost) |

Threshold: `grid-10` 0.005, `swarm-5+5` 0.005, `swarm-10+10` 0.01.

![Mean infested cells against the spread rate](../figures/spread-threshold.png)

Seeds whose field was clean before the end, and when, and the pests cleared out of those that appeared (means over the seeds):

| Spread | grid-10 | swarm-5+5 | swarm-10+10 |
| --- | --- | --- | --- |
| 0 | 5/5 clean after 69 min; 10.0 of 10.0 | 2/5 after 95 min; 8.0 of 10.0 | 3/5 after 115 min; 9.6 of 10.0 |
| 0.005 | 1/5 after 158 min; 35.8 of 62.4 | 0/5; 45.8 of 124.0 | 1/5 after 90 min; 29.4 of 87.6 |
| 0.01 | 0/5; 77.6 of 316.4 | 0/5; 55.0 of 426.4 | 0/5; 105.6 of 314.2 |
| 0.02 | 0/5; 102.2 of 768.8 | 0/5; 55.0 of 852.8 | 0/5; 110.0 of 802.0 |
| 0.05 | 0/5; 110.0 of 1049.4 | 0/5; 55.0 of 1003.0 | 0/5; 110.0 of 1034.0 |

## What the results show

- **Without spread, the grid search clears the field fastest and most reliably.** All ten pests are gone after 69 minutes on average, in every seed, about one sweep. The swarm of five plus five clears eight of ten and leaves pests behind in three seeds out of five; ten plus ten clears 9.6 and is clean in three seeds, after almost two hours.
- **The grid search stays ahead up to a spread rate of 0.005,** with less than half the infested cells of either swarm. At 0.005 every system already loses control in about half its seeds.
- **The larger swarm catches up from 0.01, but only by treating as much as the grid.** At 0.01 swarm-10+10 and grid-10 keep the same number of cells infested (84), and the swarm loses control in four seeds against five; by the criterion its threshold is 0.01, one step above the grid's. At 0.02 the two are again level. The swarm of five plus five is worse than both at every rate.
- **Above 0.01 every system is limited by its treatment supply, not by how it searches.** Each treatment uses a tenth of a robot's consumable and nothing refills it, so a robot treats at most eleven times. That is why the cleared pests stop at exactly 110 for ten treating robots and 55 for five, whatever the spread rate. From there the comparison measures how many robots can treat, not how well they find pests, and at 0.05 the field is simply full (about 940 of its 1,000 free cells infested), so the growth criterion no longer applies.

So within the range where search matters (spread rates up to about 0.01), there is no spread rate at which the swarm is needed: the grid search is as good or better. Only twice the swarm, with as many treating robots as the grid has in total, matches it at 0.01.

## What would change the outcome

- **Lift the consumable limit for this study,** by refilling it or giving robots more, so that from 0.01 on the systems are compared by how they find and reach pests rather than by how many treatments they carry. Until then, no result above 0.01 says anything about search.
- **Fix the swarm's weak points:** spread the killers over the known pests instead of letting them herd, and fix the EXPLORE oscillation at block edges (see `smoke.md`). Both raise the swarm's clearing rate directly; the study can be run again unchanged.
- **Test where a grid search scales badly:** a larger field (a sweep's period grows with the area, a swarm's response does not), fewer robots per area, or robots that fail (a failed sweeper leaves its whole strip unwatched).
