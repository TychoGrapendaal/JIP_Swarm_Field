# Research note — from which pest spread rate is a swarm needed? (#74)

Status: describes the grid-search baseline (`GridSweepSeeder` and `grid_sweep_brain` in `studies/baselines.py`), the study `studies/spread_threshold.py`, and its results, measured on the code of `main` after the body capabilities (#61), the battery lifecycle (#81), `SwarmScenario` (#84) and the spreading of killers over pests (#60). The results below are reproducible with (175 runs, about half an hour):

```bash
uv run python -m studies.spread_threshold --seeds 5 --hours 3 --jobs 6
```

Use at most about six to eight processes on a machine with 16 GB: the runs at the highest spread rates fill the field and take the most memory.

## The question

A swarm is only worth its complexity if it does something simpler robots cannot. The simplest alternative is a grid search: utility robots that each sweep a strip of the field in a fixed pattern and treat whatever they drive over, without communicating. Such a sweep revisits every cell at a fixed period, so it should keep slowly spreading pests under control but fall behind once pests spread faster than the sweep comes back. The swarm reacts to a pest as soon as an explorer sees it, so it should hold out to a higher spread rate. The study looks for that crossover, and for how both approaches scale with the number of robots.

## Setup

The field, pests and robots of the base demo (`SMOKE`, see `smoke.md`): 32 × 32 cells of 10 m with a river and two rocks, 948 crop cells, one dock in the corner. Every run starts with ten pests of severity 0.1 on random cells off the bottom row, drawn from the seed, so all systems face the same pests for the same seed. No pest appears after that, so pests only multiply by spreading, and the spread rate is the only variable: the chance per minute that a pest of severity 1 infests each free neighbour, over 0, 0.005, 0.01, 0.02, 0.03, 0.05 and 0.1. Severity grows 0.05 per minute, and a killer clears a pest of full severity in a minute (`FAST_TREATMENT`). Five seeds per point, three simulated hours per run, sampled every minute.

| System | Robots | Behaviour |
|---|---|---|
| `grid-10` | 10 utility robots (ground, 0.5 m/s) | each sweeps its own strip of about three columns in a lawnmower pattern and treats the pest underfoot; no radio |
| `grid-20` | 20 utility robots | the same with strips of one or two columns, as many treating robots as `swarm-20+20` |
| `swarm-5+5` | 5 explorers (drones, 2 m/s) + 5 seeded killers (ground, 0.5 m/s) | the base demo: explorers follow the EXPLORE field, killers claim and treat the pests they hear of over a 300 m radio |
| `swarm-10+10` | 10 explorers + 10 seeded killers | the same swarm with twice the robots |
| `swarm-20+20` | 20 explorers + 20 seeded killers | the same swarm with four times the robots |

The utility robots are killers with the grid-sweep brain, so they have the killers' speed, sensing and treatment. Each utility robot sweeps the strip numbered by its place in the swarm, so every strip has exactly one sweeper. Every killer and utility robot carries ten treatment doses, and nothing refills them, so ten treating robots can clear at most about 100 pests, twenty about 200 and five about 50. The bottom row, where robots start, has 30 free cells, so the 40 robots of `swarm-20+20` share start cells, as robots may share any cell.

A run is out of control when its infested cells still grow at the end: their mean over the last hour is more than 1.5 times, and more than 10 cells above, their mean over the hour before. A run whose infestation has filled the field stops growing for want of healthy cells, so it is out of control too when at least a quarter of the crop cells (237) are infested over its last hour. A system's threshold is the lowest spread rate at which most of its seeds are out of control.

## Results

Threshold, the lowest spread rate at which most seeds are out of control:

| System | Treating robots | Threshold |
|---|---|---|
| `grid-10` | 10 | 0.005 |
| `grid-20` | 20 | 0.02 |
| `swarm-5+5` | 5 | 0.01 |
| `swarm-10+10` | 10 | 0.03 |
| `swarm-20+20` | 20 | 0.1 |

Mean infested cells over the run, with the standard deviation over the seeds and the seeds out of control:

| Spread | grid-10 | grid-20 | swarm-5+5 | swarm-10+10 | swarm-20+20 |
| --- | --- | --- | --- | --- | --- |
| 0 | 2.1 (sd 0.3; 0/5 lost) | 1.2 (sd 0.3; 0/5 lost) | 1.2 (sd 0.1; 0/5 lost) | 0.8 (sd 0.1; 0/5 lost) | 0.7 (sd 0.1; 0/5 lost) |
| 0.005 | 17.7 (sd 7.4; 3/5 lost) | 2.0 (sd 0.8; 0/5 lost) | 3.9 (sd 3.2; 0/5 lost) | 0.9 (sd 0.1; 0/5 lost) | 0.8 (sd 0.1; 0/5 lost) |
| 0.01 | 87.5 (sd 33.4; 5/5 lost) | 9.0 (sd 13.8; 1/5 lost) | 16.9 (sd 10.9; 4/5 lost) | 1.3 (sd 0.3; 0/5 lost) | 0.9 (sd 0.2; 0/5 lost) |
| 0.02 | 290.4 (sd 44.0; 5/5 lost) | 98.7 (sd 45.6; 5/5 lost) | 173.0 (sd 77.5; 5/5 lost) | 4.9 (sd 6.0; 1/5 lost) | 1.3 (sd 0.5; 0/5 lost) |
| 0.03 | 421.0 (sd 62.2; 5/5 lost) | 228.7 (sd 76.6; 5/5 lost) | 348.6 (sd 78.9; 5/5 lost) | 51.1 (sd 49.0; 3/5 lost) | 1.6 (sd 0.5; 0/5 lost) |
| 0.05 | 568.5 (sd 45.1; 5/5 lost) | 448.7 (sd 71.1; 5/5 lost) | 541.2 (sd 56.6; 5/5 lost) | 295.5 (sd 108.8; 5/5 lost) | 31.1 (sd 42.0; 2/5 lost) |
| 0.1 | 693.3 (sd 54.6; 5/5 lost) | 681.3 (sd 43.0; 5/5 lost) | 686.2 (sd 17.7; 5/5 lost) | 659.2 (sd 28.1; 5/5 lost) | 365.8 (sd 230.9; 5/5 lost) |

![Mean infested cells against the spread rate](../figures/spread-threshold.png)

Seeds whose field was clean before the end, and when on average, and the pests cleared of those that appeared (means over the seeds):

| Spread | grid-10 | grid-20 | swarm-5+5 | swarm-10+10 | swarm-20+20 |
| --- | --- | --- | --- | --- | --- |
| 0 | 5/5 after 71 min; 10.0 of 10.0 | 5/5 after 40 min; 10.0 of 10.0 | 5/5 after 36 min; 10.0 of 10.0 | 5/5 after 23 min; 10.0 of 10.0 | 5/5 after 21 min; 10.0 of 10.0 |
| 0.005 | 0/5; 34.0 of 75.4 | 5/5 after 65 min; 19.0 of 19.0 | 3/5 after 49 min; 19.4 of 24.8 | 5/5 after 25 min; 12.2 of 12.2 | 5/5 after 24 min; 12.0 of 12.0 |
| 0.01 | 0/5; 78.8 of 315.8 | 4/5 after 75 min; 39.6 of 57.6 | 1/5 after 42 min; 39.8 of 85.4 | 5/5 after 35 min; 19.4 of 19.4 | 5/5 after 24 min; 16.2 of 16.2 |
| 0.02 | 0/5; 101.6 of 778.2 | 0/5; 122.0 of 415.2 | 0/5; 43.0 of 475.8 | 4/5 after 44 min; 38.4 of 50.0 | 5/5 after 30 min; 22.8 of 22.8 |
| 0.03 | 0/5; 109.0 of 936.6 | 0/5; 168.0 of 724.6 | 0/5; 40.6 of 808.4 | 2/5 after 44 min; 64.2 of 220.0 | 5/5 after 32 min; 30.6 of 30.6 |
| 0.05 | 0/5; 107.4 of 1039.8 | 0/5; 215.0 of 1062.6 | 0/5; 47.8 of 975.2 | 0/5; 92.0 of 766.2 | 3/5 after 27 min; 81.6 of 189.6 |
| 0.1 | 0/5; 105.6 of 1053.0 | 0/5; 212.6 of 1158.8 | 0/5; 45.0 of 993.0 | 0/5; 99.4 of 1047.4 | 0/5; 179.2 of 866.8 |

## What the results show

- **With the same number of treating robots, the swarm handles a spread rate five to six times higher than the grid search.** With ten treating robots, the grid loses control from 0.005 and the swarm of ten plus ten from 0.03; with twenty, the grid from 0.02 and the swarm of twenty plus twenty from 0.1.
- **Even with half the treating robots, the swarm does better.** The swarm of five plus five, with five killers, holds out to 0.01, twice the rate of the grid's ten sweepers, and clears the field without spread in 36 minutes against 71.
- **The swarm keeps scaling.** Each doubling raised its threshold: from 0.01 with five plus five to 0.03 with ten plus ten, and to 0.1 with twenty plus twenty. At 0.03, twenty plus twenty keeps the field nearly clean (1.6 infested cells, every seed cleared) where ten plus ten already loses three seeds of five.
- **The grid scales too, but from a lower level.** Twenty sweepers with strips half as wide pass every cell twice as often, which raises the grid's threshold from 0.005 to 0.02. That is still below the swarm with the same number of treating robots by a factor of five.
- **Without spread, more robots barely speed the swarm up.** It clears the ten starting pests in 23 minutes with ten plus ten and 21 with twenty plus twenty: the time goes to driving from the start corner to the pests, not to finding them.
- **Up to the grid's thresholds the difference comes from finding pests, not from treating more.** Where the grid loses control (0.005 for ten sweepers, 0.02 for twenty), it has used a third to two thirds of its doses, so what separates the systems is how soon a pest is found and reached.

So a grid search is only enough when pests barely spread: below about 0.005 per minute with ten robots and 0.02 with twenty, on this field. Above that a swarm is needed, and the spread rate it handles grows faster with its size than the grid's does.

## Caveats

- **The systems differ in hardware as well as in coordination.** The swarm's explorers are drones at 2 m/s that sense 3 × 3 cells, while the grid's utility robots are ground robots that sense one cell. That is the heterogeneous design being compared, but part of the swarm's advantage comes from its drones. A grid baseline of drones sweeping strips would separate the two.
- **Treatment doses limit the highest spread rates.** From 0.02 the grid of ten and the swarm of five plus five have used most of their doses, and at 0.1 the swarm of twenty plus twenty has cleared 179 of its about 200 doses' worth of pests. Those results compare treatment supply more than search; a rerun with more doses (`PestConfig.treatment_doses`) would show how far each system holds out on search alone.
- **The river shapes the grid's strips.** All robots start west of the river, and strips east of it are reached over the single bridge; a strip on the river itself has fewer cells to sweep than its neighbours. An open field (no obstacles) would show how much of the grid's weakness comes from this layout.
- **One dock for the whole swarm.** Explorers fly 30 minutes per charge and queue at the single dock, so over three hours part of their time goes to charging. This counts against the swarm, which has the explorers; the grid's ground robots last the run without charging.
- **Robots may share cells,** so the large swarms are not slowed by crowding, which ground robots in a real field would be.

## What would extend the result

- **Separate hardware from coordination** with a drone grid baseline, search from supply with more treatment doses, and the field layout from both with an open field.
- **Test where a grid search scales badly:** a larger field (a sweep's period grows with the area, a swarm's response does not), fewer robots per area, or robots that fail (a failed sweeper leaves its whole strip unwatched).
