# Research note — from which pest spread rate is a swarm needed? (#74)

Status: describes the grid-search baseline (`GridSweepSeeder` and `grid_sweep_brain` in `tests/baselines.py`), the study `studies/spread_threshold.py`, and its results. Reproducible with:

```bash
uv run python -m studies.spread_threshold --seeds 5 --hours 3 --jobs 16
```

## The question

A swarm is only worth its complexity if it does something simpler robots cannot. The simplest alternative is a grid search: utility robots that each sweep a strip of the field in a fixed pattern and treat whatever they drive over, without communicating. Such a sweep revisits every cell at a fixed period, so it should keep slow pests under control but fall behind once pests spread faster than the sweep comes back. The swarm reacts to a pest as soon as a scout sees it, so it should hold out to a higher spread rate. The study looks for that crossover.

## Setup

The field and pests of the base demo (`PestScenario.smoke`, see `smoke.md`): 32 × 32 cells of 10 m with a river and two rocks, 15 pests per hour, severity growing 0.05 per minute. The spread rate is the chance per minute that a pest of severity 1 infests each free neighbour (`WorldTick._spread`); it is varied over 0, 0.002, 0.005, 0.01 and 0.02. Five seeds per point, three simulated hours per run, sampled every minute.

| System | Robots | Behaviour |
|---|---|---|
| `grid-10` | 10 utility robots at 0.5 m/s | each sweeps its own strip of about three columns in a lawnmower pattern and treats the pest underfoot; no radio |
| `swarm-5+5` | 5 scouts (1.5 m/s) + 5 killers (0.5 m/s) | the base demo: scouts explore, killers treat what they hear of over a 300 m radio |
| `swarm-10+10` | 10 scouts + 10 killers | the same swarm with twice the robots |

The utility robots have the killers' speed and abilities. A robot moves one cell in 20 s but decides only every 15 s, so in this simulation every robot, swept or swarm, spends 30 s per cell; a full sweep of a strip takes 50 to 65 minutes.

A run is out of control when its infested cells still grow at the end: their mean over the last hour is more than 1.5 times, and more than 10 cells above, their mean over the hour before. Hourly means keep the ups and downs of a steady run from counting. A system's threshold is the lowest spread rate at which most of its seeds are out of control.

## Results

Mean infested cells over the run, with the standard deviation over the seeds and the seeds out of control:

| Spread | grid-10 | swarm-5+5 | swarm-10+10 |
| --- | --- | --- | --- |
| 0 | 5.7 (sd 1.0; 0/5 lost) | 12.8 (sd 6.3; 0/5 lost) | 12.2 (sd 6.4; 0/5 lost) |
| 0.002 | 7.8 (sd 1.4; 0/5 lost) | 24.2 (sd 10.3; 4/5 lost) | 19.7 (sd 13.3; 2/5 lost) |
| 0.005 | 16.1 (sd 3.8; 4/5 lost) | 56.3 (sd 21.3; 5/5 lost) | 47.4 (sd 34.9; 4/5 lost) |
| 0.01 | 76.9 (sd 31.3; 5/5 lost) | 142.7 (sd 54.8; 5/5 lost) | 119.4 (sd 60.2; 5/5 lost) |
| 0.02 | 230.8 (sd 29.8; 5/5 lost) | 349.4 (sd 46.9; 5/5 lost) | 312.0 (sd 62.0; 5/5 lost) |

Threshold: `grid-10` 0.005, `swarm-5+5` 0.002, `swarm-10+10` 0.005.

![Mean infested cells against the spread rate](../figures/spread-threshold.png)

Infested cells at the end of each hour, mean over the seeds:

| Spread | grid-10 (1 / 2 / 3 h) | swarm-5+5 | swarm-10+10 |
| --- | --- | --- | --- |
| 0 | 5.6 / 6.0 / 5.8 | 11.4 / 16.0 / 20.4 | 9.6 / 17.6 / 18.6 |
| 0.002 | 9.4 / 7.4 / 9.2 | 15.0 / 29.0 / 53.0 | 13.8 / 24.6 / 39.2 |
| 0.005 | 11.4 / 17.2 / 38.4 | 23.6 / 70.6 / 168.6 | 25.0 / 61.6 / 103.2 |

## What the results show

- **There is no spread rate in the range tested at which the swarm is needed: the grid search does better at every rate.** It keeps two to three times fewer cells infested than either swarm, stays steady up to a spread rate of 0.002, and only loses control from 0.005. The swarm of five plus five already loses control at 0.002, and even without spread its infested cells creep up hour by hour (11 → 16 → 20), because its killers clear only about half the pests (`smoke.md`).
- **Twice the robots barely help the swarm.** The swarm of ten plus ten reaches the grid's threshold of 0.005, but with about three times as many infested cells at every rate. Its extra killers join the group the others already form, so they add little capacity.
- **Why the grid wins here.** Every grid robot both finds and treats, and each one has its own strip, so ten robots do ten robots' work without overlap. In the swarm only half the robots can treat, and the killers drive as one group, which leaves about one killer's worth of treating for the whole field. The swarm's strength, reacting to news from anywhere, is spent driving five killers to the same pest.

## What would change the outcome

The comparison is fair, but it tests the swarm as it is now. Two directions are likely to move the crossover:

- **Fix the swarm's weak points first:** spread the killers over the known pests instead of letting them herd, and fix the EXPLORE oscillation at block edges (see `smoke.md`). Both raise the swarm's clearing rate directly, and the study can then be run again unchanged.
- **Test where a grid search scales badly:** a larger field (a sweep's period grows with the area, a swarm's response does not), fewer robots per area, or robots that fail (a failed sweeper leaves its whole strip unwatched, while the swarm covers for it). The study's systems and spread rates are a list at the top of `studies/spread_threshold.py`, so these are small additions.
