# Research note — the base demo: five scouts, five killers, one objective each (#67)

Status: describes `PestScenario.smoke` in `src/swarm_field/experiments/pests.py` (also the catalogue entry `smoke`), the pest and killer measures of `MetricsCollector` in `src/swarm_field/simulation/metrics.py`, the smoke test in `tests/test_pest_smoke.py` and the study `studies/smoke.py`. The results below are reproducible with:

```bash
uv run python -m studies.smoke --seeds 5 --hours 3 --jobs 16
```

## The question

Does the simplest swarm that does the whole job clear the pests its scouts find? Five scouts only explore, five killers only treat, so every robot has exactly one objective and the arbiter never has to trade one off against another: each behaviour proposes one candidate per decision, and `UtilityArbiter.choose` takes the only bid. If this run does not clear pests, nothing built on top of it will. Each comparison changes one part of the pipeline (sensing, memory, sharing, the TREAT field, navigation, `Treat`) to see which one limits it.

## Setup

| Setting | Value |
|---|---|
| Field | 32 × 32 cells of 10 m, river with one bridge and two rocks, terrain known at the start |
| Robots | 5 scouts (1.5 m/s) and 5 killers (0.5 m/s), side by side on the bottom row |
| Brains | scouts: the mock explore behaviour on the EXPLORE field; killers: `TreatInfestation` on the TREAT field |
| Pests | 15 per hour at random free cells, growing 0.05 per minute, no spread |
| Communication | memory and status broadcast at every decision, 300 m radio |
| Decisions | every 15 s, staggered |
| Horizon | 3 h, sampled every minute; idleness measured from 30 min on |
| Seeds | 0 to 4, the same for every condition |

Comparisons: `scouts-only` (no killers), `random-waypoint` (scouts drive to random cells instead of following the EXPLORE field), `radio-50`, `radio-150` and `radio-unlimited`, `killers-0.1` (killers at 0.1 m/s) and `spread` (pests infest a neighbour with a chance of 0.05 per minute at full severity).

## The measures

Every pest is followed from appearing to clearing (`PestRecord`), so none of these depends on the sample period:

- **Detection**: from a pest appearing to the first robot sensing it infested.
- **News to killer** (knowledge latency): from the first detection to the first killer reporting the pest among its known pests (`AgentInspection.known_pests`). This isolates the communication from the scouting and the driving.
- **Response**: from the first detection to the pest being cleared.
- **Killer time split**: the share of the killers' decisions driving to a pest (a `Move` for the TREAT objective), treating, exploring and idle.
- **Same pest** (duplicate trips): of the killers' decisions driving to a pest, the share heading for a pest another killer is heading for as well.
- **Infested cells**, mean and peak over the samples, and pests cleared out of those that appeared.

## Results

Means over seeds 0 to 4, three simulated hours each. Times in minutes unless noted.

### Pest response

| Condition | Infested (mean / peak) | Cleared / appeared | Detection (min) | News to killer (s) | Response (min) | Killers driving / treating / idle | Same pest |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smoke | 12.8 / 21.8 | 19.0 / 39.4 | 23.7 | 3 | 11.9 | 58% / 1% / 42% | 79% |
| scouts-only | 21.6 / 45.0 | 0.0 / 45.0 | 25.9 | - | - | - | - |
| random-waypoint | 13.2 / 20.8 | 24.8 / 43.4 | 22.1 | 3 | 11.6 | 72% / 1% / 27% | 98% |
| radio-50 | 18.8 / 35.0 | 7.8 / 39.0 | 32.5 | 2462 | 25.8 | 13% / 0% / 86% | 66% |
| radio-150 | 12.1 / 21.4 | 21.6 / 39.4 | 22.9 | 119 | 17.1 | 60% / 1% / 40% | 78% |
| radio-unlimited | 12.6 / 21.8 | 19.4 / 39.4 | 22.7 | 3 | 13.1 | 59% / 1% / 40% | 79% |
| killers-0.1 | 17.5 / 29.2 | 10.0 / 39.0 | 24.7 | 3 | 41.6 | 70% / 0% / 29% | 77% |
| spread | 565.7 / 947.6 | 55.0 / 1002.6 | 19.7 | 2 | 3.6 | 22% / 65% / 13% | 89% |

### Monitoring and cost

| Condition | Coverage | Idleness (min, mean / worst) | Redundancy | Messages received | Wall-clock (s per simulated hour) |
| --- | --- | --- | --- | --- | --- |
| smoke | 78% | 57.0 / 161.4 | 66% | 127,658 | 122 |
| scouts-only | 87% | 58.8 / 180.0 | 57% | 28,239 | 73 |
| random-waypoint | 85% | 51.8 / 180.0 | 68% | 128,842 | 110 |
| radio-50 | 78% | 60.6 / 165.0 | 64% | 35,622 | 41 |
| radio-150 | 82% | 47.6 / 137.0 | 59% | 66,082 | 57 |
| radio-unlimited | 77% | 56.2 / 161.4 | 65% | 129,600 | 77 |
| killers-0.1 | 77% | 57.1 / 155.6 | 60% | 126,208 | 71 |
| spread | 95% | 43.0 / 176.9 | 44% | 127,121 | 64 |

### What the results show

- **The killers work, but as one.** The full swarm keeps 12.8 cells infested on average against 21.6 without killers, and halves the peak (21.8 against 45.0), so the pipeline from sensing to `Treat` works end to end. But it clears only 19 of 39 pests, about half. The reason is in the last column: 79 % of the killers' trips head for a pest another killer is heading for too, and the video shows the five killers driving as one group. Five killers then do the work of about one. `TreatSeeder` lowers a pest's value for peers heading there, but with one or two known pests every killer still picks the same one. Spreading the killers over the pests is the clearest next step.
- **News reaches the killers almost at once, from 150 m on.** At 300 m or unlimited range a killer knows of a pest about 3 s after a scout finds it, at 150 m about 2 minutes later, and at 50 m about 41 minutes later: the killers then idle 86 % of their decisions and clear only 7.8 pests. The information bottleneck of #60 lies between 50 and 150 m on this field. More range than 150 m buys nothing here but messages (66,000 received at 150 m against 128,000 at 300 m).
- **Driving is the larger part of the response.** Response time is about 12 minutes at 0.5 m/s and 42 minutes at 0.1 m/s, with news arriving in seconds in both, so with slow killers almost all of the response is driving.
- **The EXPLORE field adds nothing yet.** Random-waypoint scouts cover more of the field (85 % against 78 %) with a lower mean idleness (52 against 57 min), and their killers clear more pests. That matches the oscillation at block edges seen earlier, where groups of scouts get stuck moving back and forth between two blocks. Coverage after three hours is only 78 %, with 66 % of the visits redundant.
- **Five killers cannot keep up once pests spread.** With a spread chance of 0.05 per minute, about 1,000 pests appear in three hours and 566 cells are infested on average. The killers are busy all the time (65 % of their decisions treating), so the response to a single pest drops to 3.6 minutes, but they clear only 55.
- **Cost.** Wall-clock times come from 16 runs in parallel on one machine, so they compare conditions only roughly: the runs with most communication (300 m, unlimited, random-waypoint) are the slowest per simulated hour, and 50 m is the fastest, consistent with every robot merging every peer's memory in range (#66).

### Against the Definition of Done

- **Clearly fewer infested cells than scouts-only:** met, 12.8 against 21.6 on average.
- **Clears most of the pests it detects:** not met. It clears 19 of the 39 pests that appear; the count of pests detected is not in the summary, but detection latency is 24 minutes against a three-hour run, so most pests are detected well before the end. The herding of the killers is what holds it back.

## Demo video

`uv run python examples/pest_demo.py --smoke --hours 3 --save runs/smoke_video/smoke.mp4` records the base run (seed 0) as a 34-second video; it is not committed. At the end, 14 of 36 pests are cleared and the five killers sit on one cell.

## Smoke test

`tests/test_pest_smoke.py` places one pest four cells from the start and runs the smoke scenario without random pests for ten simulated minutes. It passes only if a scout detects the pest, a killer hears of it and clears it, so a regression anywhere in the pipeline makes it fail. It runs in about 8 s.
