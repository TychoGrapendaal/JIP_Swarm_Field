# Research note — measuring swarm robustness to robot failure and localisation error

Status: describes the machinery implemented in `src/swarm_field/experiments/robustness.py` (`RobustnessCase`, `default_robustness_cases`, `measure_robustness`), `src/swarm_field/simulation/event.py` (`FailureCheck`) and `src/swarm_field/simulation/world.py` (`fail_robot`, `perceived_self`). The results below are produced by `examples/robustness_demo.py` and are reproducible from seed 101. **Read "What these results do and do not show" before quoting any number here**: the harness is verified end to end, but the outcome metrics are currently flat because the swarm's behaviors are stubs, so this note establishes the measurement and the experimental controls, not a finding about resilience.

## The question

A field of a few hundred small robots will lose robots: batteries run flat, motors jam, radios die. Before the swarm algorithm can be tuned for that environment, the simulation has to be able to degrade a run in a controlled way and report what the degradation cost. Two axes are worth perturbing separately, because they are the two ways a real robot's world and a real robot's belief come apart:

- **Failure.** A robot stops taking part and stays stopped. What "stops" means changes what the rest of the swarm loses, so four modes are modelled rather than one: `DEAD_BATTERY` (stopped and dark), `DISABLED` (stopped but still observable), `OFFLINE` (still deciding and moving, but nobody hears it) and `IMMOBILE` (stopped in place, but still sensing its own cell, deciding and communicating, so it stays in the network as a fixed relay; it does not know it is stuck).
- **Position error.** The robot keeps working but is told it is somewhere it is not. This is the weaker fault, and the interesting one: nothing in the simulation breaks, the robot simply acts on a wrong belief, and the question is whether the swarm's behaviour is robust to that.

## How a run stays comparable

The obvious way to build a robustness sweep — inject the fault from the same random source the world uses — is wrong, and quietly so. The pest dynamics draw from `ctx.rng` in event order, so adding a fault that consumes draws shifts every subsequent draw: the pests evolve differently, the field ends the run in a different state, and any difference in the metrics is uninterpretable. A sweep built that way measures the weather, not the fault.

So the context carries two independent random sources. `ctx.rng` is the world: the pest dynamics, nothing else. `ctx.noise` is the simulation's own hardware, and it is the only source `FailureCheck` and the position-error draw touch. Both are derived from the scenario seed, so a pair of runs is reproducible while their streams stay disjoint. The property is asserted directly in `tests/test_robustness.py`: a run with `probability=0.5` failures and a run with no failures see byte-identical `ctx.rng` streams, and the same holds for position error. In the results below, `targets_remaining` is identical to the cell across every condition, which is that guarantee showing up in the output.

Every condition in a sweep runs from the **same seed**, and each condition perturbs one axis at a time, so each row answers exactly one question. `default_robustness_cases()` is the baseline plus one axis at a time; the demo widens it to every failure mode.

## Results

`S2` (10 scouts, 5 killers, 1600 cells), seed 101, 6 simulated hours, sampled hourly. `coverage` is shown with its difference from the baseline, `energy/m2` is battery fraction spent per square metre.

| condition | active | targets | coverage | vs base | latency (ms) | energy/m² |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 15 | 1600 | 0.084 | +0.000 | 0 | 1.620e-04 |
| failure `dead_battery` 0.001 | 11 | 1600 | 0.084 | +0.000 | 0 | 1.282e-04 |
| failure `dead_battery` 0.005 | 1 | 1600 | 0.084 | +0.000 | 0 | 7.617e-05 |
| failure `disabled` 0.001 | 11 | 1600 | 0.084 | +0.000 | 0 | 1.282e-04 |
| failure `disabled` 0.005 | 1 | 1600 | 0.084 | +0.000 | 0 | 7.617e-05 |
| failure `offline` 0.001 | 11 | 1600 | 0.084 | +0.000 | 0 | 1.282e-04 |
| failure `offline` 0.005 | 1 | 1600 | 0.084 | +0.000 | 0 | 7.617e-05 |
| position error 0.1 | 15 | 1600 | 0.084 | +0.000 | 0 | 1.620e-04 |
| position error 0.3 | 15 | 1600 | 0.084 | +0.000 | 0 | 1.620e-04 |

The harness scales to the large roster: `S5` (40 robots, 4900 cells) over 3 hours gives the same shape, with 40 / 35 / 14 active robots at the three failure levels.

The presets build a SCOUT / KILLER / BATTERY roster, but it does not yet affect anything: `build_agent` takes no `body` because there is no `CapabilitySet` gate, so all 15 robots in `S2` are assembled into a byte-identical `ComposedAgent`. The body mix is therefore held fixed across the presets and is not a variable in this sweep.

## What these results do and do not show

**The machinery works, and the controls hold.** `active_robots` tracks the failure rate exactly as it should (15 → 11 → 1 for S2, 40 → 35 → 14 for S5), so failure injection reaches the world. `energy/m²` falls in proportion to the robots lost, because a stopped robot stops spending. Most importantly `targets_remaining` is identical to the cell in every row: the faults did not disturb the pests, which is exactly the guarantee the separate random source exists to provide.

> **Update.** The table was recorded when `TreatSeeder` was a stub. `TreatSeeder` now seeds the known pests and robots share their memory, but the default `BrainSpec` still carries only `TreatInfestation`. It bids only on pests the robot knows of, and a robot now senses only its own cell, so with nothing exploring the swarm still idles; `messages_sent` is no longer zero. The figures need a re-run with a brain that explores.

**The outcome metrics cannot discriminate yet, and the reason is the algorithm, not the measurement.** `coverage`, `targets_remaining` and `detection_latency` are flat for two separate reasons, both of which are stub-level gaps rather than simulation gaps:

- The swarm does not move. `TreatSeeder.seeds()` returns `()`, so `Navigator.field_for` hands the behaviour no field, `TreatInfestation.propose` returns no candidate at every decision and `ComposedAgent` commits `Idle`. All 15 robots therefore sense the same 5 × 5 neighbourhood they started in and nothing else — 135 distinct cells of 1600, which is the 0.084 in the table. A robot that never leaves its start cell cannot be made to cover less by losing peers, because its peers contributed cells it already covers.
- Position error currently reaches the algorithm only through `perception.self_state.position`. Sensing uses the true position, and with no behavior acting on its belief, a displaced belief has nothing to affect. The mechanism itself is tested at the boundary (`test_position_error_displaces_the_position_the_robot_believes`, and that a displaced belief is clamped inside the field and leaves ground truth untouched), so the gap is downstream of it.

Two secondary effects compound this. The three failure modes are indistinguishable in the table for the same reason: with no communication implemented, an `OFFLINE` robot costs the swarm nothing extra over a `DEAD_BATTERY` one, so the modes cannot separate until `messages_sent` is non-zero. And the detection latency reads `0` rather than a distribution for a concrete reason: only the KILLER and BATTERY robots are placed away from the initial infestation clusters, while the SCOUTs are not. In `S2` the scout lattice is x ∈ {8, 16, 24, 32} × y ∈ {10, 20, 30}, and `(8, 10)` is the centre of an infestation cluster, so a scout starts standing on a target and senses it on its first decision — the measured delay is exactly zero for the whole run.

**What has to land before this sweep says anything about resilience:** `TreatSeeder.seeds` returning real target cells (so the swarm moves and `coverage` becomes a function of which robots survive); a communication strategy, so `messages_sent` is non-zero and the failure modes separate; and moving the scouts off the infestation clusters (or an EXPLORE seeder to follow, #41), so detection latency has a distribution rather than a point mass. Until then the honest summary is that the experiment is built and controlled, and its result is `active_robots` and `energy/m²` — which is the correct result for an inert swarm under fault injection, and a useful regression check in its own right.

## Reproducing

```bash
uv run python examples/robustness_demo.py
uv run python examples/robustness_demo.py --scenario S5 --hours 3
uv run python examples/robustness_demo.py --seeds 5
```

Every form runs through `Experiment`, which writes each run's last metrics sample to `results.json` and the table to `tables.md` in a new folder of `output/`.
