# Research note — a hop-based potential field for pest response

Status: describes the hop field implemented in `src/swarm_field/swarm/beacon.py`, `src/swarm_field/swarm/hop_field.py` and `TreatByBeacon` in `src/swarm_field/swarm/behavior.py` (#70). It is the second adapter of the objective-field port in `src/swarm_field/swarm/guidance.py`, next to the seeded Dijkstra field. The results below are produced by `tests/benchmark_pests.py` and are reproducible from the seeds given.

## The question

An explorer that finds a pest has to bring a killer to it. The seeded approach shares the pest's cell: every robot gossips its whole memory, and a killer plans a route to the remembered cell with a Dijkstra potential field. The hop approach shares no cell at all. A robot that sees the pest broadcasts a hop value of 0, a robot in radio range of it broadcasts 1, and so on, and a killer walks down the values to the pest. The question is whether that much smaller message can guide killers as well as the shared cell, and what it costs.

## The model

In one sentence: a robot walks in the weighted average direction of the nearby beacons that are closer to the pest than itself, where a beacon weighs more the fewer hops it is from the pest and the more recently it was heard.

### Beacons

A **beacon** is a message with three things in it: a hop number, a cell, and the time it was sent (`Beacon` in `contracts/messages.py`). Every robot broadcasts one beacon per decision.

- A robot that saw a pest itself within `source_for` broadcasts hop number 0 on the pest's cell. It is the **source**.
- Every other robot is a **relay**. It broadcasts its own hop number on its own cell.

A robot remembers the newest beacon from each sender. A remembered beacon counts as long as it was sent within `fresh_for` and its hop number is at most `max_value`. These are the robot's **heard beacons**.

### A robot's own hop number

A robot's own hop number is the smallest "hop number of a heard beacon plus the cost of the hop to it":

$$\text{OwnHop}(\text{robot}) = \min_{\text{beacon} \,\in\, \text{Heard}} \Big( \text{Hop}(\text{beacon}) + \text{HopCost}(\text{robot}, \text{beacon}) \Big)$$

$$\text{HopCost}(\text{robot}, \text{beacon}) = \frac{\max\big(\text{Distance}(\text{robot}, \text{beacon}),\ \text{MinHopDistance}\big)}{\text{RadioRange}}$$

The cost of a hop is the distance to the beacon's cell in radio ranges, so a beacon at the edge of the radio range costs about 1 hop and one halfway costs about 0.5. A hop always costs at least `MinHopDistance` (one cell), so two robots on the same cell cannot hand each other the same value forever. With integer hops (no `hop_cells`) every hop costs exactly 1.

A relay broadcasts this own hop number, unless it is above `max_value`.

### The direction a robot walks in

Only the heard beacons that are closer to the pest than the robot itself pull it. These are the **downhill beacons**: their hop number is below the robot's own, and they are not on the robot's own cell.

Each downhill beacon pulls the robot towards its cell with a strength:

$$\text{Strength}(\text{beacon}) = \text{PullPerHop}^{\,\text{Hop}(\text{beacon})} \cdot e^{-\text{Age}(\text{beacon}) / \text{DecayTime}}$$

A beacon at hop 0 pulls with strength 1, a beacon at hop 1 with 0.5, at hop 2 with 0.25, and so on (at `PullPerHop` = 0.5). A beacon heard one `DecayTime` ago pulls with $1/e \approx 0.37$ of that.

The direction is the strength-weighted average of the directions to the downhill beacons:

$$\text{Pull}(\text{robot}) = \sum_{\text{beacon} \,\in\, \text{Downhill}} \text{Sign}(\text{beacon}) \cdot \text{Strength}(\text{beacon}) \cdot \text{UnitVector}(\text{robot} \to \text{beacon})$$

$$\text{Direction}(\text{robot}) = \frac{\text{Pull}(\text{robot})}{\lVert \text{Pull}(\text{robot}) \rVert}$$

$\text{Sign}$ is $+1$ for a beacon sent by a relay or a source. For a beacon sent by another killer it is $-\text{KillerRepulsion}$: a killer closer to the pest pushes instead of pulls, so two killers do not walk to the same pest. Dividing by the length keeps only the direction, which is why a sum and an average point the same way.

`HopField` hands the direction to the `Steerer` as a `Heading` towards the strongest attracting beacon's cell, and the steerer walks around obstacles on the way. `TreatByBeacon` bids to treat with the strength of that same strongest beacon, so a killer commits only when something pulls it.

**Worked example.** A killer with own hop number 2.0 hears three beacons: a relay at hop 1.2 to the east, a relay at hop 1.5 to the north, and a relay at hop 2.4 to the west. The west one is not downhill. The east one pulls with $0.5^{1.2} = 0.44$, the north one with $0.5^{1.5} = 0.35$. The killer walks north-east, a little more towards the east than the north.

### Obstacle beacons (optional)

With `obstacle_repulsion` above 0, a robot whose direction runs it into a cell it cannot enter places an **obstacle beacon** on that cell. Obstacle beacons stay with the robot that placed them; they are never broadcast. Each pushes the robot away from its cell, within `obstacle_range_cells`:

$$\text{Push}(\text{obstacle}) = \text{ObstacleRepulsion} \cdot e^{-\text{Age}(\text{obstacle}) / \text{ObstacleDecayTime}} \cdot \Big(1 - \frac{\text{Distance}(\text{robot}, \text{obstacle})}{\text{ObstacleRange}}\Big)$$

The pushes are added to $\text{Pull}(\text{robot})$, each along $\text{UnitVector}(\text{obstacle} \to \text{robot})$, before the direction is taken.

### Escaping and walkable links (optional)

Two more local rules help a killer past obstacles; both are off by default.

- **Escape** (`BeaconConfig.escape`). A killer whose own hop number has not come down by `progress_hops` within `stuck_window` places an obstacle beacon on its own cell. Each further time it is stuck before making progress, the next one pushes `escape_boost` times harder.
- **Walkable links** (`walkable_links`). A robot ignores a beacon whose straight line from its own cell crosses a known obstacle, so hop numbers spread only along ground a killer can walk. Without it, radio crosses a river and the hop numbers on the near bank point straight across the water.

### Names in the formulas and in the code

| Name in the formulas | `BeaconConfig` field | Value in `hop repel` |
| --- | --- | --- |
| `fresh_for` | `fresh_for` | 1 min |
| `source_for` | `source_for` | 1 min |
| `max_value` | `max_value` | 4.3 hops |
| RadioRange | `hop_cells` | 2 cells (20 m on 10 m cells) |
| MinHopDistance | `min_hop_cells` | 1 cell |
| PullPerHop | `gamma` | 0.5 |
| DecayTime | `decay` | 1 min |
| KillerRepulsion | `killer_repulsion` | 2 |
| ObstacleRepulsion | `obstacle_repulsion` | 0 (off) |
| ObstacleRange | `obstacle_range_cells` | 3 cells |
| ObstacleDecayTime | `obstacle_decay` | 10 min |

The supervisor's literal rule, "go to the robot with the next lower hop number", is the case of integer hops and no killer repulsion: only beacons at the lowest hop number are downhill, and they pull equally apart from their age.

## The scenario

`RELAYED_KILLERS` in `src/swarm_field/experiments/pests.py`: a 32 × 32 field of 10 m cells without obstacles, 400 stationary relay scouts on a jittered lattice, 8 killers on a lattice of their own, and a 20 m radio. A mock source on every infested cell (`PestSourceBroadcast`) stands in for the explorer that found the pest. It sends the pest's infestation fact for the seeded arm and a value-0 beacon for the hop arms. Relays and killers are kept apart because in a killers-only swarm the nearest killer is always one hop from a pest, so the chain would never be used. With 400 relays the chain from a pest to the nearest killer has a median depth of three hops.

The main outcome is `infested`: the infested cells averaged over the run after a one-hour warm-up. It counts both how many pests stay and how long. The number of treated pests is a poor outcome here, because pests from the last twenty minutes of a run are still open whatever the swarm does.

## Results

20 seeds per arm. "Paired" is the difference in infested cells to the capped hop arm on the same seeds, with its standard error.

| Arm | Infested, 15 pests/h | Paired | Infested, 30 pests/h | Paired | Cells moved per treatment, 15/h |
| --- | --- | --- | --- | --- | --- |
| seeded | 8.04 ± 0.67 | −1.07 ± 0.64 | 15.79 ± 1.09 | −1.64 ± 1.00 | 19.6 |
| hop metric, cap | 9.12 ± 0.52 | 0 | 17.42 ± 0.82 | 0 | 12.1 |
| **hop repel** (cap + repulsion 2) | **8.03 ± 0.52** | **−1.09 ± 0.29** | **16.21 ± 0.77** | **−1.21 ± 0.44** | **11.2** |
| hop metric, no cap | 9.95 ± 0.87 | +0.83 ± 0.92 | 18.48 ± 0.84 | +1.05 ± 0.64 | 22.6 |
| no cap + repulsion 1 | 8.09 ± 0.52 | −1.02 ± 0.60 | 17.01 ± 0.70 | −0.42 ± 0.72 | 19.4 |

The 15/h runs last 3 hours. The 30/h runs last 2 hours, so that no killer runs out of treatment consumable.

- **The hop field matches the shared cell.** `hop repel` leaves as many infested cells as the seeded arm at both loads. It moves its killers 40 % less per treatment, its robots receive 25 % fewer messages, and a run simulates about eight times faster, because relays send one beacon instead of their memory.
- **Limiting recruitment is what makes it good.** Without the cap and without repulsion, every killer that hears a pest walks towards it. The cap limits how far a pest recruits, and repulsion stops a killer from following one that is closer. Each cuts the infested cells by about a tenth, and they add up.

## What went wrong on the way

**Integer hops jitter.** With integer hops a robot's own value is one more than the lowest it hears, so only beacons at that one value pull, and that set changes with every cell moved. Killers reversed 10 % of their moves against 2.5 % in the seeded arm.

**Two robots on one cell hold a value forever.** With metric hops and no minimum hop, a killer standing on a relay's cell takes the relay's value at no cost and broadcasts it, and the relay hears it back at no cost. The pair keeps a value that never rises, long after its pest was cleared, and killers sat motionless on such cells 83 % of the time. The minimum hop of one cell breaks the loop. Expiring beacons by the source's send time also fixed it; an inhibitor wave sent by the treating killer helped but did not fix it.

**Cleared pests leave ghosts.** When a pest is cleared its source falls silent, but relays keep feeding each other values that rise one hop per round and never vanish (count-to-infinity). The cap removes them once they rise past it. While other pests exist, their live gradients win the minimum, so ghosts cost about two cells of travel per treatment.

## What these results do and do not show

- Relays stand still. With moving explorers as relays, chains break and re-form, and stale beacons matter more.
- The field has no obstacles. The steerer walks the heading around obstacles, but how often a direction into a wall strands a killer is not measured.
- Every robot knows its own absolute cell, and beacons carry absolute cells. The hop approach needs only relative positions within radio range, so it may be more robust to position error than a shared cell, but that is not tested.
- Exploration still gossips per-cell facts in the seeded arm. A full comparison needs exploration with coarse sharing (#71).

## Background

- Abelson, Nagpal et al. (2000), *Amorphous Computing*: the hop-count gradient.
- Payton et al. (2001), *Pheromone Robotics*: decaying virtual pheromones relayed robot to robot.
- Ducatelle et al. (2014), *Cooperative navigation in robotic swarms*: walk to where the best neighbour was heard, and age values honestly.
- Viroli, Beal et al. (2018), *Engineering Resilient Collective Adaptive Systems by Self-Stabilisation*: the rising-value problem of gradients and its fixes.
- McLurkin & Smith (2004), *Distributed dispersion*: relays should not follow the gradient they carry.

## Reproducing

```bash
uv run python -m tests.benchmark_pests --hours 3 --seeds 20 --pests-per-hour 15
```

```bash
uv run python -m tests.benchmark_pests --hours 2 --seeds 20 --pests-per-hour 30
```
