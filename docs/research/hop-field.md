# Research note — a hop-based potential field for pest response

Status: describes the hop field implemented in `src/swarm_field/swarm/beacon.py`, `src/swarm_field/swarm/hop_field.py` and `TreatByBeacon` in `src/swarm_field/swarm/behavior.py` (#70). It is the second adapter of the objective-field port in `src/swarm_field/swarm/guidance.py`, next to the seeded Dijkstra field. The results below are produced by `tests/benchmark_pests.py` and are reproducible from the seeds given.

## The question

An explorer that finds a pest has to bring a killer to it. The seeded approach shares the pest's cell: every robot gossips its whole memory, and a killer plans a route to the remembered cell with a Dijkstra potential field. The hop approach shares no cell at all. A robot that sees the pest broadcasts a hop value of 0, a robot in radio range of it broadcasts 1, and so on, and a killer walks down the values to the pest. The question is whether that much smaller message can guide killers as well as the shared cell, and what it costs.

## The model

### Beacons

A **beacon** is a hop value for one objective, broadcast with the cell where it holds (`Beacon` in `contracts/messages.py`). A robot that saw a pest itself within `source_for` beacons 0 on the pest's cell. Every other robot relays: it beacons its own value on its own cell. Beacons travel in their own `BEACON` messages, are kept per sender in memory, and are never gossiped, so a pest's cell never travels further than one hop.

A robot's own value is the lowest fresh beacon it heard plus the hop to it:

$$h_i = \min_{j} \left( h_j + c_{ij} \right), \qquad c_{ij} = \frac{\max\left(\lVert p_j - x_i \rVert,\; d_\text{min}\right)}{L}$$

| Symbol | Meaning |
| --- | --- |
| $h_i$ | The robot's own value at its cell $x_i$. A source has 0. |
| $h_j$, $p_j$ | The value and cell of a beacon $j$ the robot heard within `fresh_for`. |
| $c_{ij}$ | The cost of the hop to beacon $j$: 1 with integer hops, or the distance in radio ranges with metric hops (`hop_cells`). |
| $L$ | `hop_cells`: the radio range in cells. |
| $d_\text{min}$ | `min_hop_cells`: the distance a hop costs at least, one cell. |

Beacons above `max_value` are ignored, and a robot whose own value would exceed it beacons nothing.

### From beacons to a direction

The robot follows the **downhill beacons**: the fresh beacons below its own value and off its own cell. Each pulls the robot towards its cell with a strength that falls with its value and its age. A beacon sent by a killer pushes instead:

$$\mathbf{g}(x_i) = \sum_{j \in D_i \setminus K} s_j \, u_{ij} \;-\; \kappa \sum_{k \in D_i \cap K} s_k \, u_{ik}, \qquad s_j = \gamma^{h_j} e^{-(t - t_j)/\tau_b}, \qquad u_{ij} = \frac{p_j - x_i}{\lVert p_j - x_i \rVert}$$

| Symbol | Meaning |
| --- | --- |
| $\mathbf{g}(x_i)$ | The direction handed to the steerer, as a `Heading`. |
| $D_i$ | The downhill beacons (`downhill_beacons`). |
| $K$ | Beacons sent by killers, known from the senders' status messages. |
| $s_j$ | The strength of beacon $j$ (`beacon_strength`). |
| $\gamma$ | `gamma`, the hop discount. |
| $t - t_j$, $\tau_b$ | The beacon's age and `decay`. |
| $\kappa$ | `killer_repulsion`. |

`HopField` returns the heading towards the strongest attracting beacon's cell, and the existing `Steerer` walks it around obstacles. `TreatByBeacon` treats a pest the robot stands on, and otherwise bids the strongest downhill beacon's strength, so a killer commits only when the field has somewhere to send it. The supervisor's literal rule, "go to the robot with the next lower hop number", is the case of integer hops and $\kappa = 0$.

### Parameters

| Parameter | Value in `hop repel` | Meaning |
| --- | --- | --- |
| `fresh_for` | 1 min | How long a heard beacon counts. |
| `source_for` | 1 min | How long a robot that saw a pest stays its source. |
| `gamma` | 0.5 | Hop discount of a beacon's pull. |
| `decay` | 1 min | Decay time of a beacon's pull with age. |
| `hop_cells` | 2 cells | Metric hops: one radio range of 20 m on 10 m cells. |
| `min_hop_cells` | 1 cell | The least a hop costs. |
| `max_value` | 4.3 | The cap, where a beacon's strength falls below 0.05. |
| `killer_repulsion` | 2 | How hard a killer's beacon pushes. |

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
