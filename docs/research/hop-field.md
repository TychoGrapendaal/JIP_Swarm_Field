# Research note — a hop-based potential field for pest response

Status: describes the hop method implemented in the package `src/swarm_field/swarm/hop/` (#70): relaying in `relay.py`, the pull in `pull.py`, `HopField` in `field.py`, `TreatByBeacon` in `behavior.py`, the escape in `escape.py`, the beacons a robot remembers in `memory.py`, and `BeaconLink`, which connects them to the brain, in `link.py`. It is the second adapter of the objective-field port in `src/swarm_field/swarm/guidance.py`, next to the seeded Dijkstra field. The results below are produced by `studies/benchmark_pests.py` and are reproducible from the seeds given.

## The question

An explorer that finds a pest has to bring a killer to it. The seeded approach shares the pest's cell: every robot gossips its whole memory, and a killer plans a route to the remembered cell with a Dijkstra potential field. The hop approach shares no cell at all. A robot that sees the pest broadcasts a hop value of 0, a robot in radio range of it broadcasts 1, and so on, and a killer walks down the values to the pest. The question is whether that much smaller message can guide killers as well as the shared cell, and what it costs.

## The model

In one sentence: a robot walks in the weighted average direction of the nearby beacons that are closer to the pest than itself, where a beacon weighs more the fewer hops it is from the pest and the more recently it was heard.

### Beacons

A **beacon** is a message with three things in it: a hop number, a cell, and the time it was sent (`Beacon` in `contracts/messages.py`). Every robot broadcasts one beacon per decision.

- A robot that saw a pest itself within `source_for` broadcasts hop number 0 on the pest's cell. It is the **source**.
- Every other robot is a **relay**. It broadcasts its own hop number on its own cell.

A robot remembers the newest beacon from each sender. A remembered beacon counts as long as it was sent within `fresh_for` and its hop number is at most `max_value`. These are the robot's **heard beacons**. It keeps them in a `BeaconMemory` of its own, together with the pests it saw itself and the beacons it placed, and forgets each when it no longer counts: a heard beacon after `fresh_for`, a sighting after `source_for`, and a placed beacon once it has faded below 1 % of its push. A robot that sees the pest gone from a cell stops being its source at once. The robot's memory holds no map and no other robot's pest cells.

### A robot's own hop number

A robot's own hop number is the smallest "hop number of a heard beacon plus the cost of the hop to it":

$$\text{OwnHop}(\text{robot}) = \min_{\text{beacon} \,\in\, \text{Heard}} \Big( \text{Hop}(\text{beacon}) + \text{HopCost}(\text{robot}, \text{beacon}) \Big)$$

$$\text{HopCost}(\text{robot}, \text{beacon}) = \frac{\max\big(\text{Distance}(\text{robot}, \text{beacon}),\ \text{MinHopDistance}\big)}{\text{RadioRange}}$$

The cost of a hop is the distance to the beacon's cell in radio ranges, so a beacon at the edge of the radio range costs about 1 hop and one halfway costs about 0.5. A hop always costs at least `MinHopDistance` (one cell), so two robots on the same cell cannot hand each other the same value forever. With integer hops (no `hop_cells`) every hop costs exactly 1.

A relay broadcasts this own hop number, unless it is above `max_value`.

### The direction a robot walks in

Only the heard beacons that are closer to the pest than the robot itself pull it. These are the **downhill beacons**: their hop number is below the robot's own, and they are not on the robot's own cell. A robot standing on a beacon's cell compares with that beacon's hop number instead, if it is lower, since it is as close to the pest as that beacon. Its own hop number adds at least one minimum hop, so without this a neighbour just above the beacon would pull the robot back, and it would shuttle between the two cells.

Each downhill beacon pulls the robot towards its cell with a strength:

$$\text{Strength}(\text{beacon}) = \text{PullPerHop}^{\,\text{Hop}(\text{beacon})} \cdot e^{-\text{Age}(\text{beacon}) / \text{DecayTime}}$$

A beacon at hop 0 pulls with strength 1, a beacon at hop 1 with 0.5, at hop 2 with 0.25, and so on (at `PullPerHop` = 0.5). A beacon heard one `DecayTime` ago pulls with $1/e \approx 0.37$ of that.

The direction is the strength-weighted average of the directions to the downhill beacons:

$$\text{Pull}(\text{robot}) = \sum_{\text{beacon} \,\in\, \text{Downhill}} \text{Sign}(\text{beacon}) \cdot \frac{\text{Strength}(\text{beacon})}{\max\big(\text{Distance}(\text{robot}, \text{beacon}),\ 1\big)^{\text{DistancePower}}} \cdot \text{UnitVector}(\text{robot} \to \text{beacon})$$

$$\text{Direction}(\text{robot}) = \frac{\text{Pull}(\text{robot})}{\lVert \text{Pull}(\text{robot}) \rVert}$$

With `DistancePower` = 0 every beacon pulls equally hard from any distance. A robot between several pests is then held where the unit pulls cancel, their geometric median: the pulls are the gradient of $\sum s_j \lVert x - p_j \rVert$, which is convex, so that point is a stable minimum, and a push only moves the robot off it for a while. With any power above 0 the point between two pests becomes unstable along the line joining them. At power 1 the pulls are the gradient of $\sum s_j \ln \lVert x - p_j \rVert$, which is harmonic in the plane, so there is no minimum anywhere away from the beacons. The distance is at least one cell, so a beacon next to the robot does not dominate without bound. The weighted strength also picks the strongest beacon.

$\text{Sign}$ is $+1$ for a beacon sent by a relay or a source. For a beacon sent by another killer it is $-\text{KillerRepulsion}$: a killer closer to the pest pushes instead of pulls, so two killers do not walk to the same pest. Dividing by the length keeps only the direction, which is why a sum and an average point the same way.

`HopField` hands the direction to the `Steerer` as a `Heading` towards the strongest attracting beacon's cell, and the steerer walks around obstacles on the way. It follows a wall to its end, and is reset only when it stalls, however often the strongest beacon changes on the way. A hop killer keeps no map: the steerer uses only the terrain the robot senses around itself at the current decision (`SensedSurroundings`). With `BrainSpec.keeps_map` off, as in every hop arm, its memory stores no terrain or pest facts at all and it has no quadtree. `TreatByBeacon` bids to treat with the strength of that same strongest beacon, so a killer commits only when something pulls it.

**Worked example.** A killer with own hop number 2.0 hears three beacons: a relay at hop 1.2 to the east, a relay at hop 1.5 to the north, and a relay at hop 2.4 to the west. The west one is not downhill. The east one pulls with $0.5^{1.2} = 0.44$, the north one with $0.5^{1.5} = 0.35$. The killer walks north-east, a little more towards the east than the north.

### Escaping when stuck (optional)

A killer can be caught where pulls cancel: between pests that pull it equally hard, or where the pull runs into a wall. It then shuttles between two cells or paces in place, and on the way it may lose its heading for a few decisions, for example when it steps out of range of the beacon that pulled it. With `BeaconConfig.escape`, the killer compares how far it got over the last `window` with how far its top speed would have taken it:

$$\text{Progress} = \frac{\lVert c_{\text{now}} - c_{\text{first}} \rVert \cdot \text{CellSize}}{\text{TopSpeed} \cdot (t_{\text{now}} - t_{\text{first}})}$$

| Symbol | Meaning |
| --- | --- |
| $c_{\text{now}}$ | the killer's cell at the current decision |
| $c_{\text{first}}, t_{\text{first}}$ | its cell and the time at the oldest decision of the window |
| CellSize | the side of a cell in metres (10 m) |
| TopSpeed | the killer's top speed, from its own state (0.5 m/s for a ground killer by default, 0.1 m/s in the results below) |

Progress is near 1 for a straight walk at full speed, near 0 for a shuttle, and 0 for standing still. The window is a time, 10 minutes by default, in which a ground killer can walk 30 cells at 0.5 m/s, or 6 at 0.1 m/s. It must span several moves, because a moving killer reports the cell it left until it arrives, and one cell takes it 20 s at 0.5 m/s, or 100 s at 0.1 m/s. The killer judges itself only at a decision on a heading, and only once its record spans the whole window, but it records its cell at every decision: a killer that loses its heading while it circles is still circling. A treatment starts the record over, since the killer stands still on purpose. A heading after more than a window without one also starts it over, so a killer that waited for work is not stuck at its first step.

When Progress is below `min_progress` (0.3), the killer places an **escape beacon** on the centre of the cells it was in, the place it circles around. Its push points from that centre to the killer, so it always says "leave this place". If the centre is the killer's own cell, as for a standstill, the beacon goes on the cell ahead along the heading instead, which is the wall when there is one. The beacon pushes `escape_strength` and fades with `escape_decay`. If the killer is stuck again within one cell of its last escape beacon while that beacon is younger than its decay time, the beacon is placed again on its own cell, with `escape_boost` times its strength and `escape_boost` times its decay time:

$$\text{EscapePush}_{k+1} = \text{EscapeBoost} \cdot \text{EscapePush}_k, \qquad \text{EscapeDecayTime}_{k+1} = \text{EscapeBoost} \cdot \text{EscapeDecayTime}_k$$

So while the killer keeps circling in the same area, the beacon grows and does not fade. Elsewhere, or once the last one is older than its decay time, the escape starts over. A beacon never replaces a stronger push on its cell.

Escape beacons stay with the robot that placed them (`PlacedBeacon` in `BeaconMemory`); they are never broadcast. Each pushes the robot away from its cell, within `obstacle_range_cells`:

$$\text{Push}(\text{beacon}) = \text{EscapePush} \cdot e^{-\text{Age}(\text{beacon}) / \text{EscapeDecayTime}} \cdot \Big(1 - \frac{\text{Distance}(\text{robot}, \text{beacon})}{\text{ObstacleRange}}\Big)$$

The pushes are added to $\text{Pull}(\text{robot})$, each along $\text{UnitVector}(\text{beacon} \to \text{robot})$, before the direction is taken.

The escape cannot free a killer held at the geometric median of several pests by unit pulls: every push only moves it off a stable point for a while. A charge on a dock starts the escape's record over too, like a treatment, so that a killer does not place an escape beacon on its own dock.

### Finding a dock

A hop killer has no map, so it cannot plan to a dock. Docks are infrastructure with a radio of their own: every decision interval, every dock broadcasts a DOCK beacon of 0 on its cell (`DockBroadcast`), as far as `MapConfig.dock_range_m`, by default the robots' radio range. Every robot relays its own DOCK value next to its TREAT value, in the same message, by the same rule as for TREAT beacons, and a killer that must charge or restock walks down the DOCK gradient on `HopDockField` (`ReturnToDock` and `Resupply`). It uses the configured dock cells only to tell whether it stands on one.

A dock never moves or vanishes, so a DOCK beacon heard from a cell stays a true upper bound on the hops to a dock for the whole run. DOCK beacons therefore have settings of their own (`BeaconConfig.dock`, a `DockBeacons`): they stay fresh and decay over 30 minutes instead of one, they have no cap, since a dock leaves no ghosts, their discount per hop is 0.8 instead of 0.5, since a dock may lie many hops away, and killers do not repel each other on the way. The gradient follows the radio distance, not the walking distance, so the steerer and the escape have to find a bridge over a river.

**It finds a dock, but slowly.** `studies/hop_dock_return.py` runs the moving-explorer scenario of `explorer_hop_killers.py` for 6 h, 8 seeds, with the dock at `(0, 0)` west of the river. With 10 doses per killer and unlimited battery, every killer has to restock:

| Arm | Infested | Restock trips started | Completed | Trip time (min) | Longest (min) | Still under way at the end |
| --- | --- | --- | --- | --- | --- | --- |
| seeded | 1.32 ± 0.06 | 6.1 | 5.8 | 11.6 ± 0.7 | 19 | 0.4 |
| hop | 9.01 ± 0.86 | 4.6 | 2.4 | 58.6 ± 11.4 | 166 | 2.3 |
| seeded, a dock on each bank | 1.29 ± 0.08 | 5.9 | 5.6 | 6.0 ± 0.4 | 10 | 0.3 |
| hop, a dock on each bank | 6.32 ± 0.43 | 4.6 | 3.9 | 38.0 ± 5.4 | 133 | 0.8 |

The beacons and the arbitration work: while a hop killer is low on doses and off the dock, `resupply` wins 98 % of its decisions and a DOCK beacon pulls it 93 % of the time. But it spends 64 % of that time east of the river, and a quarter of it pulled by a beacon on the far bank, straight across the water, which the flying explorers relayed. A dock on each bank shortens the trips, but they still take six times as long as a seeded killer's. The TREAT gradient crosses the river in the same way, which is a likely reason why hop killers walk 56 cells per treatment in these runs and seeded killers 20. With half a battery and unlimited doses, a hop killer died of an empty battery in 2 of 8 runs, while seeded killers never even needed a dock.

### What a robot keeps

A hop robot keeps no list that grows with the swarm or the field. `BeaconMemory` holds at most 100 entries of each sort (heard beacons of one kind, placed escape beacons, own pest sightings, and the killers it heard), and forgets the oldest first beyond that, besides forgetting each entry when it no longer counts. A killer that keeps no map skips the memory snapshots its peers broadcast and keeps at most 100 peers it heard or sensed itself.

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
| DistancePower | `distance_power` | 0 (0.5 in `hop tuned`) |
| ObstacleRange | `obstacle_range_cells` | 3 cells |
| `window` | `escape.window` | 10 min (escape off in `hop repel`) |
| `min_progress` | `escape.min_progress` | 0.3 (escape off in `hop repel`) |
| EscapeStrength | `escape.escape_strength` | 1 (escape off in `hop repel`) |
| EscapeDecayTime | `escape.escape_decay` | 10 min (escape off in `hop repel`) |
| EscapeBoost | `escape.escape_boost` | 2 (escape off in `hop repel`) |

The supervisor's literal rule, "go to the robot with the next lower hop number", is the case of integer hops and no killer repulsion: only beacons at the lowest hop number are downhill, and they pull equally apart from their age.

## The scenario

`RELAYED_KILLERS` in `src/swarm_field/experiments/scenario.py`: a 32 × 32 field of 10 m cells without obstacles, 400 stationary relay scouts on a jittered lattice, 8 killers on a lattice of their own, and a 20 m radio. A mock source on every infested cell (`PestSourceBroadcast`) stands in for the explorer that found the pest. It sends the pest's infestation fact for the seeded arm and a value-0 beacon for the hop arms. Since the claims were added (see "The seeded arm with claims" below), the seeded arm's killers leave a pest to a closer killer that claimed it, and its relays pass on those claims; the arm without them is `seeded no yield`. Relays and killers are kept apart because in a killers-only swarm the nearest killer is always one hop from a pest, so the chain would never be used. Killers drove at 0.1 m/s in every result on this page; the default is now 0.5 m/s. A killer clears a pest of full severity in one minute (`FAST_TREATMENT`): the results below were measured when treatment cleared a pest at once, and the simulation's default rate of one severity per hour is slower than these pests grow. With 400 relays the chain from a pest to the nearest killer has a median depth of three hops.

The main outcome is `infested`: the infested cells averaged over the run after a one-hour warm-up. It counts both how many pests stay and how long. The number of treated pests is a poor outcome here, because pests from the last twenty minutes of a run are still open whatever the swarm does.

## Results

### Hop tuned against the shared cell

20 seeds per arm, on the same seeds. "Paired" is the difference in infested cells to the seeded arm, with its standard error. The seeded arm here is the one without claims, now `seeded no yield`. The 15/h runs last 3 hours, the 30/h runs 2 hours, so that no killer runs out of treatment consumable.

| Arm | Infested, 15 pests/h | Paired | Infested, 30 pests/h | Paired | Cells moved per treatment, 15/h | Messages per robot and hour, 15/h |
| --- | --- | --- | --- | --- | --- | --- |
| seeded | 8.04 ± 0.67 | 0 | 15.79 ± 1.09 | 0 | 19.6 | 1923 |
| `hop repel` | 7.08 ± 0.40 | −0.96 ± 0.60 | 15.24 ± 1.14 | −0.54 ± 1.12 | 9.6 | 1392 |
| **`hop tuned`** | **6.60 ± 0.61** | **−1.44 ± 0.68** | **14.19 ± 0.90** | **−1.59 ± 1.01** | **9.2** | **1382** |

- **The hop field beats the shared cell without claims.** `hop tuned` leaves fewer infested cells than the seeded arm at both loads, about 2 standard errors at 15/h and 1.6 at 30/h. Its killers move half as far per treatment, its robots receive 28 % fewer messages, and a run simulates about six times faster, because relays send one beacon instead of their memory.
- **The distance weight and the escape add to `hop repel`.** Paired against `hop repel`, `hop tuned` leaves 0.48 ± 0.46 fewer infested cells at 15/h and 1.05 ± 0.52 fewer at 30/h, where more pests stand close together.

These runs predate the `BeaconMemory` that forgets stale beacons and keeps the robot's own sightings apart from its peers' reports. Rerun at 15/h, 78 of the 80 hop runs give identical results with it, and the other two differ by 0.02 infested cells.

They also predate the escape's time window and speed-aware progress (see the escape section). Rerun at 15/h on the same 20 seeds with it, `hop tuned` leaves 6.57 ± 0.51 infested cells against 6.60 ± 0.61, and the other arms are unchanged. On the two obstacle fields of `studies/benchmark_obstacles.py`, 8 paired seeds each, it leaves +0.50 ± 0.93 infested cells on the small field and +1.38 ± 1.24 on the large one, both within their standard error.

### The seeded arm with claims

The results above compare the hop method with a seeded arm whose killers do not coordinate: its relays gossip only pest facts, and the TREAT seeder discounts a pest a peer heads for to 20 % of an attraction worth at most one cell of travel, which hardly changes a killer's choice. In addition, a killer's sighting of a nearby peer overwrote that peer's status in memory, so the targets of nearby killers were lost. On the obstacle fields this made several seeded killers treat the same pest at once: on the small field 141 of about 230 treatment jobs overlapped another killer's on the same pest, against 22 of 178 for `hop tuned`, and since a dose is charged per treatment job, seeded killers ran out of consumable.

Three changes make the seeded arm coordinate. Memory keeps a peer's target and objective from its last status when a newer sighting arrives. With `SeededKillerPreset.claims`, the TREAT seeder leaves out a pest that a closer killer claimed within the last two minutes (`TreatSeeder.yield_to_closer`), and a killer standing on a pest that a killer with a lower id already treats waits (`TreatInfestation.yield_to_treating_peer`). With `KnowledgeSharing.INFESTATIONS_AND_TARGETS`, relays pass on the peers that report a target, the killers, so claims travel beyond one radio range; passing on every peer instead, about 270 per message, cost 36 % more run time for the same effect. The seeded arm of `pest_response_configurations` now has all three.

12 paired seeds, 3 h, on the obstacle fields of `studies/benchmark_obstacles.py` (6 killers at 18 pests/h on 32 × 32, 18 killers at 48 pests/h on 64 × 64):

| Arm | Infested, small | Paired to no yield | Infested, large | Paired to no yield | Doses per treated pest, large |
| --- | --- | --- | --- | --- | --- |
| `seeded no yield` | 13.53 ± 1.43 | 0 | 33.16 ± 1.85 | 0 | 1.44 |
| `seeded` (claims) | 9.17 ± 1.19 | −4.36 ± 1.18 | 21.01 ± 1.17 | −12.15 ± 1.51 | 1.10 |
| `hop tuned` | 11.32 ± 0.76 | −2.22 ± 0.95 | 31.72 ± 1.31 | −1.44 ± 1.56 | 1.03 |

With claims, the seeded arm beats `hop tuned` on both obstacle fields, by about 2 infested cells on the small one and 11 on the large one. The open-field results above were measured with the seeded arm without claims and have not been rerun, so the comparison of the two methods there is open.

### Shuttles, the distance weight and the escape

12 seeds per arm, 3 h at 15 pests/h, all with the relay-shuttle fix. "Paired" is the difference to `hop repel` on the same seeds. "River" is the explainer's scenario instead: one killer, 100 relays, a 50 m radio, the river and the rocks, 12 seeds of 2 h at 6 pests/h, and its outcome is the pests treated by the killer in total.

| Arm | Infested | Paired | Cells moved per treatment | River, treated |
| --- | --- | --- | --- | --- |
| `hop repel` | 7.18 ± 0.55 | 0 | 9.7 | 35 |
| + escape | 7.36 ± 0.57 | +0.19 ± 0.22 | 9.9 | 34 |
| DistancePower 1 | 7.72 ± 0.79 | +0.54 ± 0.49 | 9.8 | 40 |
| DistancePower 1 + escape | 7.68 ± 0.84 | +0.50 ± 0.50 | 9.9 | 45 |
| DistancePower 0.5 | 6.63 ± 0.65 | −0.55 ± 0.42 | 9.4 | 42 |
| **`hop tuned`** (DistancePower 0.5 + escape) | **6.17 ± 0.53** | **−1.01 ± 0.50** | **9.2** | **42** |

- **The relay-shuttle fix matters most.** Before it, killers on a heading made less than 0.3 progress in 39 % of their ten-minute windows, nearly all of them two-cell shuttles. After it, 6 %. `hop repel` went from 8.04 to 7.18 infested cells on the same seeds, and its killers move 15 % less per treatment.
- **A distance weight frees killers between pests.** On the river, power 1 lifts the treated pests from 35 to 40, and to 45 with the escape. The two runs where the killer was held between pests went from 1 and 0 treated to 4 and 6. In the open field, power 1 is slightly worse; power 0.5 is better in both.
- **The escape adds little once the shuttles are gone.** It helps together with a distance weight, where a push can tip the killer off an unstable point.

### Moving explorers: how long a pest stays marked, and hovering (tried, dropped)

These runs use the scenario of `studies/explorer_hop_killers.py` instead: the river field, 20 moving explorers and 4 killers, no relays or mock sources, a 50 m radio, 12 pests per hour, 3 h with a 1 h warm-up, and unlimited battery and doses. Robots decide whenever a move arrives. The 16 seeds (100 to 115) were not used in the first, 8-seed run of the same arms, so they confirm it independently. "Paired" is the difference in infested cells to the arm's baseline on the same seeds, and "idle" is the change in the field's mean idleness, the cost to exploring.

An explorer that senses a pest is its source for `source_for`, 1 minute in `hop-tuned`. As it flies on, it keeps beaconing 0 on the pest's cell for that long, so it carries the pest's location away from the pest. We compared a longer `source_for` with a scout that instead hovers over the pest it found: it stands still next to it with a bid that falls over a time limit, and stops when the pest is cleared, a killer arrives, or a scout with a lower id already hovers there.

| Arm | Infested | Paired | Idle | Killer hears a 0-beacon | Its distance (cells) |
| --- | --- | --- | --- | --- | --- |
| hop, `source_for` 1 min (baseline) | 4.92 ± 0.42 | 0 | 0 | 19.5 % | 4.3 |
| hop, `source_for` 3 min | 3.86 ± 0.40 | −1.06 ± 0.35 | +1 % | 33.9 % | 6.5 |
| hop, hover 3 min | 4.42 ± 0.38 | −0.51 ± 0.48 | +13 % | 20.1 % | 3.6 |
| hop, hover 3 min and `source_for` 3 min | 3.47 ± 0.33 | −0.39 ± 0.42 against `source_for` 3 min | +11 % | 30.0 % | 5.5 |
| seeded (baseline) | 1.23 ± 0.09 | 0 | 0 | — | — |
| seeded, hover 3 min | 1.31 ± 0.09 | +0.08 ± 0.06 | +4 % | — | — |
| hop, 10 explorers, `source_for` 1 min (baseline) | 7.39 ± 0.49 | 0 | 0 | 15.0 % | 4.6 |
| hop, 10 explorers, `source_for` 3 min | 5.28 ± 0.36 | −2.10 ± 0.48 | −1 % | 29.8 % | 7.7 |
| hop, 10 explorers, hover 3 min | 6.02 ± 0.55 | −1.36 ± 0.54 | +25 % | 16.6 % | 3.7 |

- **Keeping the pest marked longer helps the hop arm.** A `source_for` of 3 minutes removes about a fifth of the infested cells with 20 explorers and better than a quarter with 10, on 13 of 16 seeds each, at no cost to exploring. A 10-minute `source_for` did a little worse than 3 minutes in the first run (−1.48 against −1.79), probably because a long mark also points killers to pests that were cleared meanwhile.
- **Hovering did not improve the response much, and it was dropped.** With 20 explorers its gain is within the error, on top of the 3-minute mark it adds nothing, and in the seeded arm it is slightly worse, since the pest fact is gossiped anyway. With 10 explorers it helps, but less than the 3-minute mark, and the field's idleness rises by a quarter. The code was not kept.
- **Why: a marked pest is useless when no killer can hear the mark.** A hovering scout was connected by radio to some killer in only 32 % of its hovering time (22 % with 10 explorers), so most of the time its beacon reached no killer at all. A scout that flies on carries the mark towards the rest of the swarm instead. With a 3-minute mark a killer heard a 0-beacon in 34 % of its samples, against 20 % with hovering, and heard it from 6.5 cells away from the pest instead of 3.6. Hovering keeps the mark where it is least needed.
- **Hovering worked as designed.** The hovering scout was the pest's 0-source in 99 % of its hovering samples, and almost every cleared pest was hovered over at some point. A hover lasted 2.75 minutes at the median, of 3 allowed. 44 % of hovers ran out of time and 33 % gave way to a lower-id scout on the same pest, while only 12 % ended because a killer arrived.

### Earlier: limiting recruitment

20 seeds per arm, run before the fix for the relay shuttle, which changes every hop arm. "Paired" is the difference in infested cells to the capped hop arm on the same seeds.

| Arm | Infested, 15 pests/h | Paired | Infested, 30 pests/h | Paired | Cells moved per treatment, 15/h |
| --- | --- | --- | --- | --- | --- |
| seeded | 8.04 ± 0.67 | −1.07 ± 0.64 | 15.79 ± 1.09 | −1.64 ± 1.00 | 19.6 |
| hop metric, cap | 9.12 ± 0.52 | 0 | 17.42 ± 0.82 | 0 | 12.1 |
| **hop repel** (cap + repulsion 2) | **8.03 ± 0.52** | **−1.09 ± 0.29** | **16.21 ± 0.77** | **−1.21 ± 0.44** | **11.2** |
| hop metric, no cap | 9.95 ± 0.87 | +0.83 ± 0.92 | 18.48 ± 0.84 | +1.05 ± 0.64 | 22.6 |
| no cap + repulsion 1 | 8.09 ± 0.52 | −1.02 ± 0.60 | 17.01 ± 0.70 | −0.42 ± 0.72 | 19.4 |

- **Then, the hop field matched the shared cell.** `hop repel` left as many infested cells as the seeded arm at both loads. It moved its killers 40 % less per treatment.
- **Limiting recruitment is what makes it good.** Without the cap and without repulsion, every killer that hears a pest walks towards it. The cap limits how far a pest recruits, and repulsion stops a killer from following one that is closer. Each cuts the infested cells by about a tenth, and they add up.

## What went wrong on the way

**Integer hops jitter.** With integer hops a robot's own value is one more than the lowest it hears, so only beacons at that one value pull, and that set changes with every cell moved. Killers reversed 10 % of their moves against 2.5 % in the seeded arm.

**Two robots on one cell hold a value forever.** With metric hops and no minimum hop, a killer standing on a relay's cell takes the relay's value at no cost and broadcasts it, and the relay hears it back at no cost. The pair keeps a value that never rises, long after its pest was cleared, and killers sat motionless on such cells 83 % of the time. The minimum hop of one cell breaks the loop. Expiring beacons by the source's send time also fixed it; an inhibitor wave sent by the treating killer helped but did not fix it.

**A killer on a relay shuttled back to a higher neighbour.** A killer standing on a relay's cell has its own hop number one minimum hop above that relay. A neighbouring relay just above the relay, but below the killer's own number, was therefore downhill and pulled the killer back, and from there the relay was downhill again. This happened wherever the next link of the chain was at the edge of the radio range, so that a killer passing the relay did not hear it. Killers shuttled in 39 % of their ten-minute windows on a heading. A robot on a beacon's cell now compares with that beacon's value.

**The escape never fired on a killer that kept losing its heading.** The escape first measured progress over its last 40 decisions on a heading, as the distance between the first and last cell over the distance walked, and every decision without a heading started that window over. On the large obstacle field a single killer shuttled between two cells on a river bank for 77 minutes: on one cell a relay across the river pulled it into the river, the steerer slid it one cell along the bank, out of that relay's range, and the relay on its old cell pulled it back. Three or four decisions in each 15-decision cycle had no heading, so the window never held more than about 12 decisions and no escape beacon was placed. The score also ignored speed: a killer that stepped one cell and then stood still for nine minutes scored a perfect 1. The window is now a time, the score divides by the distance the killer's top speed allows, and only a treatment, or a long wait without any heading, starts it over.

**Unit pulls hold a killer between pests.** With pulls of equal size from any distance, a killer surrounded by several pests of about the same strength settles at their geometric median, a stable point, and shuttles there. On the river this kept a killer from treating anything for 40 minutes. A distance weight makes that point unstable.

**Resetting the steerer turned killers back along a wall.** The steerer follows a wall with its right hand and commits to that side. It was reset whenever the strongest beacon's cell changed, which happens often, and a reset steerer may pick the other side and walk back. On the river with 4 killers, 258 detours were cut short and 12 % of the killers' ten-minute windows on a heading made less than 0.3 progress. Resetting only on a stall left no detour cut short and cut those windows to 8 %, with as many pests treated (184 against 183). A plain obstacle beacon on every wall the heading ran into, an earlier option, was always placed while the steerer was already following that wall: it fought the steerer, treated slightly fewer pests (180), and was removed.

**Cleared pests leave ghosts.** When a pest is cleared its source falls silent, but relays keep feeding each other values that rise one hop per round and never vanish (count-to-infinity). The cap removes them once they rise past it. While other pests exist, their live gradients win the minimum, so ghosts cost about two cells of travel per treatment.

## What these results do and do not show

- Relays stand still. With moving explorers as relays, chains break and re-form, and stale beacons matter more.
- The benchmark field has no obstacles. The river scenario has one killer and few seeds, so it shows the direction of an effect, not its size.
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
uv run python -m studies.benchmark_pests --hours 3 --seeds 20 --pests-per-hour 15
```

```bash
uv run python -m studies.benchmark_pests --hours 2 --seeds 20 --pests-per-hour 30
```
