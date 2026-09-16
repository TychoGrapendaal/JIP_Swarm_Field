# Navigation Module — A.3

> **Purpose.** This document is the design description of the **navigation module** (A.3 Navigation & Exploration in [`system-decomposition.md`](./system-decomposition.md)). It turns a *goal* — or the deliberate absence of one — into a concrete movement intent for the next tick: how the two robot bodies move differently, how land robots respect their movement constraints, how explorer robots keep apart from each other, and how a utility robot finds a pest by following a chained series of hop vectors broadcast by the explorers.
>
> **Scope.** The module **receives** goals from the Decision Engine (A.1); whether a pest is "close enough" to chase is a decision, not a navigation concern. The explorer *hop numbers* are defined here because navigation consumes them, but computing them is the job of A.4 (Communication & Knowledge). As in the decomposition doc, the type sketches are real contract shapes but field lists may grow — they fix the seam between A.1 (what is wanted) and A.3 (how to move), never the internals of other modules.
>
> **A note on the architecture's `Navigator` port.** [`architecture.md`](./architecture.md) sketches a `Navigator.step_towards(target: CellId) -> Action` whose target is an absolute grid cell. This module deliberately replaces that with vector-based steering: a utility robot navigates by relative vectors it composes from perceived neighbours, never by trusted absolute cell addresses (it has no ground truth). The change is internal to component A; the A.1 ↔ A.3 boundary is the `NavGoal` defined below.

---

## 1. The module's boundary

Three rules, in the spirit of the decomposition doc's boundary principles:

1. **A.3 never chooses a goal.** A.1 hands it a `NavGoal` every tick. If A.1 says "nothing is worth moving for", navigation must not invent a target — it holds still and waits.
2. **A.3 is movement-only.** It converts a `NavGoal` + current perceived state into a movement intent. It never asks the world for ground truth; everything it reads is a perception or a message already decoded by A.4. Decision *why*, navigation *how*.
3. **A.3 is deterministic given its inputs.** It holds no private random source. Random drift vectors come ready-made from A.1 (drawn from the run's single seeded RNG), so a run is reproducible.

```mermaid
flowchart LR
    A1["A.1 Decision Engine"] -->|"NavGoal"| NAV["A.3 Navigation"]
    MEM["A.2 Memory / A.4 decoded hop-field"] -->|"HopFieldEntry"| NAV
    SEN["Perception.self_state (position, facing, battery)"] --> NAV
    NAV -->|"MovementIntent"| ACT["B.2 Act stage → Move / Turn"]
    NAV -->|"no resolvable gradient"| A1
```

---

## 2. The two robot bodies

Two body types from `contracts` (SCOUT and KILLER) exist at project start, and they move fundamentally differently. Navigation's job splits into two bodies of rules.

| Aspect | Explorer (SCOUT, aerial) | Utility / killer (KILLER, land) |
|---|---|---|
| Sensory/movement medium | Flies; no cell-level obstacles | Drives on the ground |
| Movement constraints | Field border only | Field border **and** cannot enter crop cells |
| Has a goal to navigate to | Never — always in exploration mode | Yes, whenever A.1 issues one |
| Attitude to other explorers | Strong aversion (personal-space separation) | Indifferent |
| Attitude to utility robots | Indifferent | Indifferent |
| Broadcasts hop beacons | Yes (it owns the pest gradient) | No — it only consumes them |
| Publishes its own hop number | Yes, hop 0 when it sees the pest | No — it never broadcasts hop beacons |

The explorer's role in navigation is minimal and negative: it has no destination, so its "navigation" is a biased random walk plus a separation impulse (Section 10). Everything below this point concerns the utility robot unless the section says otherwise.

---

## 3. Coordinate frame and vector conventions

All vectors are expressed in the simulation's metric frame, which must agree with the grid conventions already in `contracts.ids`:

- **x increases East, y increases North** — same convention as `Direction.offset`.
- A field cell at `(cx, cy)` has its centre at metres `(cx * cell_size, cy * cell_size)` with `cell_size` from `FieldGeometry`.
- A vector is a *relative* displacement in metres, `(dx, dy)`, in this frame. `Vector2 = tuple[float, float]`.

Two consequences the whole module relies on:

1. **Navigation composes relative vectors, never absolute positions.** The utility robot measures `r` (the vector from itself to a broadcasting explorer) by perception, and receives `v` (that explorer's vector to the next hop down) in the message. Absolute addresses would be subject to a position error the robot cannot trust away; composed relative vectors cancel the shared error and stay valid as long as the messages are fresh.
2. **Heading, not position, is what persists across ticks.** Because every hop vector is re-issued and re-measured continuously, the module re-solves its desired heading every tick (Section 9) instead of remembering long waypoint chains.

---

## 4. The navigation goal (`NavGoal`)

A.1 tells navigation what it wants through one small union. There are exactly three modes in v1 — no more, because navigation should not be in the business of proliferating goals.

```python
from dataclasses import dataclass
from enum import Enum, auto

Vector2 = tuple[float, float]


class NavMode(Enum):
    HOLD = auto()     # no useful goal right now: stand still and wait
    SPREAD = auto()   # move along one fixed drift vector; there is no destination
    CHASE = auto()    # commit to following the pest gradient (hop chain) until it resolves


@dataclass(frozen=True)
class NavGoal:
    mode: NavMode
    drift: Vector2 | None = None   # SPREAD only — the random vector, metres, fixed per deployment
    max_hop: int | None = None     # CHASE only — chase while freshest heard hop is <= max_hop
```

Design intent:

- **HOLD** is what a utility robot does while it has no information: parked at (or near) the truck. Every driver that would otherwise produce movement — no pest gradient known, no charge pressure, nothing shared that warrants a trip — resolves to this until A.1 decides otherwise.
- **SPREAD** carries a single drift vector that A.1 drew once from the seeded RNG (Scenario 1, Section 6). Navigation holds the heading constant so the robot actually ends up somewhere off its start point; a per-tick re-roll would keep the robot parked in expectation at the spawn point.
- **CHASE** carries no vector — the target is a *live* value that only exists in the hop-field (Section 7). Navigation reads the field fresh each tick (Section 8).

Navigation also reports back to A.1 with one, and only one, signal: **"no resolvable gradient"** — the committed chase chain went stale (Section 8, edge cases). A.1 then re-issues a goal (typically a fresh SPREAD). This is the single output of A.3 that is not a movement intent.

---

## 5. Border and obstacle constraints

Movement constraints are checked in a fixed order, and they are the same every tick regardless of mode:

1. **Field border (both bodies).** The field is a `width × height` cell rectangle. Any candidate position that would cross the border is rejected; navigation replaces the heading with the admissible tangent (slide along the border) or rotates the drift vector back into the field. For CHASE mode this usually means "skirt the border until the hop-field re-syncs" — a pest cannot be outside the field.
2. **Crop and obstacle cells (land only).** A utility robot may not enter a cell whose ground-truth `crop` is not `None`, nor an `obstacle` cell. This is the biggest single constraint named in the system: the killer robot must keep the rows. When the straight-line heading is blocked, navigation picks the admissible neighbour whose direction has the largest dot product with the desired heading (a tangent detour), commits the turn, and reverts to the desired heading once the blocker is passed — see the per-tick flow in Section 9.
3. **Occupancy by another robot (both bodies, deferred).** Robot–robot collisions are an open decision (Section 13). Until resolved, navigation ignores the `occupant` field; the aversion rules in Section 10 are behavioral, not physical.

Aerial explorers skip rule 2 entirely: they are bounded only by the field border.

```mermaid
flowchart LR
    H["desired heading"] --> B{"crosses field border?"}
    B -- "yes" --> TB["slide / rotate back into field"]
    B -- "no" --> C{"land robot?"}
    C -- "yes" --> O{"next cell has crop or is an obstacle?"}
    O -- "yes" --> D["pick best-aligned free neighbour (detour)"]
    O -- "no" --> OK["heading OK"]
    C -- "no" --> OK
    TB --> OK
    D --> OK
    OK --> INT["MovementIntent"]
```

---

## 6. Scenario 1 — deployment and random spread

The situation at the start of the swarm:

- All robots begin at (or within sensing distance of) the truck / charging station.
- Because no pest has been found, the explorers' hop numbers are all **∞** (encoded as `None` — Section 7), so the utility robots have no gradient to chase.
- A.1 therefore issues **SPREAD** to each deployed utility robot, with a random drift vector drawn once from the seeded RNG — for example *10 m west, 20 m north*. Only a very small number of utility robots are deployed now, typically 1 or 2, possibly none.
- Navigation holds that heading (Section 4) and follows it until the border, bouncing/sliding on contact (Section 5), until A.1 hands over a real CHASE goal.

```mermaid
flowchart TB
    T["All robots at truck / charging station"] --> Q{"any pest info yet?"}
    Q -- "no: explorers at hop ∞" --> RS["A.1 draws one seeded drift vector<br/>(e.g. 10 m west, 20 m north)"]
    RS --> G["NavGoal(SPREAD, drift)"]
    G --> M["move along fixed heading; bounce on border"]
    M --> Q
```

The *purpose* of the random vector is to keep the few utility robots out of local optima and out of the truck's shadow: without it a "no goal" robot would just sit where it was spawned. A fixed drift per deployment also guarantees a spread that is reproducible for a given seed — two runs with the same seed deploy the same 1–2 utility robots drifting to the same places.

---

## 7. The pest gradient (the hop chain)

The explorers collectively hold a **gradient down to the pest**: a hop number and — crucially for navigation — the *vector toward the next hop*. Only explorers broadcast; utility robots only listen.

### 7.1 Numbers and their meaning

- An explorer that detects the infestation directly has **hop 0**. It is rooted at the pest and knows the location vector to it.
- An explorer that has, in its freshest knowledge, a beacon of hop **n − 1** within its radio range announces itself as hop **n**, and attaches the relative vector to that hop-(n−1) neighbour — the *parent vector*.
- No pest knowledge at all is hop **∞**, encoded as `None` (this is exactly the state of Scenario 1).
- A utility robot never announces a hop; it only reads the gradient.

```python
@dataclass(frozen=True)
class HopBeacon:            # payload of Message(kind=KNOWLEDGE) — shape owned by A, opaque to B
    holder_hop: int         # this explorer's hop number; None-equivalent ∞ is never broadcast
    parent_vector: Vector2  # relative vector from this explorer toward the hop-(holder_hop-1) neighbour
    seen_at: int            # tick of the observation that produced this beacon
```

### 7.2 How navigation composes the chain

Let a utility robot U measure the relative vector **r** to an in-range explorer E that announces hop h with parent vector **v**. U's vector to the hop-(h−1) node is the vector sum:

```
v_next = r + v
```

where **r** = U → E and **v** = E → next hop. The addition is valid because vectors are relative and all live in the same metre frame: the target's position relative to U is what U measured to E plus what E reports onward. Chaining down:

```
U → pest  =  r_U→E2  +  v_E2→E1  +  v_E1→E0  +  v_E0→pest
```

Each term is re-measured / re-received continuously, so the utility robot only ever needs the *freshest* single beacon to steer toward the *next hop down*, never the whole chain at once: it follows a moving target one hop at a time.

```mermaid
flowchart LR
    P["pest (infestation)"] --- E0["E0 · hop 0"]
    E0 -- "v_E0→pest" --- E1["E1 · hop 1"]
    E1 -- "v_E1→E0" --- E2["E2 · hop 2"]
    E2 -- "v_E2→E1" --- U["utility 'U' · listens"]
    U -- "r_U→E2 (measured)" --- E2
```

---

## 8. Scenario 2 — the first pest is found, and it is close

The fuller scenario, exercising CHASE end to end:

1. An explorer detects the pest and broadcasts hop 0. Hop numbers propagate outward through the radio graph — 1, 2, 3, … — each explorer attaching its parent vector.
2. The utility robot's A.1 sees, through its in-range explorers, a freshest hop ≤ **2** (the commitment threshold in `NavGoal.max_hop`). It judges this close enough to commit and issues `NavGoal(CHASE, max_hop=2)`.
3. Navigation reads the hop-field. Say U's nearest in-range beacon is E2 (hop 2) with parent vector to E1. The pilot heading is `r_U→E2 + v_E2→E1`, i.e. steer toward E1.
4. U moves on that heading, recomputing every tick from fresh beacons and fresh measurements (reactive steering — it does not memorize the destination).
5. On entering E1's radio range, a fresher hop-1 beacon arrives with the vector to E0. Navigation switches parent: heading becomes `r_U→E1 + v_E1→E0`.
6. On entering E0's range, the hop-0 beacon resolves the chain's tail: the vector to the pest itself. U's heading becomes `r_U→E0 + v_E0→pest` and it arrives at the infestation, where the utility's *Treat* module takes over (a decision concern, not navigation).

```mermaid
sequenceDiagram
    participant P as Pest
    participant E0 as Exploring hop 0
    participant E1 as Exploring hop 1
    participant E2 as Exploring hop 2
    participant U as Utility robot

    E0->>P: senses pest, hop 0
    E0-->>E1: beacon hop 1, vector to pest
    E1-->>E2: beacon hop 2, vector to E0
    E2-->>U: beacon hop 2, vector to E1
    U->>U: heading = r(→E2) + v(E2→E1)
    U->>E1: moves into range
    E1-->>U: beacon hop 1, vector to E0
    U->>U: heading = r(→E1) + v(E1→E0)
    U->>E0: moves into range
    E0-->>U: beacon hop 0, vector to pest
    U->>U: heading = r(→E0) + v(E0→pest)
    U->>P: arrives at the infestation
```

**When the chain stalls.** If committed to CHASE but the hop-field goes stale (no beacon refreshed within the freshness window, Section 13), navigation emits the single **"no resolvable gradient"** report and A.1 re-issues a goal — usually a fresh SPREAD. The utility robot is never allowed to wander in CHASE without a live gradient; without it, moving is a decision A.1 has to make again.

### 8.1 The walkthrough as a 2D view of the field

The same scenario, step by step, drawn as a crop field: the crop rows run East–West, the aisles between rows and the two headland lanes (x = 0 and x = 12) are drivable, and the land utility robot may not enter crop cells. The example uses a `radio_range` of 3 cells, so the beacon chain hops every 2 cells; positions and ranges are illustrative, not a scenario spec.

```text
Legend (one character per cell; x grows East →, y grows North ↑):
  H  headland lane   — drivable strip at the field edges (x = 0, x = 12)
  .  bare aisle      — drivable ground between crop rows
  C  crop cell       — BLOCKED for the land utility robot (its key constraint)
  P  infested crop cell — the pest, sitting inside a crop row
  0 / 1 / 2  an explorer hovering over that cell (hop numbers 0 / 1 / 2); aerial, ignores cells
  U  the utility robot (KILLER body, land)
```

**Step 0 — the chain is formed.** E0 hovers in the aisle cell (5, 4) directly south of the pest at (5, 5) and broadcasts `{hop: 0, parent_vector: (0, +1)}` (one cell north, to the pest). E1 at (7, 4) announces hop 1 with parent vector `(-2, 0)` toward E0; E2 at (9, 4) announces hop 2 with parent vector `(-2, 0)` toward E1. The utility robot has finished its Scenario-1 drift and sits at (11, 4).

```text
y=8 | H...........H
y=7 | HCCCCCCCCCCCH
y=6 | H...........H
y=5 | HCCCCCPCCCCCH
y=4 | H....0.1.2.UH
y=3 | HCCCCCCCCCCCH
y=2 | H...........H
y=1 | HCCCCCCCCCCCH
y=0 | H...........H
```

**Step 1 — U commits to CHASE.** U is 2 cells from E2 (well inside range 3), so its freshest heard hop is 2 and A.1's `NavGoal(CHASE, max_hop=2)` fires. U composes its first target `r(U→E2) + v(E2→E1) = (-2, 0) + (-2, 0) = (-4, 0)`, turns to face West on the aisle and starts driving. The field is unchanged; the lane now reads:

```text
y=4 | H....0.1.2.UH   ← U faces west, target (-4, 0)
```

**Step 2 — U catches the fresher hop-1 beacon.** U advances to (9, 4), now standing at E2's hover cell (E2 stays airborne above it) and 2 cells from E1. E1's fresher hop-1 beacon replaces E2's as the pilot: `r(U→E1) + v(E1→E0) = (-2, 0) + (-2, 0) = (-4, 0)`. Same heading, but the gradient is now two hops shorter.

```text
y=8 | H...........H
y=7 | HCCCCCCCCCCCH
y=6 | H...........H
y=5 | HCCCCCPCCCCCH
y=4 | H....0.1.U..H
y=3 | HCCCCCCCCCCCH
y=2 | H...........H
y=1 | HCCCCCCCCCCCH
y=0 | H...........H
```

**Step 3 — the hop-0 beacon resolves the tail.** U advances to (7, 4), inside E0's range (E1 stays airborne above U). E0's beacon carries the vector to the pest itself: `r(U→E0) + v(E0→P) = (-2, 0) + (0, +1) = (-2, +1)`. The heading acquires its North component.

```text
y=8 | H...........H
y=7 | HCCCCCCCCCCCH
y=6 | H...........H
y=5 | HCCCCCPCCCCCH
y=4 | H....0.U.2..H
y=3 | HCCCCCCCCCCCH
y=2 | H...........H
y=1 | HCCCCCCCCCCCH
y=0 | H...........H
```

**Step 4 — arrival at the aisle, not the crop.** U drives the last two cells to (5, 4), the aisle cell directly south of the pest. It cannot enter (5, 5): that cell is a crop — the land robot's cardinal constraint — so it stops on the aisle, E0 cedes its hover cell, and treatment happens from the row edge. Navigation ends with `STAND` and "arrived"; A.1's Treat logic takes over from here.

```text
y=8 | H...........H
y=7 | HCCCCCCCCCCCH
y=6 | H...........H
y=5 | HCCCCCPCCCCCH
y=4 | H....U.1.2..H
y=3 | HCCCCCCCCCCCH
y=2 | H...........H
y=1 | HCCCCCCCCCCCH
y=0 | H...........H
```

Summary of the chase, per step:

| Step | U at | Freshest beacon (hop, parent_vector) | Composed target | Steering |
|---|---|---|---|---|
| 1 | (11, 4) | E2: hop 2, `(-2, 0)` | `(-2, 0) + (-2, 0) = (-4, 0)` | West |
| 2 | (9, 4) | E1: hop 1, `(-2, 0)` | `(-2, 0) + (-2, 0) = (-4, 0)` | West |
| 3 | (7, 4) | E0: hop 0, `(0, +1)` | `(-2, 0) + (0, +1) = (-2, 1)` | West then North |
| 4 | (5, 4) | — (arrived) | — | Stop; Treat from the aisle |

---

## 9. The per-tick navigation decision

Every tick, regardless of mode, navigation runs one deterministic procedure:

1. **Read the goal.** If `NavGoal` is absent or `HOLD` → movement intent *stand still*, done.
2. **Resolve the current desired heading.**
   - `SPREAD`: the fixed drift vector (Section 6).
   - `CHASE`: read the freshest qualifying `HopBeacon` from the hop-field; compose `v_next = r + v` (Section 7). None fresh → report **"no resolvable gradient"** and fall back to *stand still* until A.1 replies with a new goal.
3. **Add the separation impulse (explorers only).** Each explorer sums short-range repulsion from other explorers into the heading (Section 10); utility robots skip this.
4. **Apply body constraints.** Border for both; crop / obstacle cells for land robots only (Section 5).
5. **Emit the movement intent.** If the intended cell is reachable and the robot is already aligned → *move along heading*; if misaligned → *turn toward heading first* (one physical action per tick, per contracts); if blocked → *detour turn*.

```mermaid
flowchart TB
    S["start of tick"] --> RG{"NavGoal present and not HOLD?"}
    RG -- "no" --> HI["intent: stand still"]
    RG -- "yes" --> SG{"mode?"}
    SG -- "CHASE" --> HF{"freshest hop ≤ max_hop?"}
    HF -- "no" --> NR["report 'no resolvable gradient'"]
    NR --> HI
    HF -- "yes" --> CH["heading = r + parent_vector"]
    SG -- "SPREAD" --> SD["heading = fixed drift vector"]
    SD --> SEP{"aerial explorer?"}
    CH --> SEP
    SEP -- "yes" --> AV["add separation from other explorers"]
    SEP -- "no" --> B
    AV --> B["apply border + crop/obstacle constraints"]
    B --> AL{"aligned with grid heading?"}
    AL -- "yes, clear" --> MV["intent: move along heading"]
    AL -- "no" --> TN["intent: turn toward heading"]
    AL -- "blocked" --> DT["intent: detour along free tangent"]
    MV --> O["MovementIntent out"]
    TN --> O
    DT --> O
```

Because the heading is re-solved from live beacons every tick, CHASE is a pursuit, not a path plan: there are no waypoints, no global path, nothing for an individual robot to remember across ticks. That is a deliberate consequence of the "no trusted absolute geometry" principle.

---

## 10. Explorer aversion (separation impulse)

The one behavioral constraint that is neither border nor obstacle: explorers must not clump. Each aerial explorer, in addition to its exploratory wander, adds a short-range repulsion from every *other explorer* it perceives — never from utility robots. The impulse is:

- **range-limited** to a "personal-space" radius (a multiple of `radio_range`, scenario-tunable);
- **strongest when closest**, scaled by inverse distance and capped at a maximum deflection so it never overwhelms the exploratory heading;
- **summed** over all perceived explorers, then added to the desired heading before the border check in Section 9.

Utility robots take no separation from anyone — they do not mind sharing space with explorers, and their spacing emerges from their SPREAD/CHASE targets, not from aversion.

```mermaid
flowchart LR
    E1["explorer A (me)"] --> SUM["sum of separation impulses"]
    E2["explorer B, close"] --> SUM
    E3["explorer C, close"] --> SUM
    EX["exploratory heading"] --> SUM
    SUM --> B["border check"] --> I["movement intent"]
    U["utility robot"] -- "no impulse" --> NOP["no separation applied"]
```

---

## 11. Output: movement intent

The navigation module does not move anything; it emits an intent that the Act stage (B.2) realizes as a concrete `Move` / `Turn` on the discrete grid. The current `contracts` placeholders have the parameters of movement deliberately deferred, so this document fixes the seam as a distinct shape:

```python
@dataclass(frozen=True)
class MovementIntent:
    heading: Vector2       # unit vector in the field frame, after all constraints/impulses
    impulse: str  # one of STAND, MOVE_ALIGNED, TURN_TO_ALIGN, DETOUR
    # the Act stage translates heading to a concrete Move/Turn and expends energy
    explain: str           # human-readable reason, surfaced by inspect()
```

The translation rule (living in B.2 / Act, one physical action per tick): if the robot's current `facing` differs from the grid direction best matching `heading`, the tick is spent `Turn`ing; otherwise it is spent `Move`ing into the adjacent non-blocked cell best aligned with `heading`; if that cell is blocked (land only), it spends the tick on the `DETOUR` turn from Section 5. When `impulse` is `STAND`, the action is `Idle`.

---

## 12. Interface summary

| Neighbour | Gives to navigation | Expects from navigation |
|---|---|---|
| A.1 Decision Engine | `NavGoal` each tick | `MovementIntent`; the single signal "no resolvable gradient" when a committed CHASE goes stale |
| A.4 Communication & Knowledge | decoded `HopBeacon` hop-field, freshness-filtered | — |
| A.2 Memory | `Perception.self_state` (perceived position, facing, battery) | nothing directly |
| B.2 Act / Simulation | (translation of intents into actions) | `MovementIntent` per tick, exactly one |
| C.1 Visualization | — (via `inspect()`) | short human-readable `explain` string for the "click a robot" panel |

The boundary is deliberately asymmetric: A.1 is the *only* module that may put a goal in front of navigation, and "no resolvable gradient" is the *only* way navigation talks back to it.

---

## 13. Open decisions register

All open items below are internal to A.3 (or a B.2 translation detail), so the module boundary above is stable regardless of how they resolve.

| # | Decision | Affects boundary? |
|---|---|---|
| 1 | **Grid snapping and turn hysteresis.** Continuous `heading` → one `Move`/`Turn` per tick; threshold for "aligned enough", to stop turn oscillation on a straight chase | No |
| 2 | **Gradient freshness window.** Number of ticks a `HopBeacon` stays usable (decay), and the tie-break between same-hop beacons: prefer lower hop, then closer, then freshest | No |
| 3 | **SPREAD behaviour on stuck detection.** Single drift per deployment vs. re-roll after a stuck timeout; who detects "stuck" | No |
| 4 | **Robot–robot collisions.** Currently both bodies ignore the `occupant` cell field; land robots might later yield to each other | No |
| 5 | **Arrival semantics.** Stop exactly at the pest (`STAND`) vs. continuous pursuit around it while it "lives" | No |
| 6 | **Speed / energy coupling.** `MovementIntent` carries a `speed_fraction` vs. `v_max`; how battery consumption modifies it (a decision concern, surfaced here) | No |
| 7 | **Aversion parameters.** Personal-space radius, impulse cap and decay curve for the explorer separation (scenario-tunable) | No |

Every entry marked "Affects boundary? No" — the same success criterion as the decomposition doc: the A.1 ↔ A.3 seam (`NavGoal` / `MovementIntent` / "no resolvable gradient") holds no matter which option the implementers pick.

---

## 14. Dependency summary

```mermaid
flowchart TB
    C2[contracts]
    A1[A.1 Decision Engine]
    A2[A.2 Memory]
    A4[A.4 Comms & Knowledge]
    A3[A.3 Navigation]
    B2[B.2 Act stage]

    A1 -->|NavGoal| A3
    A2 -->|self_state| A3
    A4 -->|HopFieldEntry| A3
    C2 --> A1
    C2 --> A2
    C2 --> A3
    C2 --> A4
    A3 -->|MovementIntent| B2
    A3 -->|no resolvable gradient| A1
```

- Navigation depends only on `contracts` and on the values its neighbours hand it behind that contract — it is unit-testable with hand-built `NavGoal`s and a fabricated hop-field, no simulation.
- Navigation never reads ground truth, never touches a goal on its own, and never sends `Message`s itself (the hop beacons it consumes are produced by A.4).