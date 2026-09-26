# Exploration with Potential Fields

## Purpose

The exploration behaviour combines local sensor observations with a short-term memory of visited cells. It represents the robot's surroundings as several scalar potential fields, converts each field into a local force map, and superimposes those maps into one net force map.

This approach is intended to support decentralized exploration: every robot can make a movement decision from its own state, recent observations, and messages received from peers. No global map is required by the field calculation.

## Design

`Explore` owns a `PotentialField` and the force maps derived from it. At each decision point it:

1. Updates the potential field from the robot state and sensed observations.
2. Computes a force map for exploration, crop, infestation, obstacle, and peer potentials.
3. Rotates the obstacle force by 90 degrees so that obstacle avoidance can produce lateral motion rather than only pushing directly away from the obstacle.
4. Adds the individual force maps cell by cell into `net_force`.

The current implementation exposes this update pipeline through `Explore.update(ctx)`. The final step, converting the resulting vector field into `ActionCandidate` values, is still to be implemented or should be implemented in decision engine.

## Coordinate system

Potential maps use coordinates relative to the robot. The robot is always at `(0, 0)` and an observation at world position `(x, y)` is stored at:

```text
(x - robot_x, y - robot_y)
```

Only cells inside the configured circular radius are retained. When the robot moves, the persistent maps are shifted by the movement delta so that the same world cells keep their identity in the new local coordinate system. Cells that leave the radius should be remembered for a certain time.

The current implementation shifts the persistent `exploration` map and its `timemap`. The other maps are derived again from observations at the next update.

## Potential maps

`PotentialField` maintains these maps:

| Map | Meaning | Current status |
| --- | --- | --- |
| `exploration` | Memory of observed cells | Implemented and persistent |
| `crop` | Crop-related attraction or repulsion | Placeholder; not implemented |
| `infestation` | Repulsion from infestation | Implemented as a negative Gaussian |
| `obstacle` | Obstacle avoidance signal | Implemented as a positive Gaussian |
| `peer` | Peer-related influence | Implemented as a positive Gaussian |
| `timemap` | Most recent observation time per local cell | Implemented |

### Exploration memory

An observation of a previously unseen or weakly explored cell increases its exploration potential by `1`, capped at `4`. This makes exploration persistent across decision points while preventing repeatedly observed cells from growing without bound.

### Infestation field

An infestation observation is spread across nearby cells as a confidence-weighted radial Gaussian. Severity controls both the peak magnitude and the spread:

$$
U_{infestation}(p) = -c s \exp\left(-\frac{\lVert p - p_0 \rVert^2}{2s^2}\right)
$$

where $c$ is confidence, $s$ is severity, and $p_0$ is the observed infestation cell. The negative amplitude makes the resulting gradient produce a repulsive contribution when the force is defined as $-\nabla U$.

Observations whose three-sigma support cannot reach the local field are ignored. Non-positive severity produces no field.

### Obstacle field

Obstacles use a positive Gaussian with fixed standard deviation `1.0` and amplitude `1.5 * confidence`. Its force is subsequently rotated by 90 degrees in `Explore`, producing a lateral steering contribution around the obstacle.

### Peer field

Peers use a positive Gaussian with fixed standard deviation `1.0` and amplitude `0.7 * confidence`. This currently models a local peer influence only. Peer observations received through messages are not yet transformed from the peer's coordinate frame or merged into the field.

### Crop field

The crop potential is not implemented yet as it is unclear should be attractive or repulsive to the explorer drone.

## From potentials to forces

For each potential map, `Explore._compute_force` estimates the local gradient using central differences where both neighboring samples exist, and one-sided differences at an edge. The force is the negative gradient:

$$
F(p) = -\nabla U(p)
$$

For a cell $p = (x, y)$, the implementation computes approximate derivatives along both axes and stores the resulting vector in a force map. Missing potential cells contribute `(0.0, 0.0)`. The force maps are then summed:

$$
F_{net}(p) = F_{exploration}(p) + F_{crop}(p) + F_{infestation}(p) + F_{obstacle}(p) + F_{peer}(p)
$$

The maps are kept per source as well as in `net_force`, which makes the contributions inspectable during experiments and tuning.

## Temporal update rules

At the start of an update, derived maps are cleared and the current simulation time is recorded. Each observation is compared with `timemap`; an older observation for the same local cell is ignored. Newer observations update the timestamp and regenerate that observation's potential contribution.

The exploration map is intentionally not cleared. It moves with the robot and acts as bounded local memory. Infestation, obstacle, crop, and peer maps are rebuilt from the observations available for the current update.

## Relationship to the composed brain

The broader swarm architecture treats a behaviour as a producer of scored action candidates. `Explore` is intended to be the movement-oriented behaviour in that composition, while an arbiter chooses between exploration, treatment, charging, or other behaviours. The potential field is an internal navigation mechanism; it does not own world state and should only use the read-only `DecisionContext` supplied by the brain.

This separation leaves room to replace the potential-field navigator with another method, such as greedy coverage or a learned policy, without changing the simulation boundary.

## Current implementation boundary

The following pieces are present:

- local coordinate conversion for sensed observations;
- bounded circular fields;
- persistent exploration memory and timestamp filtering;
- Gaussian infestation, obstacle, and peer fields;
- finite-difference force calculation;
- force superposition and inspectable per-source maps.

The following pieces remain open:

- implement specific dataclass for the exploration;
- implement the crop potential;
- transform and merge peer observations from their original coordinate frames;
- implement decay rate for `net_force` deliberately at each update so stale cells will be diminished;
- convert the net force into legal movement actions and scored `ActionCandidate` values;
- decide how robot heading, speed, boundaries, and blocked target cells affect action selection;
- add experiments that compare coverage, infestation avoidance, obstacle clearance, and peer separation under different field weights.

## Research questions

The approach can be evaluated with repeatable seeded simulations using metrics such as explored-cell coverage, time to first infestation detection, distance travelled, collision or obstacle proximity, and redundancy between robots. Important parameters to sweep include the field radius, exploration cap, Gaussian widths, confidence scaling, and the relative weights of each force source.

The main hypothesis is that a bounded persistent exploration field, combined with repulsive hazard fields and peer influence, produces useful decentralized coverage without requiring a centralized world model. The action-conversion step is the key remaining implementation point needed to test that hypothesis end to end.