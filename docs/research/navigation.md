# Research note — goal-directed navigation for the utility robot

Status: describes the approach implemented in `src/swarm_field/swarm/navigator.py` (v1), the A.3 steering policy that turns a goal into `Move` / `Turn` actions. This note covers only goal-directed navigation for the utility (killer) land robot, which must steer toward a sensed pest while staying on passable ground (it cannot cut through crops). Coverage planning for the explorer is a related but separate problem, sketched in "Coverage control". The design context lives in `docs/navigation-module.md`; the domain background in the literature review (§3.2).

## The problem

The robot gets a goal (a relative direction to a pest, fused from radio readings) and must approach it on a partially known field: crops form obstacles it must walk around, the field edge must not be crossed, and the robot has no absolute position — it feels a crop ahead and its own heading, nothing more. In one tick it may turn or step one cell. The questions are: which heading leads toward the goal, and what to do when that heading is blocked.

## Candidate approaches

Artifical potential fields are the classic reactive answer (Khatib 1986): an attractive field pulls toward the goal, repulsive fields push off obstacles, and the robot follows the gradient. It is cheap, smooth and entirely local, which fits a swarm of resource-limited robots. Its famous failure mode is local minima: fields cancel on concave features, so a robot can circle a pocket forever (Borenstein & Koren 1989). Fields also need tuning (gains, influence radii) to behave near corners, and they are brittle on a discrete grid where a "force" can point into a crop cell.

Wall-following / Bug algorithms (Lumelsky & Stepanov 1987) are the other classical escape: when the direct path is blocked, hug the obstacle's boundary until the straight line to the goal becomes clear again. They are deterministic and need no representation of the world, just local touch; on a known-connectivity crop field, sliding along a row edge naturally deposits the robot at the row end, which is exactly where the aisle becomes reachable.

The Vector Field Histogram family (Borenstein & Koren 1991; Ulrich & Borenstein 1998) builds a polar histogram of obstacle density from range sensors and steers to the valley closest to the goal. It is a good middle ground but assumes a multi-direction rangefinder and, again, a continuous metric space; its goal alignment can also pick a corridor that dead-ends. Global planners (A* / Dijkstra / wavefront) provably find a route but need a map of free space, which the robot does not have (belief is partial and noisy, per `docs/navigation-module.md` §3.1) — they are the rescue for a *replanning* layer, not the per-tick reactive policy.

## Coverage control (why the explorer is a different problem)

Coverage control (Cortés, Martínez & Bullo 2004) drives a team to spread so that a density function over the field is optimally approximated, typically via Lloyd/Voronoi iteration over the robots' sensing footprints. That fits the explorer's job (watch the whole field) but is the wrong tool for the utility robot, which has a definite goal to run down. The two policies are complementary: the explorer keeps the field covered; when it broadcasts a found pest, the utility robot's navigator takes over to steer there.

## The chosen approach

The navigator combines the two classical ideas at their best point: gradient alignment for the direct approach (pick the compass heading whose unit offset has the largest dot product with the goal direction, preferring the one needing the least turning) and, when that heading is blocked, a committed right-hand wall-follow — pick the most goal-aligned *clear* heading once (ties broken clockwise from the goal heading) and stick to it for the whole episode. This commitment is the load-bearing choice: re-deriving the tangent against the goal bearing every tick is what traps a robot between two clear neighbours on either side of a crop row, shuttling forever (a pure greedy potential-field local minimum). The robot only re-routes when the wall itself forces it: a blocked tangent ahead rounds the corner toward the goal if that way is open, otherwise it shifts clockwise to keep hugging the wall. It breaks off the wall only once the goal is genuinely reachable — right next to it, or level along the slide axis with a clear bearing — which lets an east-west crop row be rounded cleanly at its headland end. When no route exists at all (all neighbours blocked, or the goal distance never beating its running minimum for a long stretch) the navigator stops and reports `is_stalled`, handing the decision back to the brain (A.1), which may re-issue or replan the goal; it does not invent one itself.

This choice is the low-risk, hard-to-misbehave option for the stated constraints: no absolute position, no map, cheap per-tick computation, fully deterministic (no RNG), and — because the detour is committed wall-following plus a fixed hand — free of both potential-field minima on concave corners and of the oscillation of naive "try the goal, else random step" policies. Its known limitations are accepted for v1 and recorded as open decisions in `docs/navigation-module.md` §13: no memory of visited cells (a long dead-end is only detected by the stall timeout, a tunable `stuck_ticks`), diagonal corner-cutting is permitted by construction, and there is no guarantee of a route where a global planner would find one.

## References

- Khatib, O. — *Real-Time Obstacle Avoidance for Manipulators and Mobile Robots*. IEEE ICRA / IJRR, 1986 (artificial potential fields).
- Borenstein, J. & Koren, Y. — *Real-Time Obstacle Avoidance for Fast Mobile Robots*. IEEE Transactions on Systems, Man, and Cybernetics 19(5), 1989 (potential-field local minima).
- Borenstein, J. & Koren, Y. — *The Vector Field Histogram — Fast Obstacle Avoidance for Mobile Robots*. IEEE Transactions on Robotics and Automation 7(3), 1991.
- Lumelsky, V. J. & Stepanov, A. A. — *Path-Planning Strategies for a Point Mobile Automaton Moving Amidst Unknown Obstacles of Arbitrary Shape*. Algorithmica 2, 1987 (Bug / wall-following algorithms).
- Ulrich, I. & Borenstein, J. — *VFH+: Reliable Obstacle Avoidance for Fast Mobile Robots*. IEEE ICRA, 1998.
- Cortés, J., Martínez, S. & Bullo, F. — *Spatially-Distributed Coverage Optimization and Control with Limited-Range Interactions*. Mathematics of Control, Signals, and Systems 17, 2004 (coverage control, the explorer's problem).