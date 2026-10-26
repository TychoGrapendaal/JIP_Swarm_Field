# The arbiter

This covers how a robot decides what it is currently trying to do (#13, #39): the score scale every behaviour bids on, and the `UtilityArbiter` that picks one bid per decision point and decides when the robot may change its mind. Navigation works out how to get somewhere; this works out where. The code is in `swarm/arbiter.py`.

## What it has to solve

Every behaviour bids for the robot with `ActionCandidate`s, and exactly one wins, because a killer either drives toward a pest or explores or heads back to a dock, never some blend of the three. Following one objective at a time also keeps the robot's state readable: "K1 is treating P1" is a sentence you can print.

So there are two questions at every decision point. Which bid wins, and is the winner enough of an improvement to justify abandoning whatever the robot was already doing.

The second one comes straight out of how information moves through this swarm. There is almost never a connected network: robots meet now and then, and information arrives in bursts when they pass each other. So scores jump rather than drift, and a robot that just followed the highest score every tick would turn around every time an encounter revealed a slightly better pest, drive back and forth, and finish nothing.

## The score scale

The arbiter only sees the bids, so it can only compare them if they share one scale. Every score lies in `[0, 1]`, and the same score means roughly the same urgency whichever behaviour bids it:

| Score | Meaning |
|---|---|
| 0 | nothing to gain |
| 0.5 | an even need, such as one known pest per known killer for `TreatInfestation` |
| 0.9 and up | a need that overrides everything else, such as a battery about to strand the robot, or treating the pest the robot stands on (0.99) |

Every behaviour computes its score directly from memory and the robot's state, never from a potential field: building a field is the expensive part, so only the winner's field is built, and it then picks the target (see [`robot-memory-and-navigation.md`](../robot-memory-and-navigation.md)). Distance to a target therefore does not enter a score.

`TreatInfestation` bids 0.99 to treat the pest it stands on, so that only a more urgent need, such as the battery, interrupts a treatment. Otherwise it bids to follow the TREAT field with `1 / (1 + exp(-(pests - killers)))`, counting the pests and the killers it knows of: 0.5 with one pest per killer, more as pests outnumber killers. A recharge behaviour would read the battery and score high when it is low, crossing 0.9 when the robot must go now.

A bid is a plain `ActionCandidate`: action, score, source and target. The arbiter needs nothing else.

## The rules

The arbiter applies five rules in order. It knows nothing about specific behaviours: a bid's `source` names its objective and its `target` the cell it commits to, and both are compared for equality only.

**Overrides skip everything.** A bid scoring at least `override_threshold` (0.9) wins outright, whatever the other scores, the dwell timer or the robot's commitment. This is for the handful of situations that must not be traded off at all. A battery low enough that the robot would strand itself is the main one: stranding costs the robot and the pest, and then somebody has to walk out into the field and carry it back. A penalty could always be outweighed by a good enough target; an override cannot. Ordinary bids stay below 0.9. Among several overrides the highest wins.

**A finished objective frees the robot.** If no bid is for the committed objective any more, because the last pest is treated or the field is explored, the highest score wins at once.

**A finished target keeps the objective.** If the committed target is gone but its objective still bids, the robot takes on the objective's best new target and stays on the objective. An explorer reaches the cell it was heading for, observes it, and moves on to the next unexplored cell without having to win the objective again.

**A new objective is held for a minimum dwell time.** For `min_dwell` after the robot takes on an objective, no rival is considered. This is the cheapest rule and probably the most effective, and it costs one timer: encounters deliver information in bursts, and the robot should not re-plan at the rate its map changes.

**Switching needs a margin that grows with effort.** A rival has to beat the current bid by

```
margin = floor + effort_weight * effort
effort = min(1, time spent on the current target / effort_horizon)
```

So the margin starts at the floor when the robot takes on a target and climbs to floor + `effort_weight` once it has spent `effort_horizon` (5 minutes) on it.

Effort is counted per target, not per objective, because it stands for what switching throws away. A killer that has driven four minutes towards a pest loses those four minutes if it turns around. A drone that has explored for ten minutes loses almost nothing by going to treat a pest: the cells it explored stay explored, and only the few seconds towards the cell it was heading for are lost. Counted per objective, a long-running objective such as exploring would lock the robot in, unable to respond to a pest found next to it.

Effort needs nothing from the bid but its target, so every behaviour's plain `ActionCandidate` works, and the arbiter only needs the clock. The price: a robot that is stuck, wall-following around an obstacle without getting closer, keeps building effort and becomes harder to pull away. The cap at `effort_horizon` bounds that, and a stuck robot's bid usually drops or vanishes, which frees it. If that turns out not to be enough, a bid could also report its route to the target, and the margin grow with the distance driven instead.

## Why effort rather than a percentage

The obvious alternative is "switch only if the new objective is at least 20 % better". It protects the wrong thing. What deserves protecting is the work that would be thrown away by switching, and a percentage scales with how good the current target is, which is a different quantity and frequently the opposite one. Take a killer that has just committed to an excellent pest far away: a high score, so a percentage margin is big and the robot is hard to redirect, though redirecting it costs nothing because it has not driven anywhere yet. Now take a killer two cells from a mediocre pest: a low score, a small margin, easy to pull away at precisely the moment that is most wasteful. The percentage is strictest when switching is cheapest and loosest when it is most expensive. Scaling by effort gets both cases the right way round.

Counting effort means letting a sunk cost influence a decision, which is not rational: strictly only the remaining distance should matter. It is a deliberate trade of a bit of myopic optimality for stability. It is also roughly what ants do, since site fidelity keeps a forager going back to a known source instead of continuously re-optimising.

## There are two kinds of thrashing, not one

Flip-flopping between objectives is the obvious failure. The field structure creates a second one quietly. Because the potential takes the minimum over all targets, the target can change without the objective changing: a killer heading for P1 meets a drone, hears about a slightly better P4, and turns around. It is still treating the whole time, and still thrashing.

So a commitment is the pair (source, target) rather than the source alone, and a rival from the same source uses the smaller `target_floor`. Swapping one pest for another is a smaller decision than giving up treatment for exploration, so it should be easier, but not free.

That puts one requirement on the behaviours: keep bidding for the committed target while it is still valid. A behaviour that only ever bids its current best leaves the arbiter nothing to compare against, so the arbiter finds no match for its commitment, decides it has vanished, and adopts the newcomer with no margin at all. A behaviour that knows many targets should bid its best few and always include the committed one.

Since a behaviour never reads a field, a `FollowField` bid carries no target (`None`): the field picks the target only once the bid has won. The commitment of such a bid is to the objective alone, so this defence covers only bids that know their target from memory, such as `Treat` on the spot. Such a commitment also builds no effort: there is no work towards a target that a switch would throw away, and counting the time spent on the objective instead would lock an explorer in. Before this, an explore commitment reached the full margin of 0.5 after five minutes, so only an override could pull a killer away to treat (#60). Within one field, a target switch is left to the field (see "Still open").

## Parameters

The margins of the first arbiter were in cells: 5 to change objective, 2.5 to change target, and 15 more near the end of a journey. On the score scale they are converted at the steepest point of the logistic that turned a field's potential into a score in the first design, where one cell is worth $1 / (4 \times 10) = 0.025$:

| Parameter | Default | In cells, mid-scale | Meaning |
|---|---|---|---|
| `override_threshold` | 0.9 | | score from which a bid overrides everything |
| `objective_floor` | 0.125 | 5 | margin to change objective, with no effort spent |
| `target_floor` | 0.0625 | 2.5 | margin to change target within an objective, with no effort spent |
| `effort_weight` | 0.375 | 15 | extra margin once `effort_horizon` is spent on the target |
| `effort_horizon` | 5 min | | time on a target after which its effort counts in full |
| `min_dwell` | 30 s | | how long a new objective is held before reconsidering |

A robot that has spent five minutes or more on its target needs a rival 0.5 better to be pulled away, which from any commitment above 0.4 takes an override.

These are guesses. The number that will tell us whether they are right is switches per completed job (`ArbiterChoice.switched` marks each abandoned commitment; moving on from a finished target or objective does not count): above roughly 0.5 the margins are too low.

## Using it

```python
arbiter = UtilityArbiter()
agent = ComposedAgent(robot_id, behaviors, memory, fields, Navigator(fields.quadtree), arbiter)
```

One arbiter per robot, since it holds that robot's commitment. `ComposedAgent` requires one.

## What it deliberately does not do

It does not compute potentials, which is navigation's job, it does not pick targets, which the winner's field does, and it does not read memory or the robot's state: a behaviour that cares about the battery or the tank reads them and says so in its score. It does not know which objectives suit which body either: a drone never bids TREAT because it was never assembled with that behaviour.

It has no idea other robots exist. Nothing here stops two killers converging on the same pest, because they read roughly the same shared map and bid identically. That is the crowding problem, and claims solve it by removing a target from everyone else's set (#42).

## Still open

- **Exchange rates between behaviours.** Each behaviour's heuristic sets how much it wants the robot, and the heuristics need tuning against each other once TREAT and EXPLORE run together.
- **Committing to a target within a field.** A `FollowField` bid carries no target, so the arbiter's target margin does not hold the robot to the cell its field leads to. If target switches within a field turn out to thrash, the field provider could keep the committed target's seed unless a rival beats it by a margin.
- **Standby and composition.** How many killers should be active rather than parked at a dock is an arbitration question, since STANDBY would bid like anything else, but its score is not designed yet.
- **Where the energy feasibility gate lives.** Dropping targets the robot cannot reach and still get back from belongs inside each behaviour, where the targets and the dock distance are known; the RECHARGE score ramp is the recharge behaviour's.
