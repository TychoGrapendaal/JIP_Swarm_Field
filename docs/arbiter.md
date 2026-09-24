# The arbiter

Status: implemented, 2026-09-24. This covers §3.4 of the field-level design, the part that decides what a robot is currently trying to do. Navigation works out how to get somewhere; this works out where, and when the robot is allowed to change its mind.

## What it has to solve

Every objective owns a potential field. A behaviour evaluates its own field at the robot's cell and reports how attractive that objective looks right now. Several of them bid at the same time and exactly one has to win, because a killer either drives toward a pest or explores or heads back to a dock, never some blend of the three. Following one field at a time also keeps the robot's state readable: "K1 is treating P1" is a sentence you can print.

So there are two questions at every decision point. Which objective wins, and is the winner enough of an improvement to justify abandoning whatever the robot was already doing.

The second one is the awkward one, and it comes straight out of how information moves through this swarm. There is almost never a connected network. A killer meets another killer roughly every 20 minutes, and information arrives in bursts when robots happen to pass each other. So scores jump rather than drift. A robot that just follows the highest score every tick would turn around every time an encounter revealed a slightly better pest, drive back and forth, and finish nothing.

## Getting objectives onto the same scale

Treating a pest, exploring unknown ground and returning to a dock are measured in completely different things. The potential field already sorted this out and the arbiter just inherits it: every field comes out in cells, because Φ subtracts a value term scaled by β from a route cost, so a target of value 1 is worth β cells of driving. That is the shared currency. Bids from different objectives compare directly, with no per-objective weighting inside the arbiter at all.

A bid's score is

```
score = -potential + urgency
```

The sign flips because lower potential means better. `urgency` is an additive term, also in cells, for priority the field has no way to express. A battery running down is the obvious one: it has nothing to do with how far the dock is.

I kept those two separate deliberately. Folding urgency into β would mean that raising β for TREAT, to make killers chase distant pests harder, would also quietly make TREAT beat EXPLORE more often, with no way to pull the two effects apart afterwards. β shapes the field, urgency sets priority, and they get tuned independently.

## The three rules

They apply in this order.

**Overrides skip everything.** A bid marked as an override wins outright, whatever the scores say, whatever the dwell timer says, whatever the robot was doing. This is for the handful of situations that must not be traded off at all. A battery low enough that the robot would strand itself is the main one; a killer standing on a pest it has already claimed is the other. Stranding costs us the robot and the pest, and then somebody has to walk out into the field and carry it back, so there is no infestation severe enough to make the gamble worth it. A penalty could always be outweighed by a good enough target. An override cannot.

**A commitment is held for a minimum dwell time.** Under that, the current commitment stays if it is still on offer. This is the cheapest of the three and probably the most effective, and it costs one timer. The reasoning is just that encounters deliver information in bursts, and the robot should not re-plan at the rate its map changes.

**Switching needs a margin that grows with progress.** A rival has to beat the current commitment by

```
margin = floor + commitment_weight * progress
```

where progress is the fraction of the original journey already driven. Objective switches and target switches use different floors, see below. So the margin starts at the floor the moment the robot commits and climbs to floor + `commitment_weight` by the time it arrives.

## Why progress rather than a percentage

The obvious alternative, and the one the design doc originally suggested, is "switch only if the new objective is at least 20 % better". I tried that first and then dropped it. Three reasons.

It protects the wrong thing. What actually deserves protecting is the work that would be thrown away by switching, and a percentage scales with how good the current target is, which is a different quantity and frequently the opposite one. Take a killer that has just committed to an excellent pest 80 cells away: big score, so a percentage margin is big, so the robot is hard to redirect. But redirecting it costs nothing, because it has not driven anywhere yet. Now take a killer two cells from a mediocre pest: small score, small margin, easy to pull away, at precisely the moment that is most wasteful. The percentage is strictest when switching is cheapest and loosest when it is most expensive. Scaling by progress gets both cases the right way round.

It throws away the units. Percentages make sense when a scale is arbitrary, and ours is not. Φ is in cells, and "5 cells better" already means something I can defend in a review: worth 50 m of extra driving to change your mind. Turning that into a ratio discards the interpretability the field handed us for free.

It breaks where the scores cross zero. Score is `-potential + urgency`, and Φ is negative on a good target (−8.28 at P1 in the worked example) and positive when the target is far away, so scores run from positive through zero to negative. A multiplicative threshold flips over across that sign change. At a score of −2 a rival at −1 passes `rival > 1.2 * current` trivially; at +2 a rival has to reach +2.4. Same rule, opposite strictness, depending only on which side of zero the robot happens to be standing. Taking the magnitude patches that particular symptom, but the first objection still stands afterwards, so it is not worth the trouble.

One thing to be upfront about: counting progress means letting a sunk cost influence a decision, which is not rational. Strictly only the remaining distance should matter. It is a deliberate trade of a bit of myopic optimality for stability and I would rather say so than pretend it falls out of an optimum. It is also roughly what ants do, since site fidelity keeps a forager going back to a known source instead of continuously re-optimising.

## There are two kinds of thrashing, not one

Flip-flopping between objectives is the obvious failure. The field structure creates a second one quietly.

Because Φ takes the minimum over all targets, the winning target can change without the objective changing at all. A killer heading for P1 meets a drone, hears about P4 which is slightly better, and turns around. It is still doing TREAT the whole time, and it is still thrashing. A margin applied only between objectives would never notice.

So the arbiter identifies a commitment by the pair (objective, target) rather than by the objective on its own, and uses a smaller floor when the rival shares the current objective. Swapping one pest for another is a smaller decision than giving up treatment for exploration, so it should be easier, but not free.

That puts one requirement on the behaviours: keep proposing the committed target while it is still valid. A behaviour that only ever bids its current best leaves the arbiter nothing to compare against, so the arbiter finds no match for its commitment, decides the commitment must have vanished, and adopts the newcomer with no margin at all. A behaviour that knows about many targets should bid its best few and always include the committed one.

A commitment that has genuinely disappeared from the bids, because someone else treated the pest or claimed it, counts as finished and the robot is freed immediately.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `objective_floor` | 5.0 cells | margin to change objective at zero progress |
| `target_floor` | 2.5 cells | margin to change target within an objective |
| `commitment_weight` | 15.0 cells | extra margin at full progress |
| `min_dwell` | 30 s | how long a commitment is held before reconsidering |

These are guesses. 5 cells is 50 m of driving, and a `commitment_weight` of 15 means a robot about to arrive needs a rival 20 cells better to be pulled away. Nothing derives any of them.

The number that will tell us whether they are right is switches per completed job. Above roughly 0.5 the margins are too low. We cannot measure it until the navigator produces real potentials and the loop actually runs robots, so treat the defaults as placeholders.

## What it deliberately does not do

It does not compute potentials, which is navigation's job, and it does not pick targets, which each behaviour does from its own field. It also does not know which objectives suit which body. A drone never gets a TREAT bid because it was never assembled with the TREAT behaviour in the first place, and putting that rule in the arbiter would duplicate a decision that belongs in assembly.

It has no idea other robots exist, either. Nothing in here stops two killers converging on the same pest, because they read roughly the same shared map, rank targets the same way and bid identically. That is the crowding problem, and claims solve it by removing a target from everyone else's set. Arbitration cannot.

## Still open

Composition rebalancing. How many killers should be active rather than parked at a dock is an arbitration question, since STANDBY competes like anything else, but I have not designed the urgency term for it. The response-threshold idea from the design doc fits here: a parked killer's STANDBY urgency decays stochastically as the believed pest load per active killer rises, and the randomness is what stops every parked robot waking up on the same encounter.

The SHARE urgency. It should scale with how much unshared information the robot is sitting on, except a robot has no way of knowing what its peers already know. Cheapest proxy I can think of is counting records picked up since the last encounter with a body type that needs them, weighted by value. Crude, and probably a research variable rather than something to tune.

Whether MONITOR and VERIFY stay distinct. Both target stale information with similar β, and for an explorer they might turn out to be nearly the same objective in practice. Worth checking once the fields are real, and collapsing them if so.

Where the energy feasibility gate lives. Dropping targets the robot cannot reach and still get back from belongs inside each behaviour, since that is where the targets and the dock distance are known. The RECHARGE urgency ramp is arbitration. Splitting the energy logic across two places works but it is not elegant, and I have not thought of anything better.
