# The explore behaviour

This document describes `ExploreBehaviour` in [`behavior.py`](../src/swarm_field/swarm/behavior.py): what it proposes, how it scores its bid, and what is left to do. The EXPLORE field it follows, `ExploreSeeder`, is described in [`explore.md`](./explore.md).

## Purpose and boundary

`ExploreBehaviour`, registered in the runner as `explore`, proposes one candidate at every decision point: `FollowField("explore")`, for the objective `EXPLORE`, with no target of its own. It never reads the field. The brain builds the field only if this candidate wins the arbitration, and the field then picks the target and leads the robot there.

The behaviour takes the name of the field to follow and the field's `ExploreConfig`, the same one the seeder gets. A brain spec passes it under `"explore"`, and `ExplorerPreset(explore=...)` does so for every explorer of a scenario:

```python
explore = ExploreBehaviour(seeder="explore", config=ExploreConfig(revisit_age=14 * TimeUnit.MINUTE))
```

## The bid

The bid is the mean of three criteria, each from 0 to 0.99:

$$\text{Bid} = \frac{1}{3}\Big(\text{NonScoutShare} + \text{MeanRevisitWorth} + \text{ExploredShare}\Big)$$

| Symbol | Meaning |
| --- | --- |
| NonScoutShare | The share of the peers the robot remembers that are not scouts; 0.99 when none of them is a scout. A robot among few explorers explores more eagerly. |
| MeanRevisitWorth | The mean over the field's cells of what revisiting each is worth, $1 - 2^{-\text{age}/\text{RevisitAge}}$ once its age reaches the configured `revisit_age`, and 1 for a cell never observed; at most 0.99. A field gone stale draws the robot out. |
| ExploredShare | The share of the field's cells the robot has observed at least once. |

An explorer has only this behaviour, so its bid always wins and its value changes nothing yet. It matters once a robot weighs exploring against another behaviour, e.g. a killer that also explores, whose `TreatInfestation` bids 0.99.

## What is left to do

- **"Nearby" peers.** NonScoutShare counts every peer the robot remembers, however long ago it heard from it. Counting only peers heard within `ExploreConfig.peer_max_age` would match its intent.
- **Weights.** The three criteria count equally. A question that needs exploring to compete with other behaviours should choose the weights, e.g. from how much of the field is stale.
- **More criteria.** Recent revisits, uncertainty, terrain accessibility and the robot's battery are candidates.
