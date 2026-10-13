# Explore Behaviour v2.0

This document describes the current `ExploreBehaviour` implementation in [behavior.py](../src/swarm_field/swarm/behavior.py), including how it obtains its field, builds a score, and chooses between following the field and idling. It also records the main work needed to develop the behaviour further.

## Purpose and boundary

`ExploreBehaviour` proposes one scored candidate at each decision point. It does not move the robot itself: when a usable field target exists, it proposes `FollowField`; otherwise, it proposes `Idle`. The brain selects among behavior candidates and executes the selected intent.

The behaviour is constructed with the name of a seeder registered in its `FieldProvider`:

```python
explore = ExploreBehaviour(seeder="explore")
```

The name must match a registered seeder. There is no built-in `ExploreSeeder` or default seeder name yet, so the configured name must be supplied by the caller.

## Field access

At each decision, the behaviour asks `ctx.fields.field_for(self._seeder, position, ctx.time)` for the field. `FieldProvider` keeps a cached field while its inputs remain valid and builds a replacement when needed. The solved field is derived from the robot's memory and terrain; `MemoryView` itself stores observations, not a solved potential field.

The returned field may be `None`, and a present field may have no target for the robot's position. Those cases still result in an `Idle` candidate, but the score is still built from the remaining criteria using their fallback values.

## Candidate and score

`propose(ctx)` returns a tuple containing one `ActionCandidate`. Its score is the weighted sum of three criteria:

$$
score = \sum_{i=1}^{3} w_i c_i
$$

The current criteria are:

- **Non-scout fraction:** one minus the fraction of `PeerObservation` payloads in the inbox whose body is `BodyType.SCOUT`. If there are no peer observations, the criterion returns `1.0`.
- **Target revisit worth:** the revisit worth of the chosen target cell, computed from `revisit_worths(last_observed_times[target], monitorable_cells[target], now, revisit_age)`. If there is no useful target, the criterion falls back to `1.0`.
- **Unexplored-cell criterion:** one minus the fraction of cells in `last_observed_times()` that are still `-inf`, which marks cells never observed. If there are no cells, the criterion returns `1.0`.

The candidate uses `FollowField(self._seeder)` when the field has a target other than the robot's current position. It uses `Idle` when the field is missing, the target is missing, or the robot is already at the target. The candidate's `target` is the field target, or `None` when there is no field target.

## Current weight behavior

The behaviour stores its weights and last update time between calls. It initializes all three weights to one third. When simulation time advances, `_update_weight` blends the previous weights toward normalized criterion importances using `alpha=0.2`; at the same or an earlier timestamp, it keeps the previous weights.

All current importances are `1.0`, so their normalized values are equal. Starting from equal weights therefore leaves the weights equal on every update. The state and update method are scaffolding for dynamic weights, not an adaptive weighting strategy yet.

## Further implementation

- **Name and implement the exploration seeder:** add an `ExploreSeeder` with a stable name, register it with `FieldProvider`, and pass that registered name to `ExploreBehaviour`. Its seeds should be derived from exploration-relevant facts in memory.
- **Add extra exploration criteria:** replace the exact potential-value check for unexplored cells with explicit coverage or visitation information, and consider recent revisits, uncertainty, expected information gain, terrain accessibility, and peer separation.
- **Implement dynamic weights:** derive criterion importances from a stated objective or feedback signal rather than assigning every criterion `1.0`. Define how elapsed simulation time affects updates, keep weights normalized and bounded, and specify behavior for repeated timestamps and missing criteria.
- **Add a demo:** show the behavior operating with a registered exploration seeder and several robots, with its candidate scores observable during a run.

## Summary

Version 2.0 documents the current implementation: one candidate is scored from peer composition and the provided field, with field-based fallback scores when data is absent. The candidate follows the configured field when it has a distinct target and otherwise idles. Seeder implementation, richer criteria, and truly dynamic weights remain future work.