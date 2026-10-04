# Explore Behaviour v1.0

This document explains what `ExploreBehaviour` in [src/swarm_field/swarm/behavior.py](../src/swarm_field/swarm/behavior.py) is doing, how it is intended to be used, what type of object it returns, how the score is built, and what it does not do.

## What the class is doing

`ExploreBehaviour` is a heuristic exploration behaviour for a robot. It does not drive the robot directly. Instead, it reads the current decision context and proposes one or more candidate actions for the arbiter to choose from.

The class is a behaviour in the same pattern as the other swarm behaviours: it implements `propose(ctx)` and returns a tuple of `ActionCandidate` objects. Each candidate says:

- what the robot would like to do (`action`),
- how strongly it wants it (`score`),
- where it is targeting (`target`),
- where the bid came from (`source`).

In this implementation, the behaviour mainly does the following:

1. Reads the current `EXPLORE` potential field at the robot's position.
2. Searches the field window for the best candidate cells.
3. Rejects cells that are too close to peers.
4. Scores each candidate with a mix of field quality, separation, and crowd dispersion.
5. Returns the top-ranked candidates as `ActionCandidate` objects.

This means it is working as a local exploration heuristic rather than a full planner.

## How to use the class

The class is instantiated as:

```python
explore_behaviour = ExploreBehaviour()
```

It also accepts a custom seeder name:

```python
explore_behaviour = ExploreBehaviour(seeder="EXPLORE")
```

The default value is `ExploreSeeder.name`, so the normal case uses the standard exploration field.

To use it in the brain, you call:

```python
candidates = explore_behaviour.propose(ctx)
```

where `ctx` is a `DecisionContext` containing the robot's state, sensed observations, memory, field access, and inbox. The behaviour inspects this context and emits its best bids.

The produced candidate is shaped like:

```python
ActionCandidate(
    action=FollowField(self._seeder),
    score=score,
    source=self.name,
    target=cell,
)
```

That means it is proposing “follow the explore field toward this target” rather than directly moving the robot.

## What type of object is returned

`propose()` returns:

```python
tuple[ActionCandidate, ...]
```

So the return value is a tuple of zero or more `ActionCandidate` records. It may be:

- `()` when nothing useful is available,
- a tuple with several candidates when many cells are viable,
- a single fallback candidate in the crowd-spread case.

Each `ActionCandidate` contains:

- `action`: usually `FollowField(self._seeder)` in this behaviour,
- `score`: a `float` representing the strength of the bid,
- `source`: the behaviour name (`"explore"`),
- `target`: the cell this candidate wants to move toward.

## How many candidates can be proposed

The class sets:

```python
self._max_candidates = 16
```

This is the upper bound for the candidate list. In `propose()`, it calls `_top_candidates(..., count=self._max_candidates)`, so it will never consider more than 16 cells before scoring.

The returned tuple therefore has a maximum of 16 candidates in the normal path. The only exception is the fallback path, where it may return a single `ActionCandidate` if it has to use `fast_spread()`.

## How the score is generated

The final score is built from three terms:

- visible exploration score,
- separation score,
- dispersion score (used only when the swarm is crowded).

The code does this:

```python
final_scores = [
    self.weight_visible * visible - self.weight_separation * separation
    for visible, separation in zip(visible_scores, separation_scores)
]

if 'dispersion_scores' in locals():
    final_scores = [
        score - self.weight_dispersion * dispersion
        for score, dispersion in zip(final_scores, dispersion_scores)
    ]
```

So the effective formula is:

$$
score_{final} = 0.8 \cdot visible - 0.3 \cdot separation - 0.1 \cdot dispersion
$$

where the dispersion term is only added when the robot is in a crowded area.

### Visible score

`_visible_exploration_score()` looks at a local 3x3 neighbourhood around each candidate cell, because:

```python
self._visible_offset = 1.0
```

and the offsets are generated for cells within one tile in each direction. For each candidate:

- it examines the cells in the local square,
- fetches the field value at each one,
- keeps only finite values,
- sums them,
- divides by the number of offsets in the square.

So the visible score is the average explore-field value seen around the candidate cell. A higher visible score indicates “more promising nearby exploration value” in the local area.

### Separation score

`_separation_score()` computes a penalty for being too close to teammates.

For each candidate, it measures the squared distance to every peer within a radius of roughly $3\sigma$:

```python
max_distance_squared = 9 * sigma_squared
```

where `sigma = self._separation_sigma = 2.0`.

It then computes:

```python
proximity = sum(exp(-distance_squared / (2 * sigma_squared)) for ...) / len(...)
score = 1.0 - proximity
```

This is a Gaussian-like comfort score: if nearby peers are very close, the exponential term is large and the score gets close to 1.0, which is then subtracted from the visible score via the weight `self.weight_separation = 0.3`.

In plain words: the closer a candidate is to other robots, the more it gets penalized.

### Dispersion score

`_dispersion_score()` is used only when the swarm is considered crowded.

It checks:

```python
if peers_num > self._dispersion_robot_threshold:
    dispersion_scores = self._dispersion_score(window, candidate_cells, peers)
```

with:

```python
self._dispersion_robot_threshold = 15
```

This means the behaviour treats the swarm as crowded if there are more than 15 visible peers.

For each candidate cell, it computes the average Euclidean distance to all peers, normalized by the map diagonal:

```python
sum(hypot(cell[0] - x, cell[1] - y) for x, y in peers) / (len(peers) * map_diagonal)
```

This produces a value in roughly $[0, 1]$ that measures how far the candidate is from the swarm. The behaviour then subtracts `0.1 * dispersion`, so a candidate far from peers is rewarded and a candidate too close to the cluster is penalized. The penalty of too many robots nearby will help the swarm to rebalance its composition due to a low exploration score when comparing to other scores. 

## When the swarm is considered crowded

The code treats the swarm as crowded when:

```python
peers_num > self._dispersion_robot_threshold
```

Because the threshold is 15, the dispersion term activates when there are 16 or more peers. This is a simple local indicator that the robot should spread out rather than continue greedily following the strongest local field value.

## How close to other robots will be blocked

The code blocks cells around peers by building a forbidden-cell set:

```python
radius = ceil(self._peer_forbidden_distance)
```

with:

```python
self._peer_forbidden_distance = 1.0
```

It then marks all cells in a square around each peer using Manhattan-like bounds, effectively creating a forbidden area roughly one cell around them. In practice:

- it scans a `3 x 3` area around each peer,
- any cell in that set is skipped,
- the candidate list is built only from cells that are not in those exclusion zones.

This prevents robots from proposing destinations directly on top of or immediately beside each other.

## What `fast_spread` does

`_fast_spread()` is a fallback when the normal top-field search produces no viable candidate. It is only used when there are enough nearby robots:

```python
if len(peers) <= self._dispersion_robot_threshold:
    return None
```

This means it only triggers in crowded situations.

The method then:

- iterates over edge cells of the field window,
- ignores cells that are not traversable,
- computes how many peers are within a radius of roughly `3 * sigma`,
- picks the edge cell with the least nearby traffic.

So `fast_spread` is a simple emergency strategy: if normal exploration is blocked or crowded, choose the least congested boundary cell and propose that as the fallback target.

## Further improvements

The current implementation is a usable heuristic, but there are several natural improvements:

- adapt the thresholds to map size and swarm density instead of fixed constants;
- include obstacle-aware or map-coverage-aware terms for better exploration efficiency;
- add memory of recently visited cells to avoid oscillation or repeated loops;
- make the crowd threshold dynamic rather than fixed at 15 peers.

## Summary

`ExploreBehaviour` is a local decision heuristic: it creates a ranked shortlist of promising exploration cells, avoids cells too close to other robots, blends field value with separation and crowding terms, and returns a tuple of `ActionCandidate` objects for the arbiter.

The behaviour is intentionally simple and modular: it contributes decisions, not direct movement, and leaves the final execution of the move to the rest of the swarm control stack.
