# Research note — hop-based pest navigation for the killers

Status: describes two methods, both selectable in `HopPestScenario` (`src/swarm_field/experiments/hop.py`) next to the memory-sharing baseline:

- **Hop-limited beacons** (`navigation="hop"`, `src/swarm_field/swarm/hop.py`, contract `PestBeacon`): news of a pest is relayed hop by hop and carries the pest's cell. The hop count only ranks the pests; the killer drives to the cell in the beacon. This is hop-based *dissemination*, not hop-based navigation.
- **Hop-gradient navigation** (`navigation="gradient"`, `src/swarm_field/swarm/gradient.py`, contracts `HopGradient` and `HopClaim`): no message carries a position. The killer drives from neighbour to neighbour down the hop counts until it stands on the pest. This is hop-based navigation proper.

Both use `PestId` (`src/swarm_field/contracts/ids.py`) as the target of the killer's bid. `examples/hop_explainer.py` shows either one with every robot's hop count drawn.

## The problem

With memory sharing every robot broadcasts its whole memory snapshot, and a killer seeds the TREAT field with every pest it knows of. The snapshot grows with the field, every pest is believed with an absolute cell that may be old, and nothing tells a killer which of many pests is near *through the swarm*.

## Method 1: hop-limited beacons

News of a pest travels as a small `PestBeacon`: the pest's id, its believed cell, a hop count, when it was found, and whether it is known cleared.

- **Start.** A robot that senses a pest starts a beacon at hop 0, with a new `PestId(finder, number)`, unless it already has a live beacon for that cell.
- **Relay.** Every robot that hears a beacon keeps it one hop further than the sender had it, and passes it on. Per pest it keeps the fewest hops it has ever heard. Beacons are not passed on beyond `max_hops` (20) or after `lifetime` (30 min), and at most `max_beacons` (12) go in one message.
- **Clear.** A robot that senses a cell clean marks the live beacons of that cell cleared. A cleared beacon wins wherever it arrives and is never undone.
- **Choose.** A killer follows the live beacon with the fewest hops, skipping pests whose cell another killer recently reported heading for.
- **Go.** `BeaconSeeder` seeds the potential field at the pest's cell from the beacon, and `FollowBeacon` bids `0.8 − 0.01 · hops` (never below 0.65), or treats on the spot.

What this is not: the hop count is the fewest hops ever heard, never refreshed, so it says how fast the news reached a robot rather than how far the pest is now. News also travels by robots driving with it (a scout can carry a beacon for many minutes before it meets anyone), so "1 hop" can be far away. The killer's route comes from the cell in the beacon and its own map.

## Method 2: hop-gradient navigation

The robots form a gradient of hop counts that points to the pest, and the killer descends it.

- **Anchor.** Robots sense only their own cell, so a robot that senses a pest stands on it. It becomes the pest's anchor: at every decision it sends `HopGradient(pest, round, hops=0)` with a new round. A scout anchor stays on the pest (`AnchorPest`, bid 0.95, above the arbiter's override threshold) for up to `anchor_time` (15 min), or up to `claimed_anchor_time` (45 min) while a killer has claimed the pest, so a killer already on its way does not arrive to find the anchor gone; then it explores again. A robot that lands on a pest its parent anchors from the same cell takes over that pest's id instead of starting a new one, so the killer that treats it clears the pest everyone was following, and two scouts on one cell do not report it twice.
- **Relay.** A robot that hears a count keeps the newest round it has heard and, within a round, the fewest hops, and remembers the neighbour it heard that count from: its **parent**. It passes on its own count, one more. Because a new round replaces an old count whatever its hops, the counts follow the swarm as robots move.
- **Fade.** A count not refreshed for `max_age` (2 min) is dropped. Only a round newer than any heard before is accepted, so once the anchor leaves, robots cannot learn the dead count back from each other, one hop more each time ("count to infinity"); without this rule counts climbed to 20 in the first test run.
- **Descend.** A killer picks the pest with the fewest hops that no other killer claims, and drives to its parent's current cell, which it knows because neighbours in radio range report their cells. As it closes in it hears a robot one hop nearer, which becomes its parent. Next to the anchor its parent is the anchor itself, standing on the pest, so the killer arrives on the pest, senses it and treats it. The potential field is used only to drive to the parent around obstacles.
- **Claim.** A killer following a gradient sends a `HopClaim` with its hop count, relayed like the counts. A killer leaves a pest to another killer with fewer hops to it (ties to the lower id) and takes the next pest it knows; every killer applies the same rule to the same claims, so of two killers on one pest exactly one gives way. When a killer knows no other live pest it still follows the claimed one.
- **Clear.** The robot that senses the anchored cell clean sends a cleared count, which wins wherever it arrives.

No message carries the pest's position, and no robot but the anchor knows where the pest is. Every rule is local, so this is a swarm gradient in the usual sense, like the hop-count gradients of amorphous computing.

## What the arbiter needs

The arbiter itself is unchanged. What it needs is a target that stays the same for the length of a trip. Both killer behaviours bid with the pest's `PestId` as the target, so:

- the effort of a trip to a pest builds up and protects it (the margin rule), even while the cell estimate or the parent changes;
- PR #65's rule that a commitment without a target builds no effort keeps applying to exploring (#60);
- the brain still reports a cell to peers, not the `PestId`.

`Target = CellId | PestId` is the type of `ActionCandidate.target` and `Commitment.target`; the arbiter compares targets for equality only.

## Comparison

`examples/hop_demo.py --compare` runs the pest experiment (5 scouts, 3 killers, 32 × 32 cells of 10 m, 15 pests per hour, killers at 0.5 m/s, 3 h) with each method, at a short (50 m) and a long (300 m) radio range. Per run, seeds 0, 1 and 2 (pests spawned: 37, 49, 37):

| Radio | Navigation | Cleared | Left at 3 h | Mean infested cells | Killer time heading for or treating a pest |
| --- | --- | --- | --- | --- | --- |
| 300 m | gradient | 19, 29, 23 | 18, 20, 14 | 8.9, 12.7, 12.0 | 32%, 74%, 45% |
| 300 m | hop | 22, 33, 25 | 15, 16, 12 | 7.9, 10.0, 9.7 | 49%, 63%, 41% |
| 300 m | memory | 18, 28, 23 | 19, 21, 14 | 9.7, 14.7, 11.5 | 21%, 30%, 17% |
| 50 m | gradient | 18, 24, 19 | 19, 25, 18 | 9.9, 15.2, 11.9 | 8%, 22%, 8% |
| 50 m | hop | 23, 27, 20 | 14, 22, 17 | 9.1, 11.2, 10.6 | 26%, 50%, 31% |
| 50 m | memory | 27, 31, 16 | 10, 18, 21 | 7.7, 12.2, 12.6 | 24%, 49%, 10% |

Totals over the three seeds:

| Radio | Gradient cleared / mean infested | Hop cleared / mean infested | Memory cleared / mean infested |
| --- | --- | --- | --- |
| 300 m | 71 / 11.2 | 80 / 9.2 | 69 / 12.0 |
| 50 m | 61 / 12.3 | 70 / 10.3 | 74 / 10.8 |

The rows for the gradient above were run before the anchor wait, the pest-id takeover and the hop-count claims were added; the single-killer numbers below too.

With one killer, six scouts and a 70 m radio (`examples/hop_explainer.py`, 1.5 h, seeds 0–4), the gradient got news to the killer faster in every seed where news reached it at all, four of the five (seed 2: 1.7 min against 6.1 min after sensing; seed 4: 0.5 against 1.8), because the anchor refreshes it every round instead of waiting for robots to meet. It cleared fewer pests in total: 14 treatments against 17 for the beacons, all of the gap in one seed (4 against 8).

With three killers, ten scouts, a 70 m radio and 10 pests per hour (`examples/hop_explainer.py --scouts 10 --killers 3 --pests-per-hour 10`, 1.5 h, seeds 0–3, pests appearing: 9, 21, 14, 16), the killers treated 23 pests with the gradient (2, 5, 8, 8) against 36 with the beacons (6, 14, 8, 8). Before the three changes above the gradient managed 18. Making counts live 4 minutes instead of 2, so a killer keeps driving towards its last parent longer, made it worse (18), so it stays at 2. Part of that gap was not the gradient's: four to six scouts stood together for the whole run and moved as one (see "Robots standing together" in `docs/explore.md`). With that fixed, the gradient's killers treated 35 pests over the same seeds (5, 13, 10, 7), as many as with memory sharing (35), and the mean infested cells fell from 5.0 to 3.9; the share of time scouts stood together fell from 36% to 9%. The beacon numbers above are from before the fix. Before the fix, the event logs showed why the gradient trailed: scouts move about 9 cells a minute against a 7-cell radio, so a chain from a killer to an anchor lasts a minute or two, and anchors far from every killer wait with nobody coming.

## What the numbers say

- **Hop-limited beacons clear the most pests** at 300 m and stay level with memory sharing at 50 m, but much of that is the fixed 0.8 bid and the spreading of killers over pests, not the hops: `TreatInfestation` scores `1 / (1 + e^-(pests − killers))`, only 0.12 with one known pest and three known killers, so memory killers keep exploring.
- **The hop gradient is the only coordinate-free method, and it pays for it with connectivity.** A killer can descend a gradient only while an unbroken chain of robots joins it to the anchor. With 5–7 robots on a 320 m field a 50 m radio almost never makes such a chain: killers spent only 8–22% of their time going to pests, against 26–50% with beacons, which keep the cell after the chain breaks. At 300 m chains form and the gradient beats memory sharing on mean infested cells in two of three seeds.
- **Anchoring costs exploring.** A scout that waits up to 15 minutes on a pest is not finding new ones; with few scouts that is a real share of the search.

So for this small, sparse swarm the beacons are the practical choice. The gradient is the one that would keep working without positions (no GPS, drifting odometry) and that improves as the swarm gets denser, since more robots mean unbroken chains; testing it with more robots or a longer radio is the next experiment. Three seeds of one field are a small sample.

## Limits of both

- Pests are found only by the robot standing on them, so detection by the scouts limits both methods.
- Both send at most a dozen small records per message whatever the field size; what grows with density is the number of messages on the channel.

## Keeping in contact, and one anchor per pest

`HopPestScenario.keep_connected` (off by default, so the three navigations compare as before; `examples/hop_explainer.py` turns it on) keeps every robot in radio contact with at least one other (`swarm/tether.py`). Each step is checked before it is taken: it is safe if from the new cell some peer stays within the radio range less a 1.5-cell margin (the peer may step away too before its next report), and if no anchor that only this robot connects is left behind, since an anchor cannot follow. An unsafe step is replaced by the safe neighbouring cell lowest on the robot's own potential field, so it still follows the quadtree route around obstacles, or by staying. A robot that has heard no peer for 30 s heads back to where it last heard one, for up to 2 minutes; past that the peers have moved on, and it goes about its work until it meets one (waiting at the old spot stranded a killer for half an hour). Positions come from memory: the peers' own reports and the ones relayed in memory snapshots. So that robots are still pushed to explore, the explore field then prefers the stale cells within that range of some peer (`ExploreConfig.reach_cells`): a robot always has a target it can reach without losing contact, the group spreads out together, and when nothing in reach is stale the far targets drag the group along. A first version that also refused to leave any peer with no other neighbour froze the swarm (coverage fell to 42%), so only anchors are protected that way.

Each pest has one anchor. A robot already anchoring keeps the pest against one that has just arrived; between robots that arrived together, or that both anchor already, the one first in the order of `swarm/priority.py` keeps it, and the other leaves. Robots on the same cell are sensed at once, so robots arriving together settle it at their first decision. The anchor bids 0.85: above exploring, but below the arbiter's override threshold, so a need that must not be traded off, such as a battery about to run out, still takes it away.

Ties (who gives way when exploring, who keeps a pest, which of two equally near killers takes it) go by `swarm/priority.py`: a hash of the robot's id and a 10-minute time window. Every robot computes the same order, so they agree without talking, but the order is reshuffled every window, so the same low-id robots do not always win and always carry the waiting.

Measuring this found one more bug in the memory fix of `docs/explore.md`: a robot senses before it reads its messages, so a peer's report always arrived older than the sighting of it and was thrown away, and a robot never learned what a peer on its cell was doing. Two anchors sat on one pest for 18 minutes because of it. Memory now keeps each peer's newest report apart from its newest position.

Over four seeds with ten scouts, three killers, a 70 m radio and 10 pests per hour (1.5 h), after all of the above:

| Navigation | Contact | Time a robot is out of contact | Longest alone | Coverage | Mean idleness | Treated | Mean infested | Most anchors on one pest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient | off | 20% | 20.5 min | 100% | 17.2 min | 37 | 3.8 | 2, for 30 s at most |
| gradient | on | 0.4% | 5.2 min | 100% | 18.9 min | 32 | 3.9 | 2, for 30 s at most |
| memory | on | 0.7% | 12.5 min | 100% | 17.9 min | 37 | 4.3 | – |

Keeping contact costs the gradient some treatments (37 to 32) but hardly any infested cells (3.8 to 3.9), and cells wait under 2 minutes longer between visits. Memory sharing with contact now treats as many pests as the gradient without it; before the report fix it managed 22, because robots did not know where the others were heading. Two anchors still meet on one pest for up to 30 s, when robots arrive together and their reports lag. Contact is still lost now and then, at most for 12.5 minutes in one run, because positions are known only from reports up to 30 s old, robots decide one after another, and a robot lost for more than 2 minutes looks for others by exploring.

Every robot keeping a neighbour does not make the swarm one connected network: it can split into groups that each hold together. An anchor in a group without a killer then waits with nobody coming. Keeping the whole swarm connected would need robots to know more than their neighbours, such as how many hops away the nearest killer is, which the gradient could carry.
