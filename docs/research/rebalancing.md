# Role rebalancing: when, and how many, robots switch between monitor and killer

This covers #64: how the swarm decides, without a central controller, how many robots monitor the field (scouts) and how many treat pests (killers), and when a robot changes from one role to the other. The code is in `swarm/rebalance.py`; the simulation side is the `SwitchRole` action in `contracts/actions.py` and `WorldState._switch_role` in `simulation/world.py`. The results below come from `examples/arbiter_demo.py --rebalance --report` and are reproducible from seed 0.

## What it has to solve

Monitors are fast (1.5 m/s in the scenario, against 0.5 m/s for killers), so one monitor keeps more of the field fresh than a killer could. Killers are only worth having while there are pests to treat. With bodies fixed when a run is assembled, the split is a guess made before the first pest appears: too many killers and the field goes stale, too few and pests pile up. Rebalancing lets that split follow the pests.

Three questions have to be answered at every decision point: **how many** killers the swarm needs, **when** an imbalance is real enough to act on, and **who** switches, so that a shortage of two killers produces two switches and not ten.

## How many

Every robot counts, from its own memory, the monitors, the killers and the pests it knows of. Peers report their body in every status message, and the robot's own body comes from its perception, so the count needs nothing new on the wire. Peers last heard of longer ago than `peer_horizon` are left out, so a robot that has gone silent stops counting.

The swarm needs `min_killers` plus one killer for every `pests_per_killer` known pests, but never so many that fewer than `min_monitors` monitors are left:

```
killers_needed = min(min_killers + ceil(pests / pests_per_killer), max(min_killers, robots - min_monitors))
```

`min_monitors` comes from the field: the monitors needed to sense every free cell once per revisit age. A monitor senses at most one new cell per decision, and no faster than it can drive across one, so `ArbiterScenario.monitors_needed` takes the slower of the two. On the 32 × 32 river-and-rocks field that is about 950 free cells, a 28 min revisit age and 6 cells per minute per monitor (capped by the 10 s decision interval rather than by speed), so 6 monitors. It ignores overlap, so it is a lower bound.

## When

The same idea as the arbiter's hysteresis, for the same reason: news arrives in bursts when robots meet, and a robot should not change role at the rate its map changes.

- **Persistence.** An imbalance in the same direction has to last `persistence` (2 min) before anyone acts on it. A single encounter that briefly shows a shortage does nothing.
- **Dwell.** A robot keeps a new role for `role_dwell` (15 min) before it may switch again. This bounds switches to four per robot per hour, whatever the pests do.
- **Cost.** A switch keeps the robot busy for `switch_duration` (60 s): it decides, but does not act.

The switch is bid at `switch_score` (0.95), above the arbiter's override threshold. A lower bid would hardly ever win: an explore commitment holds at the maximum margin of 0.5 after five minutes (#60), so a rival needs more than 1.0. Treating the pest underfoot (0.99) and charging on a dock (1.0) still outbid it, so a robot finishes those first. With `switch_at_dock`, a robot only switches while it stands on a dock, so the switch can ride along with a recharge stop.

## Who

Every robot sees roughly the same imbalance. If the rule were "killers are short, so switch", every monitor would switch at once and the swarm would overshoot. Instead each robot ranks itself among the robots that could switch and only acts if it is among the first `|gap|`:

- monitor to killer: nearest a known pest first;
- killer to monitor: furthest from any known pest first;
- ties, and the case with no known pest, go to the lowest robot id.

Robots rank the same peers from slightly different maps, so two may occasionally both decide they are the one. Persistence and dwell absorb that: the next count shows the overshoot, and nobody can switch back for 15 minutes.

```mermaid
flowchart TD
    count[Count monitors, killers, pests] --> need[killers_needed vs killers known]
    need -->|balanced| none[No bid]
    need -->|gap for less than persistence| none
    need -->|gap has lasted| dwellcheck{Role kept for role_dwell?}
    dwellcheck -->|no| none
    dwellcheck -->|yes| rank{Among the first gap robots in the ranking?}
    rank -->|no| none
    rank -->|yes| bid[Bid SwitchRole at 0.95]
    bid --> world[World changes body and speed, busy for switch_duration]
```

## One brain for both roles

A robot that can switch carries the behaviours of both roles: `TreatInfestation`, the explore behaviour, `ReturnToDock` and `RebalanceRoles`. `BodyGate` wraps `TreatInfestation` so it only bids while the robot is a killer; the body is read from the perception at every decision, so the gate follows a switch at once. The runner applies a gate to every behaviour named under `"body_gates"` in a `BrainSpec`'s parameters. The world changes the robot's body and sets its speed from `RoleSwitchSpec.speeds`; the renderer colours robots by body, so a switch is visible in the viewer.

The capability gate of #14 (PR #61) does the same job more generally, by action type and from the same perceived body. Once it is merged, `BodyGate` can go and `SwitchRole` only needs adding to the killer's and scout's capability sets.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `min_killers` | 1 | killers kept with no known pest, so a new one is treated quickly |
| `min_monitors` | 6 (from the field) | monitors kept whatever the pests |
| `pests_per_killer` | 2 | known pests one killer is expected to handle |
| `persistence` | 2 min | how long an imbalance must last before anyone acts |
| `role_dwell` | 15 min | how long a robot keeps a new role |
| `peer_horizon` | 20 min | peers heard of longer ago are not counted |
| `switch_score` | 0.95 | the bid to switch, an override |
| `switch_at_dock` | off | switch only while on a dock |
| `switch_duration` | 60 s | how long a switch keeps a robot from acting (scenario) |

Like the arbiter's margins, these are first guesses.

## Results

Fourteen robots (8 monitors and 6 killers at the start) on the 32 × 32 river-and-rocks field with four docks, 20 pests an hour, 300 m radio, seed 0, three simulated hours. Both runs are the same scenario; the only difference is whether robots may switch.

| After 3 h | Fixed roles | Rebalancing |
|---|---|---|
| Mean infested cells over the run | 8.6 | 7.4 |
| Infested cells at the end | 7 | 6 |
| Killers | always 6 | 1 to 6, about 2.5 on average |
| Role switches | 0 | 30 |
| Arbiter switches per completed job | 0.98 | 0.83 |
| Lowest battery / robots stranded | 0.11 / 0 | 0.11 / 0 |

| Minute | 0 | 15 | 30 | 45 | 60 | 75 | 90 | 105 | 120 | 135 | 150 | 165 | 180 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Killers | 6 | 2 | 1 | 3 | 4 | 2 | 2 | 2 | 2 | 3 | 3 | 4 | 2 |
| Infested cells | 1 | 3 | 4 | 8 | 9 | 9 | 7 | 7 | 6 | 8 | 9 | 10 | 6 |

The swarm starts by giving up most of its killers while there is little to treat, takes them back as pests pile up (45 to 60 min, and again from 135 min) and lets them go again once the pests are down. It keeps fewer cells infested than the fixed split while using fewer killers. This is one seed and one scenario, so it is a first signal rather than a finding; a sweep over seeds and pest rates is the next step.

## What it does not do yet

- **It only counts the pests it knows of.** From 75 to 120 min the killers stay at 2 while 6 to 9 cells are infested, because most robots have heard of only a few of them. Robots can also count a pest that a peer has already treated until they hear otherwise. The estimate is only as good as the swarm's shared map.
- **It does not weigh where pests are.** A cluster of pests far from every killer needs more killers than the same number next to one; `pests_per_killer` is a flat rate.
- **The ranking ignores the battery.** A robot about to recharge may still be the one chosen to switch.
- **It does not interact with crowding yet.** Two new killers can head for the same pest; that is the crowding problem of #42 and the TREAT seeder's peer discount, not of rebalancing.

## Using it

```bash
uv run python examples/arbiter_demo.py --rebalance
uv run python examples/arbiter_demo.py --rebalance --switch-at-dock
uv run python examples/arbiter_demo.py --rebalance --report --hours 3 --out runs/rebalance.json
```

Without `--rebalance` the same scenario runs with fixed roles, which is the baseline above. In code, pass a `RebalanceConfig` as `ArbiterScenario.rebalance`.
