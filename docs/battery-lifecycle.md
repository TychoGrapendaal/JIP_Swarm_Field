# Battery lifecycle (Issue #77)

The battery lifecycle extends the existing `ComposedAgent` / `UtilityArbiter` / potential-field
navigation architecture. It does not add a second decision or navigation stack.

## Battery accounting

Battery level is a fraction in `[0.0, 1.0]`. `EnergySpec` configures energy cost per metre,
per decision, and per treatment action. The world charges a movement's estimated travel cost
before it starts; an unaffordable movement does not partially consume energy and then begin.
Treatment similarly starts only if its configured action cost can be paid. A movement already
under way is allowed to reach its committed destination before off-dock depletion is settled.

A robot at zero battery away from a charging dock becomes `FailureMode.DEAD_BATTERY`. The failed robot stops normal movement, treatment, replacement and charging work, its battery-failure mode is recorded, and it no longer schedules normal decisions or appears in peer observations. Other robots and the event loop continue. A zero-battery robot physically at a configured dock charges and recovers; a dock has no queue.

A robot away from a dock whose battery cannot pay for even one straight move is stranded: it never
moves again, and when decisions cost nothing its battery never drains to zero either. It fails in
the same way unless a live BATTERY robot still carries a spare that could reach it. Without this
rule a stranded robot would idle forever while still counted in `active_robots`.

Every `SwarmScenario` has a dock at `(0, 0)`, and its explorers and seeded killers have `return-to-dock`, so they recharge instead of running flat. `MapConfig.docks` moves it, adds more, or removes them all. Hop killers keep no map, so they follow the DOCK beacons instead: in a swarm that relays beacons, every dock broadcasts a DOCK beacon of 0 on its cell (`DockBroadcast`), every robot relays its own DOCK value, and a hop killer walks down them on `HopDockField`, with the same `ReturnToDock` and `Resupply` behaviours. The catalogue's presets have the dock and `return-to-dock` too.

## Dock charging and restocking

A `Charge` action is effective only at a configured dock. A dock serves every robot that asks at once, with no queue. It simulates a battery swap: the battery charges with simulated time, from empty to full in `CapabilitySet.dock_recharge_seconds`, 5 minutes for every body (`DOCK_SWAP_SECONDS`). A robot that leaves the dock or fails stops charging.

The service also swaps what can be swapped at once. A KILLER's treatment consumable is full again as soon as it asks a dock for service, even with a full battery, and a BATTERY robot's spare batteries are replenished. Battery replacement between a BATTERY robot and a co-located SCOUT/KILLER remains a separate timed operation.

`MapConfig` has one dock at `(0, 0)` by default, which `PESTS`, `EXPLORE` and `RELAYED_KILLERS` keep. `RIVER_SMALL` has one on each bank of its river, at `(0, 0)` and `(31, 31)`, and `RIVERS_LARGE` two, at `(0, 0)` and `(63, 63)`.

## The `Resupply` behaviour

`swarm_field.swarm.Resupply` returns a killer to a dock when its doses run low, configured by `ResupplyConfig`. At or below `low_doses` doses left it bids to follow the dock field, from `low_priority` (0.6) at `low_doses` up to `empty_priority` (0.95, an override) with no dose left. On a dock it bids `Charge` at `dock_priority`, which restocks at once. A full tank never bids, however small. The threshold is a number of doses rather than a share of the tank, so it means the same with any tank size. A killer holds the dose of a treatment under way until the treatment ends, so it is not empty while it treats with its last dose. A killer with no dose left proposes no treatment, so it neither stands on a pest it cannot treat nor holds it as a claim.

The two arms restock by different rules, because they bid to treat on different scales. `SeededKillerPreset` carries `Resupply` on the `DockSeeder` with the default `low_doses` of 2. Its treat bid is 0.8 to 0.89, above the restock bid of 0.6 to 0.78 with doses left, so a seeded killer treats the pests it knows of first and heads for the dock before it runs dry. `HopKillerPreset` carries `Resupply` on `HopDockField` with `low_doses` of 0, so it restocks only an empty tank. Its treat bid is the strength of a beacon, often 0.02 to 0.5, so a restock bid of 0.6 would send a hop killer with doses left to the dock, and once it was committed, a treat bid would need 0.6 plus the arbiter's margin to win it back.

## Recharge policy

`swarm_field.swarm.RechargeConfig` configures the current return-to-dock policy. A robot charges on a dock only when it came to charge: `ReturnToDock` bids `Charge` there at or below `low_battery_threshold`, and once the robot has started, it keeps bidding until the battery is full or the robot leaves the dock. A robot that only passes a dock, with a battery that is merely not quite full, does not stop, since a `Charge` starts no move and would hold it for a whole decision interval. A BATTERY robot short of spares still bids `Charge` on a dock, which refills them.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `low_battery_threshold` | `0.25` | Begin prioritising the dock below this level |
| `critical_battery_threshold` | `0.10` | Critical return priority at or below this level |
| `low_priority` | `0.91` | Score used at the low threshold |
| `critical_priority` | `1.0` | Score used at the critical threshold |
| `no_spares_priority` | `0.95` | BATTERY robot return priority with no spares |
| `dock_priority` | `1.0` | Priority of `Charge` for a robot on a dock that came to charge |

For custom agent assembly, pass an instance to `ReturnToDock(..., recharge_config=...)`. With the registry-based experiment runner, put a `RechargeConfig` instance in `BrainSpec.params["recharge"]`, select the `"return-to-dock"` behaviour, and include the `dock` seeder, or the `dock-hop` field for a brain without a map. The runner passes the configured scenario docks to both components. Behaviour scores are still selected by the existing `UtilityArbiter`; the potential field is built only after a movement intent wins. The `DecisionContext` does not use a potential field to score behaviours.

## Failure metrics

`MetricsSample.battery_failures` reports the cumulative number of robots that failed due to battery
depletion. `active_robots` also decreases when a robot fails. The failure count is available in the
normal metrics series and CSV/JSON exports.

## Visual demo

The styled live field demo uses the same simulation and agent architecture while showing dock
charging, low-battery support, infestations, potential fields, communication and telemetry in one
window. Run it from the repository root:

```bash
uv run --with matplotlib python examples/field_simulation_demo.py
```

It opens an interactive Matplotlib window and does not save a GIF or video. See
[`live-field-simulation-demo.md`](live-field-simulation-demo.md) for controls and architecture details.

## Tests

Run the focused acceptance tests with:

```bash
uv run pytest tests/test_battery_lifecycle.py
```

They cover charging several robots at once, restocking, recovery at a dock from zero battery, off-dock
failure and metrics, the rejection of off-dock charging, unaffordable movement, battery bounds, the
PR #61 numeric-default feedback, and recharge-policy validation.
