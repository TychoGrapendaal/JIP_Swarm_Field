# Battery lifecycle (Issue #77)

The battery lifecycle extends the existing `ComposedAgent` / `UtilityArbiter` / potential-field
navigation architecture. It does not add a second decision or navigation stack.

## Battery accounting

Battery level is a fraction in `[0.0, 1.0]`. `EnergySpec` configures energy cost per metre,
per decision, and per treatment action. The world charges a movement's estimated travel cost
before it starts; an unaffordable movement does not partially consume energy and then begin.
Treatment similarly starts only if its configured action cost can be paid. A movement already
under way is allowed to reach its committed destination before off-dock depletion is settled.

A robot at zero battery away from a charging dock becomes `FailureMode.DEAD_BATTERY`. The failed
robot stops normal movement, treatment, replacement and charging work, its battery-failure mode is
recorded, and it no longer schedules normal decisions or appears in peer observations. Other robots
and the event loop continue. A zero-battery robot physically at a configured dock can wait for a
charger and recover.

A robot away from a dock whose battery cannot pay for even one straight move is stranded: it never
moves again, and when decisions cost nothing its battery never drains to zero either. It fails in
the same way unless a live BATTERY robot still carries a spare that could reach it. Without this
rule a stranded robot would idle forever while still counted in `active_robots`.

Every `SwarmScenario` has a dock at `(0, 0)`, and its explorers and seeded killers have `return-to-dock`, so they recharge instead of running flat. `MapConfig.docks` moves it, adds more, or removes them all. Hop killers keep no map, so they follow the DOCK beacons instead: in a swarm that relays beacons, every dock broadcasts a DOCK beacon of 0 on its cell (`DockBroadcast`), every robot relays its own DOCK value, and a hop killer walks down them on `HopDockField`, with the same `ReturnToDock` and `Resupply` behaviours. The catalogue's presets have the dock and `return-to-dock` too.

## Dock charging and restocking

A `Charge` action is effective only at a configured dock. A dock serves every robot that asks at once, with no queue. It simulates a battery swap: the battery charges with simulated time, from empty to full in `CapabilitySet.dock_recharge_seconds`, 5 minutes for every body (`DOCK_SWAP_SECONDS`). A robot that leaves the dock or fails stops charging.

The service also swaps what can be swapped at once. A KILLER's treatment consumable is full again as soon as it asks a dock for service, even with a full battery, and a BATTERY robot's spare batteries are replenished. Battery replacement between a BATTERY robot and a co-located SCOUT/KILLER remains a separate timed operation.

The small ready fields have one dock at `(0, 0)`, and `RIVERS_LARGE` has two, at `(0, 0)` and `(63, 63)`.

## Restocking

`swarm_field.swarm.Resupply` returns a killer to a dock when its consumable runs low, configured by `ResupplyConfig`. At or below `low_consumable` (0.2, two doses of ten) it bids to follow the dock field, from `low_priority` (0.6, below a treat bid, so a killer with doses left treats first) up to `empty_priority` (0.95, an override) when empty. On a dock it bids `Charge` at `dock_priority`, which restocks at once. A killer with no consumable left proposes no treatment, so it neither stands on a pest it cannot treat nor holds it as a claim. `SeededKillerPreset` carries `Resupply` on the `DockSeeder`, and `HopKillerPreset` on `HopDockField`.

## Recharge policy

`swarm_field.swarm.RechargeConfig` configures the current return-to-dock policy:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `low_battery_threshold` | `0.25` | Begin prioritising the dock below this level |
| `critical_battery_threshold` | `0.10` | Critical return priority at or below this level |
| `low_priority` | `0.91` | Score used at the low threshold |
| `critical_priority` | `1.0` | Score used at the critical threshold |
| `no_spares_priority` | `0.95` | BATTERY robot return priority with no spares |
| `dock_priority` | `1.0` | Priority when a robot at the dock requests service |

For custom agent assembly, pass an instance to `ReturnToDock(..., recharge_config=...)`. With the
registry-based experiment runner, put a `RechargeConfig` instance in `BrainSpec.params["recharge"]`,
select the `"return-to-dock"` behaviour, and include the `"dock"` seeder. The runner passes the
configured scenario docks to both components. Behaviour scores are still selected by the existing
`UtilityArbiter`; the potential field is built only after a movement intent wins. The
`DecisionContext` does not use a potential field to score behaviours.

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
