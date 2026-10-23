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

`PestScenario`, `ExploreScenario` and the presets have a dock at `(0, 0)` and give every robot
`return-to-dock`, so robots recharge instead of running flat. `PestScenario.docks` and
`ExploreScenario.docks` move it, add more, or remove them all.

## Dock charging

A `Charge` action is effective only at a configured dock. Charging progresses in simulation time
according to the robot body's `CapabilitySet.dock_recharge_seconds`; the duration is the time for a
full charge from empty. Each dock has one active charging slot. Further requests are queued in FIFO
order of requests processed by the deterministic simulation event queue.
When the active robot completes charging, leaves, or fails, the next eligible waiter can use the
slot. Waiting robots do not gain charge while queued.

BATTERY robots replenish their spare-battery inventory when they request dock service. They still
use their own body-specific charge rate, independently of their spare inventory. Battery replacement
between a BATTERY robot and a co-located SCOUT/KILLER remains a separate timed operation.

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

They cover exclusive charging and queue hand-off, recovery at a dock from zero battery, off-dock
failure and metrics, the rejection of off-dock charging, unaffordable movement, battery bounds, the
PR #61 numeric-default feedback, and recharge-policy validation.
