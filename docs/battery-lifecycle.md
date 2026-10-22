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

## Dock charging

A `Charge` action is effective only at a configured dock. Charging progresses in simulation time
according to the robot body's `CapabilitySet.dock_recharge_seconds`; the duration is the time for a
full charge from empty. Each dock has one active charging slot. Further requests are queued in FIFO
order, with robot ID as a deterministic tie-breaker for requests at the same simulation timestamp.
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

Pass an instance through the body-specific `build_agent(..., recharge_config=...)` constructor to
tune the policy. The registry-based experiment runner can opt in with a `BrainSpec` containing the
`"return-to-dock"` behaviour and `"dock"` seeder; the runner passes the configured scenario docks
to both components. Behaviour scores are still selected by the existing `UtilityArbiter`; the
potential field is built only after a movement intent wins. The `DecisionContext` does not use a
potential field to score behaviours.

## Failure metrics

`MetricsSample.battery_failures` reports the cumulative number of robots that failed due to battery
depletion. `active_robots` also decreases when a robot fails. The failure count is available in the
normal metrics series and CSV/JSON exports.

## Visual demo

From the repository root, run:

```bash
uv run --with matplotlib python examples/battery_lifecycle_demo.py
```

The script uses the normal simulation loop and agent assembly, then plays the captured frames in an
interactive Matplotlib window. It does not write a GIF or other animation file. The animation shows
a configured dock, progressive charging, a waiting queue, BATTERY spare replenishment, and a KILLER
that becomes `DEAD_BATTERY` away from the dock. The frame capture reads simulation state; it does not
change agent decisions or world transitions. Close the Matplotlib window to finish the demo.

## Tests

Run the focused acceptance tests with:

```bash
uv run pytest tests/test_battery_lifecycle.py
```

They cover exclusive charging and queue hand-off, recovery at a dock from zero battery, off-dock
failure and metrics, the rejection of off-dock charging, unaffordable movement, battery bounds, the
PR #61 numeric-default feedback, and recharge-policy validation.
