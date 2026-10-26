# Body capability model

This document records the initial high-level simulation profile for each robot body. The values
are simulation assumptions for comparing heterogeneous swarm behaviour, not specifications for a
particular commercial robot.

## Body profiles

| Feature | SCOUT — Explorer drone | KILLER — Ground robot | BATTERY — Support robot |
|---|---|---|---|
| Role | Explores, detects pests/disease, and relays observations | Navigates to known infestations and treats them | Visits low-battery peers and supplies charged replacement batteries |
| Movement | Aerial; can enter cells marked as ground obstacles | Ground; cannot enter obstacle cells | Ground; cannot enter obstacle cells |
| Speed | 8 m/s | 1 m/s | 1 m/s |
| Default cell size in preset scenarios | 10 m, therefore 1.25 s for a straight one-cell move | 10 m, therefore 10 s for a straight one-cell move | 10 m, therefore 10 s for a straight one-cell move |
| Endurance | 30 minutes | 8 hours | 8 hours (initial assumption) |
| Dock recharge duration | 45 minutes | 60 minutes (initial assumption) | 60 minutes (initial assumption) |
| Pest sensing | 3 × 3 cell footprint, including terrain | Current cell only | Does not sense infestations |
| Local obstacle / peer sensing | 3 × 3 footprint | 3 × 3 local ring | 3 × 3 local ring, including peer battery level |
| Treatment | Not permitted | Severity decreases linearly; severity 1.0 takes one simulated hour | Not permitted |
| Battery support | Can receive a replacement | Can receive a replacement | Carries three charged spares; one exchange takes 224 seconds |
| Dock behaviour | Returns when battery is nearly exhausted | Returns when battery is nearly exhausted | Returns when empty of spares or low on energy |

The body profiles live in `src/swarm_field/swarm/capabilities.py`. In particular, `CapabilitySet`
contains the allowed action classes and the physical parameters used by assembly and sensing.

## Capability gate

`ComposedAgent` filters behaviour candidates before comparing scores. A `FollowField` candidate is
checked as a `Move` capability because the navigator resolves it to a concrete move only after the
candidate wins. The resulting concrete action is checked a second time before it leaves the brain.

A configured profile may restrict a body's canonical actions, but cannot grant actions outside its
body type. If no permitted candidate remains, the agent emits `Idle`.

| Action | SCOUT | KILLER | BATTERY |
|---|:---:|:---:|:---:|
| `Move` | Yes | Yes | Yes |
| `Treat` | No | Yes | No |
| `ReplaceBattery` | No | No | Yes |
| `Charge` | Yes, at a dock | Yes, at a dock | Yes, at a dock |
| `SwitchRole` | Yes, to killer | Yes, to scout | No |
| `Idle` | Yes | Yes | Yes |

`SwitchRole` is for swarms that rebalance their roles (#64, [`research/rebalancing.md`](research/rebalancing.md)): the world gives the robot the other body and that body's speed. A scout flies and a killer drives, so a real robot cannot change one airframe into the other; the switch stands for a robot whose role is a module, swapped at a dock with `RebalanceConfig.switch_at_dock`. Robots that may switch plan their routes on the ground whatever their body, so a robot that becomes a killer is never planned across the river.

## Brains for each body

An experiment builds each robot's brain from a `BrainSpec` with `swarm_field.runner.build_agent`, which resolves the behaviours and seeders the spec names from the runner's registries. The registries hold the explore and pest-response parts. `ReturnToDock` and `SupplyBattery`, with their seeders `DockSeeder` and `BatteryRequestSeeder`, are not registered yet, so only a `ComposedAgent` built by hand uses them, as `tests/test_capabilities.py` does.

Each brain owns its own `AgentMemory`, field provider, navigator and capability profile. `WorldState`
produces role-specific observations and validates actions again at the simulation boundary.

## Simulation details and limitations

- Travel reduces the normalized battery level by travel duration divided by the body's endurance.
  This is a coarse movement-energy model; idle and sensing power are not yet simulated.
- A dock is a configured cell in `ScenarioConfig.docks`. Preset scenarios place a dock at `(0, 0)`.
  Charging can only begin on one of those cells.
- Battery replacement begins when support and recipient share a cell, then completes after 224
  simulated seconds if they remain together and inventory is still available.
- Treatment progresses while a killer continuously issues `Treat` for its current cell. Moving,
  idling or choosing another action stops the treatment process.
- Battery support uses recent peer observations from its local sensing ring. There is no dedicated
  request-message protocol, exclusive service reservation, or battery-swap animation yet.
- The current world model does not distinguish different disease-specific treatment methods.
