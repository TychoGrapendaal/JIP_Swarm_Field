# Live field simulation demo

`examples/field_simulation_demo.py` is an interactive visual front end for the repository's current integrated simulation. It borrows the layout of the supplied field video: a large field and potential map on the left, and live telemetry, robot state, communication and recent events on the right. The demo does not save a GIF or video.

## Run it

From the repository root:

```bash
uv run --with matplotlib python examples/field_simulation_demo.py
```

Use `space` or `p` to pause/resume, `s` to process one event, `+` / `-` to change the simulation speed, `0` to return to 1x, and `h` to hide/show telemetry. Click a robot to view the potential field it currently uses; click an empty walkable cell to inject an infestation into the live world.

For headless checking or documentation, write one PNG preview (not an animation):

```bash
uv run --with matplotlib python examples/field_simulation_demo.py --preview docs/media/field_simulation_demo_preview.png --preview-minutes 15
```

## Architecture used

The view is an observer over a `LiveRun` created by `swarm_field.runner`. The scenario is assembled using `ExperimentConfig` and the runner registries. Movement and actions still flow through the repository's actual components:

1. `SimulationLoop` schedules and processes world, decision, and metrics events.
2. Each robot uses `ComposedAgent` with its body-specific registered behaviours and `UtilityArbiter`.
3. The chosen movement intent names a seeder registered in the agent's `FieldProvider`. The existing potential-field implementation and navigator select the next cell.
4. Local perception, agent memory, status/knowledge messages and the range-based communication channel remain the sources of robot knowledge.
5. The existing world model executes movement, treatment, charging, battery replacement and battery lifecycle transitions.

The visualization queries the selected agent's field from its existing `FieldProvider`; it does not calculate replacement paths or issue movement actions of its own. The right panel observes simulation state, metrics, current objectives and events. Radio links are drawn from current positions and configured ranges to make the communication topology visible.

## Scenario shown

The deterministic scenario has 10 SCOUT robots, 3 KILLER robots and 1 BATTERY support robot; a 40 × 30 cell field; two docks; river-like obstacle cells with crossings; several initial infestation clusters; and enabled pest dynamics. One SCOUT and one KILLER start at low battery so the run can demonstrate both dock return and peer battery support. The BATTERY robot starts alongside the low-battery KILLER to make the timed battery exchange easy to observe.

The bottom colorbar maps the currently focused robot's actual potential values. In the repository, lower field values are more attractive. The radio circle represents the focused robot's configured radio range, while connecting lines show robot pairs within their configured radio ranges. Robot positions are slightly offset for display only when several robots occupy the same cell; the underlying world positions are not changed.
