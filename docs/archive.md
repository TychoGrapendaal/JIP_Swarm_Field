# Archive

Files deleted from the repository, and why. Each one can still be read from the commit before its deletion, for example `git show 1335f0f:examples/renderer_demo.py`.

## Cleanup of branch `70-hop-field`

| File | Last commit with it | Why it was deleted |
| --- | --- | --- |
| `update.md` | `1335f0f` | An assistant's report on the PR that added body capabilities. `docs/body-capabilities.md` describes the capability model, and git history keeps the rest. |
| `examples/mass_swarm_visualization.py` | `1335f0f` | A prototype from before the robot brain existed: hundreds of robots under a potential-field controller of its own. The brain in `swarm/` replaced that controller, and every current demo runs the real brain. |
| `examples/renderer_demo.py` | `1335f0f` | A prototype from before the robot brain existed: a rule-based controller of its own, drawn through the `Renderer` port. Every current demo draws through the same port with the real brain. |
| `examples/simple_world_sim.py` | `1335f0f` | The earliest prototype of the cell model, with scripted robots. The simulation in `simulation/` replaced it. |
| `examples/worldstate_demo.py` | `1335f0f` | Printed a few `WorldState` steps to the terminal. It explained nothing the tests and the visual demos do not. |
| `examples/scenario_builder_demo.py` | `1335f0f` | Printed the scenario presets. Nothing referenced it, and the presets are described in `docs/experiments.md`. |
| `examples/treat_infestation_demo.py` | `1335f0f` | One killer treating three pests from hand-fed perceptions. `examples/pest_demo.py` shows the same with the real swarm. |
| `examples/robot_decision_demo.py` | `1335f0f` | A scripted agent on the `RobotDecision` event, an early explainer of the event loop. `docs/architecture.md` explains the loop, and every demo now runs on it. |
