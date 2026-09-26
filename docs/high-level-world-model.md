# High-level world model

The current simulation uses a coarse grid. One grid cell represents a patch or area of field, not an individual plant. This keeps the simulation focused on swarm behaviour, coverage, potential fields and scalability to hundreds of robots.

## Field cells

Each `CellState` contains the ground-truth state of one field patch:

- `crop`: aggregate crop type for the patch.
- `growth_stage`: aggregate crop maturity.
- `infestation`: true pest severity in the patch, from 0 to 1.
- `obstacle`: whether the patch is blocked.

A physical obstacle can therefore occupy multiple cells. A river three cells wide is simply three adjacent cells with `obstacle=True` (and can be much longer).

## Robot occupancy

Cells have no robot-capacity limit and do not store a single `occupant`. Robot poses live in `WorldState.robots`. Multiple robots can share the same cell, which is important for swarm experiments where agents can temporarily converge.

## Movement

Movement is a direct target-cell command:

```python
Move(to=(x, y))
```

The simulation does not model turning or heading. A robot selects its next target cell through the navigation algorithm, such as a potential-field controller, and the world computes the travel time from the robot's speed and the physical cell size.

Robot speeds are stored in metres per second. Different robot body types can therefore be assigned different speeds while sharing the same movement interface.

## Obstacles and potential fields

The ground-truth world marks obstacles. The robot brain receives local obstacle observations. A potential-field implementation can then store an obstacle as a repulsive value in its own local belief map rather than modifying the world truth.

A typical local decision cycle is:

1. Sense the current cell and neighbouring cells.
2. Add newly observed obstacles to the local repulsive field.
3. Add observed infestation to the local attractive field.
4. Evaluate candidate cells.
5. Choose a target cell with the lowest combined potential.
6. Send `Move(to=target)`.

The included mass-swarm example demonstrates this pattern with a lightweight local controller.

## Scalability demonstration

`examples/mass_swarm_visualization.py` starts with 300 robots by default and can be configured with `--robots`, for example:

```bash
uv run --with matplotlib python examples/mass_swarm_visualization.py --robots 500
```

The visualization uses large cells and plots robot positions as a scatter plot rather than rendering individual plant geometry, so it remains conceptually aligned with the intended swarm-level model.
