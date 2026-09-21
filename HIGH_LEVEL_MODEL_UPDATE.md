# High-level model update

This version updates the B.1 world model following the team's decision to study swarm behaviour at a coarser level.

## Main changes

1. **Cell = field patch.** A grid cell is an area of field and no longer represents one plant.
2. **Multi-cell obstacles.** Rivers, ponds and other obstacles are represented by any set of obstacle cells; a three-cell-wide river is three adjacent blocked cells.
3. **Unlimited robot occupancy.** `CellState` no longer stores one `occupant`. Multiple robots can share one cell, and `WorldState.robots` owns robot poses.
4. **Direct target movement.** `Move(to=CellId)` replaces turn-then-step movement. There is no heading/facing state in the robot exchange.
5. **Robot-specific speed.** Every `RobotState` has `speed_mps`. The world calculates movement duration from direct cell distance, cell size and robot speed.
6. **Potential-field seam.** The world reports obstacle and infestation observations. The included mass-swarm demo stores obstacle repulsion and infestation attraction in per-robot local memory and selects a target cell from local candidates.
7. **Hundreds of robots.** `examples/mass_swarm_visualization.py` defaults to 300 robots and accepts `--robots` for larger runs.

## What is still deliberately simple

Infestation is still a static per-cell severity value. Pest spreading, richer sensing, communication, belief fusion, dynamic role allocation and benchmark metrics remain separate layers.
