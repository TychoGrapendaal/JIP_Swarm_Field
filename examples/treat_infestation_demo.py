"""Simple animated demonstration of TreatInfestation following the field of TreatSeeder.

A killer robot knows of three pests and should walk to each, treat it once, and then idle.

Run from the repository root with:
    uv run python examples/treat_infestation_demo.py
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from swarm_field.contracts import BodyType, FieldGeometry, InfestationObservation, Perception
from swarm_field.simulation import RobotState, WorldState
from swarm_field.swarm import (
    AgentMemory,
    ComposedAgent,
    FieldProvider,
    Navigator,
    TreatInfestation,
    TreatSeeder,
    UtilityArbiter,
)
from swarm_field.visualization import MatplotlibRenderer

ROBOT_ID = 1
PESTS = {(3, 6): 0.5, (9, 6): 1.0, (10, 1): 0.7}
TICK_MS = 200


def make_world() -> WorldState:
    field = FieldGeometry(
        width=12,
        height=8,
        cell_size=2.0,
        row_sp=2.0,
        in_row_sp=1.0,
        headland=5.0,
        connectivity=8,
    )
    world = WorldState(field, robots=[RobotState(ROBOT_ID, (1, 1), BodyType.KILLER, speed_mps=2.0)])

    for y in range(0, 6):
        world.cell((6, y)).obstacle = True
    for cell, severity in PESTS.items():
        world.cell(cell).infestation = severity

    return world


def make_agent(field: FieldGeometry) -> ComposedAgent:
    memory = AgentMemory(ROBOT_ID, field)
    # The robot already knows about the pests, as if a scout had reported them.
    for cell, severity in PESTS.items():
        memory.merge([InfestationObservation(cell, 1.0, 0, 2, severity=severity, pest=None)])
    fields = FieldProvider(memory, flies=False, seeders=[TreatSeeder()])
    return ComposedAgent(
        robot_id=ROBOT_ID,
        behaviors=[TreatInfestation()],
        memory=memory,
        fields=fields,
        navigator=Navigator(fields.quadtree),
        arbiter=UtilityArbiter(),
    )


def main() -> None:
    world = make_world()
    agent = make_agent(world.field)
    renderer = MatplotlibRenderer(show_targets=False)

    def update(frame: int) -> None:
        time = frame * TICK_MS
        world.advance_to(time)
        robot = world.robots[ROBOT_ID]

        if robot.move_target is None:
            perception = Perception(
                time=time,
                self_state=world.perceived_self(ROBOT_ID),
                sensed=world.observe_for(ROBOT_ID, time),
                inbox=[],
            )
            action = agent.decide(perception).action
            world.apply_action(ROBOT_ID, action, time=time)

        renderer.render(world.snapshot(time))
        renderer.axes.set_title(
            f"time={time / 1000:.1f}s | {robot.current_action} | "
            f"consumable={robot.consumable:.1f} (one treatment costs 0.1)"
        )

    animation = FuncAnimation(renderer.figure, update, frames=400, interval=40, repeat=False)
    _ = animation
    plt.show()


if __name__ == "__main__":
    main()
