"""Small terminal demonstration of the high-level WorldState model.

Run from the repository root with:
    uv run python examples/worldstate_demo.py
"""

from swarm_field.contracts import BodyType, FieldGeometry, Move, Treat
from swarm_field.simulation import RobotState, WorldState


def main() -> None:
    field = FieldGeometry(
        width=8,
        height=6,
        cell_size=5.0,
        row_sp=2.0,
        in_row_sp=1.0,
        headland=5.0,
        connectivity=8,
    )

    robots = [
        RobotState(robot_id=1, position=(1, 1), body=BodyType.SCOUT, speed_mps=2.0),
        RobotState(robot_id=2, position=(1, 1), body=BodyType.KILLER, speed_mps=1.0),
        RobotState(robot_id=3, position=(1, 1), body=BodyType.BATTERY, speed_mps=0.75),
    ]
    world = WorldState(field, robots=robots)

    for x in (3, 4, 5):
        world.cell((x, 3)).obstacle = True
    world.cell((6, 4)).infestation = 1.0

    print("High-level WorldState demo")
    print("---------------------------")
    print(f"Field: {field.width} x {field.height} cells, {field.cell_size:g} m/cell")
    print(f"Robots initially sharing cell (1,1): {world.robots_at((1, 1))}")
    print("River/obstacle cells: (3,3), (4,3), (5,3)")
    print("Infestation: (6,4) = 1.0")

    fast_arrival = world.apply_action(1, Move(to=(2, 2)), time=0)
    slow_arrival = world.apply_action(2, Move(to=(2, 2)), time=0)
    print(f"Scout arrival time:  {fast_arrival} ms")
    print(f"Killer arrival time: {slow_arrival} ms")
    print(f"Current positions: {world.positions()}")

    world.advance_to(3536)
    print(f"Positions after 3.536 s: {world.positions()}")

    world.apply_action(2, Treat(cell=(6, 4)), time=3536)
    print(f"Infestation after treatment: {world.cell((6, 4)).infestation}")
    print("Multiple robots per cell remain allowed throughout the simulation.")


if __name__ == "__main__":
    main()
