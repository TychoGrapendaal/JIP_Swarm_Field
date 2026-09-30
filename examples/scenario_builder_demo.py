"""Build and inspect the five built-in scenario presets."""

from swarm_field.contracts import ScenarioId
from swarm_field.simulation import build_scenario, build_world


def main() -> None:
    """Print the main characteristics of every built-in scenario."""
    for scenario in ScenarioId:
        config = build_scenario(scenario)
        world = build_world(config)
        infestations = sum(state.infestation > 0.0 for state in world.cells.values())
        obstacles = sum(state.obstacle for state in world.cells.values())
        crops = len({state.crop for state in world.cells.values() if state.crop is not None})
        print(
            f"{scenario.name}: "
            f"{config.field.width}x{config.field.height}, "
            f"robots={len(world.robots)}, crops={crops}, "
            f"infested_cells={infestations}, obstacles={obstacles}"
        )


if __name__ == "__main__":
    main()
