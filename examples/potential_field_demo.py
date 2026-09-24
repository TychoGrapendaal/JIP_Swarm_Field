"""Visualize the local potential field built from robot observations.

Run from the repository root with:
    uv run --with matplotlib python examples/potential_field_demo.py
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.ticker import MultipleLocator

from swarm_field.contracts import (
    BodyType,
    Direction,
    InfestationObservation,
    Observation,
    ObservationBase,
    ObstacleObservation,
    PeerObservation,
    RobotSelfState,
)
from swarm_field.swarm import DecisionContext, Explore, PotentialField


def make_context(
    time: int,
    position: tuple[int, int],
    sensed: list[ObservationBase | Observation],
) -> DecisionContext:
    """Build the small read-only input handed to the potential field."""
    return DecisionContext(
        time=time,
        self_state=RobotSelfState(
            robot_id=1,
            position=position,
            facing=Direction.N,
            battery=1.0,
            body=BodyType.SCOUT,
            consumable=1.0,
            health=1.0,
        ),
        sensed=sensed,
    )


def observations(
    time: int, position: tuple[int, int], explore_radius: int
) -> list[ObservationBase | Observation]:
    """Return the current scan, replacing the infestation at time 1."""
    infestation_cell = (4, 2) if time == 0 else (-4, 4)
    return [
        InfestationObservation(
            cell=infestation_cell,
            confidence=1.0,
            observed_at=time,
            source=1,
            severity=0.8,
            pest=None,
        ),
        ObstacleObservation(
            cell=(-2, -3),
            confidence=1.0,
            observed_at=time,
            source=1,
            is_obstacle=True,
        ),
        PeerObservation(
            cell=(3, 1),
            confidence=1.0,
            observed_at=time,
            source=1,
            peer=2,
        ),
        *_explored_observations(time, position, explore_radius),
    ]


def _explored_observations(
    time: int, position: tuple[int, int], explore_radius: int
) -> list[ObservationBase]:
    """Create one exploration observation for every cell in the configured radius."""
    observations: list[ObservationBase] = []
    for x in range(-explore_radius, explore_radius + 1):
        for y in range(-explore_radius, explore_radius + 1):
            if x * x + y * y > explore_radius * explore_radius:
                continue
            observations.append(
                ObservationBase(
                    cell=(position[0] + x, position[1] + y),
                    confidence=1.0,
                    observed_at=time,
                    source=1,
                )
            )
    return observations


def render(
    explore: Explore,
    field: PotentialField,
    axis: plt.Axes,
    position: tuple[int, int],
    infestation_cell: tuple[int, int],
    explore_radius: int,
    show: int = 1,
) -> plt.AxesImage:
    """Render the robot-relative combined field and its component sources."""
    radius = field.radius
    coordinates = range(-radius, radius + 1)
    values = np.full((2 * radius + 1, 2 * radius + 1), np.nan)
    if show == 1:
        maps = field.exploration
        force_map = explore.force_exploration
    elif show == 2:
        maps = field.infestation
        force_map = explore.force_infestation
    elif show == 3:
        maps = field.obstacle
        force_map = explore.force_obstacle
    elif show == 4:
        maps = field.peer
        force_map = explore.force_peer
    elif show == 5:
        maps = {}
        force_map = explore.net_force
    for x in coordinates:
        for y in coordinates:
            if (x, y) in maps:
                values[y + radius, x + radius] = maps[(x, y)]

    for row_index, y in enumerate(coordinates):
        for col_index, x in enumerate(coordinates):
            if x * x + y * y <= radius * radius and np.isnan(values[row_index, col_index]):
                values[row_index, col_index] = 0.0

    x = [pos[0] for pos in force_map]
    y = [pos[1] for pos in force_map]

    fx = [force[0] for force in force_map.values()]
    fy = [force[1] for force in force_map.values()]

    axis.clear()
    axis.quiver(x, y, fx, fy, color="black", scale=1, scale_units="xy", angles="xy", width=0.005)
    axis.add_patch(
        Circle(
            (0, 0),
            explore_radius,
            color="green",
            fill=False,
            alpha=0.2,
            linewidth=1,
            label="explore radius",
        )
    )
    limit = radius + 0.5
    image = axis.imshow(
        values,
        extent=(-limit, limit, -limit, limit),
        origin="lower",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    axis.scatter(0, 0, marker="+", s=110, color="black", linewidths=2, label="robot")
    axis.scatter(
        -2 - position[0],
        -3 - position[1],
        marker="s",
        s=100,
        color="black",
        label="obstacle",
    )
    axis.scatter(
        infestation_cell[0] - position[0],
        infestation_cell[1] - position[1],
        marker="*",
        s=100,
        color="gold",
        label="infestation",
    )
    axis.scatter(
        3 - position[0],
        1 - position[1],
        marker="o",
        s=100,
        color="blue",
        label="peer",
    )
    axis.set(
        aspect="equal",
        title="Local combined potential",
        xlabel="relative X",
        ylabel="relative Y",
        xticks=list(coordinates),
        yticks=list(coordinates),
    )
    axis.xaxis.set_major_locator(MultipleLocator(1))
    axis.yaxis.set_major_locator(MultipleLocator(1))
    axis.xaxis.set_minor_locator(MultipleLocator(0.5))
    axis.yaxis.set_minor_locator(MultipleLocator(0.5))
    axis.grid(which="minor", color="white", alpha=0.35)
    axis.legend(loc="upper left", fontsize="small")
    return image


def main(show: int) -> None:
    explore_radius = 3
    radius = 5
    field = PotentialField(radius=radius)
    explore = Explore(radius=radius)
    for time in (0, 1):
        figure = plt.figure(figsize=(7, 6))
        axis = figure.add_subplot(1, 1, 1)
        figure.suptitle("PotentialField prototype: robot moves north")
        position = (0, time)
        infestation_cell = (4, 2) if time == 0 else (-4, 4)
        field._update(
            make_context(
                time=time, position=position, sensed=observations(time, position, explore_radius)
            )
        )
        explore.propose(make_context(time, position, observations(time, position, explore_radius)))
        image = render(explore, field, axis, position, infestation_cell, explore_radius, show)
        axis.set_title(f"time={time}: robot at {position}")
        figure.colorbar(image, ax=axis, label="potential")
        figure.tight_layout()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a local potential-field component.")
    parser.add_argument(
        "-s",
        "--show",
        type=int,
        choices=range(1, 6),
        default=1,
        metavar="{1,2,3,4,5}",
        help="1: exploration, 2: infestation, 3: obstacle, 4: peer, 5: combined",
    )
    main(parser.parse_args().show)
