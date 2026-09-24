"""Composable robot behaviours for the high-level cell model."""

from dataclasses import dataclass, field
from math import exp
from typing import Protocol

import numpy as np

from swarm_field.contracts import (
    Action,
    CropObservation,
    InfestationObservation,
    Message,
    Observation,
    ObservationBase,
    ObstacleObservation,
    PeerObservation,
    RobotSelfState,
    Time,
)


@dataclass(frozen=True)
class ActionCandidate:
    """A scored action proposed by a behaviour."""

    action: Action
    score: float
    source: str


@dataclass(frozen=True)
class DecisionContext:
    """Read-only view handed to every behaviour at a decision point."""

    time: Time
    self_state: RobotSelfState
    sensed: list[Observation]
    inbox: list[Message] = field(default_factory=list)


class Behavior(Protocol):
    """A single composable piece of decision logic."""

    name: str

    def propose(self, ctx: DecisionContext) -> ActionCandidate | tuple[ActionCandidate, ...]: ...


class Explore:
    """Mock exploration behaviour using direct target-cell moves."""

    name: str = "explore"

    def __init__(self, radius: int) -> None:
        # The behavior holds its own RNG. For reproducible runs it should be seeded from the
        # simulation's single seeded source when the brain is assembled.
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.radius: int = radius
        self.PotentialField: PotentialField = PotentialField(radius=self.radius)
        self.net_force: dict[tuple[int, int], tuple[float, float]] = {}
        self.force_exploration: dict[tuple[int, int], tuple[float, float]] = {}
        self.force_crop: dict[tuple[int, int], tuple[float, float]] = {}
        self.force_infestation: dict[tuple[int, int], tuple[float, float]] = {}
        self.force_obstacle: dict[tuple[int, int], tuple[float, float]] = {}
        self.force_peer: dict[tuple[int, int], tuple[float, float]] = {}

    def propose(self, ctx: DecisionContext) -> None:
        field = self.PotentialField
        field._update(ctx)
        self._compute_force(field.exploration, self.force_exploration, False)
        self._compute_force(field.crop, self.force_crop, False)
        self._compute_force(field.infestation, self.force_infestation, False)
        self._compute_force(field.obstacle, self.force_obstacle, True, np.pi / 2)
        self._compute_force(field.peer, self.force_peer, False)
        self._superimpose_forces(
            [
                self.force_exploration,
                self.force_crop,
                self.force_infestation,
                self.force_obstacle,
                self.force_peer,
            ]
        )
        field.initiate()
        # TODO: Implement the logic to convert the net force into a set of ActionCandidates
        # based on the robot's current state and the net force vector.

    def _compute_force(
        self,
        potential: dict[tuple[int, int], float],
        force_map: dict[tuple[int, int], tuple[float, float]],
        is_rotated: bool = False,
        angle: float = 0.0,
    ) -> None:
        """Compute the net force vector from the potential map.

        The force is the negative gradient of the potential. The robot is at (0, 0) in
        local coordinates; the force is computed as a sum of vectors from each cell to
        the robot, weighted by the cell's potential.
        """
        for x in range(-self.radius, self.radius + 1):
            for y in range(-self.radius, self.radius + 1):
                # Only include points inside the circular area
                if x * x + y * y > self.radius * self.radius:
                    continue

                # No potential at this location -> zero force
                if (x, y) not in potential:
                    force_map[(x, y)] = (0.0, 0.0)
                    continue
                # dU/dx
                if (x - 1, y) in potential and (x + 1, y) in potential:
                    du_dx = (potential[(x + 1, y)] - potential[(x - 1, y)]) / 2.0
                elif (x + 1, y) in potential:
                    du_dx = potential[(x + 1, y)] - potential[(x, y)]
                elif (x - 1, y) in potential:
                    du_dx = potential[(x, y)] - potential[(x - 1, y)]
                else:
                    du_dx = 0.0

                # dU/dy
                if (x, y - 1) in potential and (x, y + 1) in potential:
                    du_dy = (potential[(x, y + 1)] - potential[(x, y - 1)]) / 2.0
                elif (x, y + 1) in potential:
                    du_dy = potential[(x, y + 1)] - potential[(x, y)]
                elif (x, y - 1) in potential:
                    du_dy = potential[(x, y)] - potential[(x, y - 1)]
                else:
                    du_dy = 0.0

                fx = -du_dx
                fy = -du_dy
                if is_rotated:
                    fx, fy = self._rotate_vector((fx, fy), angle)

                force_map[(x, y)] = (fx, fy)

    def _rotate_vector(self, vector: tuple[float, float], angle: float) -> tuple[float, float]:
        """Rotate a 2D vector by a given angle in radians."""
        x, y = vector
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

    def _superimpose_forces(
        self, force_maps: list[dict[tuple[int, int], tuple[float, float]]]
    ) -> None:
        """Combine multiple force maps into a single net force map."""
        for force_map in force_maps:
            for cell, (fx, fy) in force_map.items():
                if cell not in self.net_force:
                    self.net_force[cell] = (0.0, 0.0)
                self.net_force[cell] = (self.net_force[cell][0] + fx, self.net_force[cell][1] + fy)


class PotentialField:
    """Maintain exploration, infestation, and obstacle potential maps.

    All maps use coordinates relative to the robot. The robot is always at ``(0, 0)``;
    maps are shifted when the robot moves.
    """

    name: str = "potential-field"

    def __init__(self, radius: int) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.radius: int = radius
        self.exploration: dict[tuple[int, int], float] = {}
        self.crop: dict[tuple[int, int], float] = {}
        self.infestation: dict[tuple[int, int], float] = {}
        self.obstacle: dict[tuple[int, int], float] = {}
        self.peer: dict[tuple[int, int], float] = {}
        self.timemap: dict[tuple[int, int], Time] = {}
        self._current_time: Time = 0
        self._position: tuple[int, int] | None = None

    def initiate(self) -> None:
        """Clear only the derived combined view before rebuilding it."""
        self.crop.clear()
        self.infestation.clear()
        self.obstacle.clear()
        self.peer.clear()

    def _generate_timemap(self, ctx: DecisionContext) -> None:
        """Update the timemap with the current observations.

        The timemap is used to track when each cell was last observed, which can be used
        to decay the potential values over time or for updates potential maps.
        """
        self._move_map_to(ctx.self_state.position)
        for obs in ctx.sensed:
            cell = (
                obs.cell[0] - ctx.self_state.position[0],
                obs.cell[1] - ctx.self_state.position[1],
            )
            if cell not in self.timemap:
                self.timemap[cell] = ctx.time

    def _update(self, ctx: DecisionContext) -> None:
        """Construct the potential maps from the robot's current state,
        observations and peer messages.

        Exploration is persistent. Infestation, obstacle, and peer maps are constructed
        from their observation lists for this update and folded into the combined view.
        """
        self.initiate()
        self._current_time = ctx.time
        self._move_map_to(ctx.self_state.position)
        for obs in ctx.sensed:
            cell = (
                obs.cell[0] - ctx.self_state.position[0],
                obs.cell[1] - ctx.self_state.position[1],
            )
            if cell in self.timemap and obs.observed_at < self.timemap[cell]:
                continue
            self.timemap[cell] = obs.observed_at

            self._generate_potentialmap(obs, cell)

        # TODO: include the peer's observations. Implement after the relative positions are handled.

    def _generate_potentialmap(
        self, obs: ObservationBase | Observation, cell: tuple[int, int]
    ) -> None:
        """Generate the potential map from a new sensor reading."""
        if isinstance(obs, CropObservation):
            self._potential_crop(obs, self.crop)
            return

        if isinstance(obs, InfestationObservation):
            self._potential_infestation(obs, cell, self.infestation)
            return

        if isinstance(obs, ObstacleObservation):
            self._potential_obstacle(obs, cell, self.obstacle)
            return

        if isinstance(obs, PeerObservation):
            self._potential_peer(obs, cell, self.peer)
            return

        if isinstance(obs, ObservationBase):
            self._potential_observation(cell)
            return

        raise TypeError(f"unexpected observation type: {type(obs).__name__}")

    def _potential_observation(self, cell: tuple[int, int]) -> None:
        """Return the potential for an observation at the given cell."""
        if self.exploration.get(cell, 0.0) <= 4.0:
            self.exploration[cell] = self.exploration.get(cell, 0.0) + 1.0
        return

    def _potential_crop(
        self, obs: CropObservation, potentials: dict[tuple[int, int], float]
    ) -> None:
        """Return the potential for a crop at the given cell."""
        raise NotImplementedError(
            "Potential for crops is not yet implemented in\
                PotentialField._potential_crop()"
        )

    def _potential_infestation(
        self,
        obs: InfestationObservation,
        center: tuple[int, int],
        potentials: dict[tuple[int, int], float],
    ) -> None:
        """Spread an infestation as a confidence-weighted radial Gaussian.

        Severity controls both the peak strength and the standard deviation. An
        observation whose three-sigma support cannot reach the field is ignored.
        """
        sigma = obs.severity
        if sigma <= 0:
            return

        center_distance = (center[0] ** 2 + center[1] ** 2) ** 0.5
        if center_distance > self.radius + 3 * sigma:
            return

        amplitude = -obs.confidence * obs.severity
        sigma_squared = sigma * sigma
        for x in range(-self.radius, self.radius + 1):
            for y in range(-self.radius, self.radius + 1):
                target = (x, y)
                if not self._in_radius(target):
                    continue
                distance_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2
                value = amplitude * exp(-distance_squared / (2 * sigma_squared))
                potentials[target] = potentials.get(target, 0.0) + value

    def _potential_obstacle(
        self,
        obs: ObstacleObservation,
        center: tuple[int, int],
        potentials: dict[tuple[int, int], float],
    ) -> None:
        """Create a Gaussian potential whose local force is perpendicular to an obstacle.

        A rotated Gaussian ellipse would still produce a mostly radial gradient. Instead,
        this field is a positive Gaussian-weighted lateral potential. Both sides of the
        obstacle's 90-degree axis receive the same positive potential.
        """
        sigma = 1.0
        sigma_squared = sigma * sigma
        amplitude = obs.confidence * 1.5

        center_distance = (center[0] ** 2 + center[1] ** 2) ** 0.5
        if center_distance > self.radius + 3 * sigma:
            return
        for x in range(-self.radius, self.radius + 1):
            for y in range(-self.radius, self.radius + 1):
                target = (x, y)
                if not self._in_radius(target):
                    continue
                distance_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2
                value = amplitude * exp(-distance_squared / (2 * sigma_squared))
                potentials[target] = potentials.get(target, 0.0) + value

    def _potential_peer(
        self,
        obs: PeerObservation,
        center: tuple[int, int],
        potentials: dict[tuple[int, int], float],
    ) -> None:
        """Return the potential for a peer at the given cell."""
        sigma = 1.0
        sigma_squared = sigma * sigma
        amplitude = obs.confidence * 0.7

        center_distance = (center[0] ** 2 + center[1] ** 2) ** 0.5
        if center_distance > self.radius + 3 * sigma:
            return
        for x in range(-self.radius, self.radius + 1):
            for y in range(-self.radius, self.radius + 1):
                target = (x, y)
                if not self._in_radius(target):
                    continue
                distance_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2
                value = amplitude * exp(-distance_squared / (2 * sigma_squared))
                potentials[target] = potentials.get(target, 0.0) + value
        return

    def _move_map_to(self, position: tuple[int, int]) -> None:
        """Shift stored local coordinates when the robot changes position."""
        if self._position is None:
            self._position = position
            return
        delta = (position[0] - self._position[0], position[1] - self._position[1])
        if delta == (0, 0):
            return

        shifted_exploration = {
            (cell[0] - delta[0], cell[1] - delta[1]): potential
            for cell, potential in self.exploration.items()
            if self._in_radius((cell[0] - delta[0], cell[1] - delta[1]))
        }
        self.exploration.clear()
        self.exploration.update(shifted_exploration)

        shifted_timemap = {
            (cell[0] - delta[0], cell[1] - delta[1]): observed_at
            for cell, observed_at in self.timemap.items()
            if self._in_radius((cell[0] - delta[0], cell[1] - delta[1]))
        }
        self.timemap.clear()
        self.timemap.update(shifted_timemap)
        self._position = position

    def _transform_map(self, peer_position: tuple[int, int]) -> None:
        """Transform the peer's observations to the robot's local coordinates."""
        raise NotImplementedError(
            "PotentialField._transform_map() is not yet implemented.\
            The relative positions are not yet handled."
        )

    def _in_radius(self, cell: tuple[int, int]) -> bool:
        """Return whether a local cell belongs to the circular field."""
        return cell[0] * cell[0] + cell[1] * cell[1] <= self.radius * self.radius


class TreatInfestation:
    """Placeholder for the future pest-response behaviour."""

    name: str = "treat-infestation"

    def propose(self, ctx: DecisionContext) -> ActionCandidate:
        raise NotImplementedError
