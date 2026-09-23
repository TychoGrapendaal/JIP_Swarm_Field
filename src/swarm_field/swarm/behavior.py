"""The ``Behavior`` port and a couple of example behaviors.

A behavior is one composable piece of decision logic: given a read-only view of the robot
at a decision point, it proposes scored candidate actions. The brain combines several
behaviors and picks between their candidates.

The two examples below are still unimplemented: ``Explore`` drives field coverage and
``TreatInfestation`` reacts to sensed pests. More behaviors (return-to-charge,
avoid-crowding, ...) are added as they are built, each as its own small class.
"""

from dataclasses import dataclass, field
from random import Random
from typing import Protocol
import numpy as np
from math import exp

from swarm_field.contracts import (
    Action,
    Move,
    ObservationBase,
    Observation,
    CropObservation,
    InfestationObservation,
    ObstacleObservation,
    PeerObservation,
    RobotSelfState,
    Time,
    Turn,
    Message
)


@dataclass(frozen=True)
class ActionCandidate:
    """A scored action proposed by a behavior.

    Attributes:
        action: The action being proposed.
        score: How strongly the behavior wants it; the highest score wins.
        source: Name of the proposing behavior, for inspection / debugging.
    """

    action: Action
    score: float
    source: str


@dataclass(frozen=True)
class DecisionContext:
    """Read-only view handed to every behavior at a decision point.

    Minimal for now — a memory view (accumulated belief) and the message inbox are added
    when those parts are built.

    Attributes:
        time: The current simulation time.
        self_state: The robot's view of itself.
        sensed: What the robot's sensors picked up at this decision point.
    """

    time: Time
    self_state: RobotSelfState
    sensed: list[ObservationBase | Observation]
    inbox: list[Message] = field(default_factory=list)


class Behavior(Protocol):
    """A single, composable piece of decision logic.

    ``propose`` returns a single candidate, a tuple of candidates, or an empty tuple to
    propose nothing. Returning several candidates only matters to arbiters that
    fuse or fall back between them; the current brain simply takes the highest-scoring
    candidate across all behaviors, so returning your single best is usually enough.

    Attributes:
        name: A short identifier, used as the candidate's ``source``.
    """

    name: str

    def propose(self, ctx: DecisionContext) -> ActionCandidate | tuple[ActionCandidate, ...]: ...


class Explore:
    """Propose a move that improves field coverage.

    A real implementation would return several scored directions for a fusion arbiter (hence
    the tuple return type). This one is a MOCK that ignores coverage entirely.
    """

    name: str = "explore"

    def __init__(self, radius: int) -> None:
        # The behavior holds its own RNG. For reproducible runs it should be seeded from the
        # simulation's single seeded source when the brain is assembled.
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.radius: int = radius
        self.PotentialField     : PotentialField = PotentialField(radius=self.radius)
        self.net_force          : dict[tuple[int, int], tuple[float, float]] = {}
        self.force_exploration  : dict[tuple[int, int], tuple[float, float]] = {}
        self.force_crop         : dict[tuple[int, int], tuple[float, float]] = {}
        self.force_infestation  : dict[tuple[int, int], tuple[float, float]] = {}
        self.force_obstacle     : dict[tuple[int, int], tuple[float, float]] = {}
        self.force_peer         : dict[tuple[int, int], tuple[float, float]] = {}

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        field = PotentialField(radius=self.radius)
        self.PotentialField = field
        field._update(ctx)
        self._compute_force(field.exploration, self.force_exploration, False)
        self._compute_force(field.crop, self.force_crop, False)
        self._compute_force(field.infestation, self.force_infestation, False)
        self._compute_force(field.obstacle, self.force_obstacle, True, np.pi / 2)  # rotate obstacle force by 90 degrees
        self._compute_force(field.peer, self.force_peer, False)
        self._superimpose_forces([
            self.force_exploration,
            self.force_crop,
            self.force_infestation,
            self.force_obstacle,
            self.force_peer,
        ])
        # field.initiate()
        #TODO: Implement the logic to convert the net force into a set of ActionCandidates based on the robot's current state and the net force vector.
    
    def _compute_force(
        self,
        potential: dict[tuple[int, int], float],
        force_map: dict[tuple[int, int], tuple[float, float]],
        is_rotated: bool = False,
        angle: float = 0.0) -> None:
        """Compute the net force vector from the potential map.

        The force is the negative gradient of the potential. The robot is at (0, 0) in
        local coordinates; the force is computed as a sum of vectors from each cell to
        the robot, weighted by the cell's potential.
        """
        for (x, y) in potential:
            # dU/dx
            if (x - 1, y) in potential and (x + 1, y) in potential:
                dU_dx = (
                    potential[(x + 1, y)]
                    - potential[(x - 1, y)]
                ) / 2.0
            elif (x + 1, y) in potential:
                dU_dx = (
                    potential[(x + 1, y)]
                    - potential[(x, y)]
                )
            elif (x - 1, y) in potential:
                dU_dx = (
                    potential[(x, y)]
                    - potential[(x - 1, y)]
                )
            else:
                dU_dx = 0.0

            # dU/dy
            if (x, y - 1) in potential and (x, y + 1) in potential:
                dU_dy = (
                    potential[(x, y + 1)]
                    - potential[(x, y - 1)]
                ) / 2.0
            elif (x, y + 1) in potential:
                dU_dy = (
                    potential[(x, y + 1)]
                    - potential[(x, y)]
                )
            elif (x, y - 1) in potential:
                dU_dy = (
                    potential[(x, y)]
                    - potential[(x, y - 1)]
                )
            else:
                dU_dy = 0.0

            fx = -dU_dx
            fy = -dU_dy
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
        self,
        force_maps: list[dict[tuple[int, int], tuple[float, float]]]
    ) -> dict[tuple[int, int], tuple[float, float]]:
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
        self.radius         : int = radius
        self.exploration    : dict[tuple[int, int], float] = {}
        self.crop           : dict[tuple[int, int], float] = {}
        self.infestation    : dict[tuple[int, int], float] = {}
        self.obstacle       : dict[tuple[int, int], float] = {}
        self.peer           : dict[tuple[int, int], float] = {}
        self.timemap        : dict[tuple[int, int], Time ] = {}
        self._current_time  : Time = 0
        self._position      : tuple[int, int] | None = None

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
        """Construct the potential maps from the robot's current state, observations and peer messages.
        
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
        """Return the potential for an observation at the given cell.
        """
        if self.exploration.get(cell, 0.0) <= 4.0:
            self.exploration[cell] = self.exploration.get(cell, 0.0) + 1.0
        return 

    def _potential_crop(self, obs: CropObservation, potentials: dict[tuple[int, int], float]) -> None:
        """"Return the potential for a crop at the given cell.
        """
        raise NotImplementedError("Potential for crops is not yet implemented in PotentialField._potential_crop()")
    
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
        """Return the potential for a peer at the given cell.
        """
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
        """Shift stored local coordinates when the robot changes position.
        """
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
        raise NotImplementedError("PotentialField._transform_map() is not yet implemented.\nThe relative positions are not yet handled.")
                
    def _in_radius(self, cell: tuple[int, int]) -> bool:
        """Return whether a local cell belongs to the circular field."""
        return cell[0] * cell[0] + cell[1] * cell[1] <= self.radius * self.radius


class TreatInfestation:
    """Propose treating a sensed infestation, scored by severity and proximity.

    Returns a single candidate (narrowing the protocol's return type to just
    ``ActionCandidate``), since it commits to one target at a time.
    """

    name: str = "treat-infestation"

    def propose(self, ctx: DecisionContext) -> ActionCandidate:
        raise NotImplementedError
