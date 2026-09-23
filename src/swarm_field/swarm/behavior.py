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
from typing import Protocol, TypeVar
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

ObservationT = TypeVar("ObservationT", bound=ObservationBase)


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
    sensed: list[Observation]
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

    def __init__(self, rng: Random) -> None:
        # The behavior holds its own RNG. For reproducible runs it should be seeded from the
        # simulation's single seeded source when the brain is assembled.
        self._rng: Random = rng

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        # MOCK: coin flip — 50% go straight ahead, 50% turn 45 degrees left. No real
        # exploration; replace with coverage-driven logic.
        if self._rng.random() < 0.5:
            action: Action = Move()
        else:
            action = Turn(facing=ctx.self_state.facing.ccw())
        return (ActionCandidate(action=action, score=1.0, source=self.name),)


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
        self.potential_map: dict[tuple[int, int], float] = {}
        self.exploration: dict[tuple[int, int], float] = {}
        self.infestation: dict[tuple[int, int], float] = {}
        self.obstacles: dict[tuple[int, int], float] = {}
        self.peers: dict[tuple[int, int], float] = {}
        self.infestation_observations: list[InfestationObservation] = []
        self.obstacle_observations: list[ObstacleObservation] = []
        self.peer_observations: list[PeerObservation] = []
        self.timemap: dict[tuple[int, int], Time] = {}
        self._current_time: Time = 0
        self._position: tuple[int, int] | None = None

    def _initiate(self) -> None:
        """Clear only the derived combined view before rebuilding it."""
        self.potential_map.clear()
        
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

    def update(self, ctx: DecisionContext) -> None:
        """Construct the potential maps from the robot's current state, observations and peer messages.
        
        Four maps are maintained: exploration, infestation, obstacles, and peers. 
        Each is updated with the robot's current observations and peer messages, 
        and then superimposed into a single combined view.
        """
        self._initiate()
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
            
            self._generate_potential_map(obs, cell)
        
        # TODO: include the peer's observations. Implement after the relative positions are handled.
        
        maps = (
            self.exploration,
            self.infestation,
            self.obstacles,
            self.peers,
        )
        self._superimpose(maps)
        return
    
    def _generate_potential_map(self, obs: Observation, cell: tuple[int, int]) -> None:
        """Rebuild the current combined field from the individual maps.

        This is called after updating the individual maps, and after shifting them when
        the robot moves.
        """
        if isinstance(obs, CropObservation):
            self._potential_crop(obs)
            return

        if isinstance(obs, InfestationObservation):
            self._store_observation(self.infestation_observations, obs)
            return

        if isinstance(obs, ObstacleObservation):
            self._store_observation(self.obstacle_observations, obs)
            return

        if isinstance(obs, PeerObservation):
            self._store_observation(self.peer_observations, obs)
            return

        if isinstance(obs, ObservationBase):
            self._potential_observation(cell)
            return

        raise TypeError(f"unexpected observation type: {type(obs).__name__}")

    @staticmethod
    def _store_observation(
        observations: list[ObservationT],
        observation: ObservationT,
    ) -> None:
        """Replace an older observation for the same world cell."""
        for index, stored in enumerate(observations):
            if stored.cell == observation.cell:
                observations[index] = observation
                return
        observations.append(observation)

    def _potential_observation(self, cell: tuple[int, int]) -> None:
        """Return the potential for an observation at the given cell.
        """
        if self.exploration.get(cell, 0.0) <= 4.0:
            self.exploration[cell] = self.exploration.get(cell, 0.0) + 1.0
        return 

    def _potential_crop(self, obs: CropObservation) -> None:
        """"Return the potential for a crop at the given cell.
        """
        raise NotImplementedError("Potential for crops is not yet implemented in PotentialField._potential_crop()")
    
    def _potential_infestation(
        self,
        obs: InfestationObservation,
        center: tuple[int, int],
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
                self.infestation[target] = self.infestation.get(target, 0.0) + value
    
    def _potential_obstacle(self, center: tuple[int, int]) -> None:
        """Spread a negative, sideways-oriented Gaussian around an obstacle.

        The narrow axis points toward the obstacle and the broad axis is rotated 90
        degrees from it. This gives nearby cells a lateral gradient to steer around
        the obstacle instead of only penalising the occupied cell.
        """
        sigma_forward = 0.75
        sigma_sideways = 1.5
        amplitude = -0.6
        center_distance = (center[0] ** 2 + center[1] ** 2) ** 0.5
        support = 3 * sigma_sideways
        if center_distance > self.radius + support:
            return

        forward_squared = center[0] * center[0] + center[1] * center[1]
        for x in range(-self.radius, self.radius + 1):
            for y in range(-self.radius, self.radius + 1):
                target = (x, y)
                if not self._in_radius(target):
                    continue

                offset = (x - center[0], y - center[1])
                along_obstacle_axis = (
                    offset[0] * center[0] + offset[1] * center[1]
                )
                if forward_squared == 0:
                    along_obstacle_axis = offset[0]
                    across_obstacle_axis = offset[1]
                else:
                    across_obstacle_axis = (
                        offset[0] * center[1] - offset[1] * center[0]
                    )

                exponent = -0.5 * (
                    (along_obstacle_axis * along_obstacle_axis)
                    / (forward_squared * sigma_forward * sigma_forward)
                    if forward_squared > 0
                    else (along_obstacle_axis * along_obstacle_axis)
                    / (sigma_forward * sigma_forward)
                )
                exponent -= 0.5 * (
                    (across_obstacle_axis * across_obstacle_axis)
                    / (forward_squared * sigma_sideways * sigma_sideways)
                    if forward_squared > 0
                    else (across_obstacle_axis * across_obstacle_axis)
                    / (sigma_sideways * sigma_sideways)
                )
                value = amplitude * exp(exponent)
                self.obstacles[target] = self.obstacles.get(target, 0.0) + value
    
    def _potential_peer(self, obs: PeerObservation, center: tuple[int, int]) -> None:
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
                self.peers[target] = self.peers.get(target, 0.0) + value
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

    def _superimpose(self, maps: tuple[dict[tuple[int, int], float], ...]) -> None:
        """Add fields to the current combined view."""
        self.infestation.clear()
        self.obstacles.clear()
        self.peers.clear()
        if self._position is None:
            return

        for observation in self.infestation_observations:
            center = (
                observation.cell[0] - self._position[0],
                observation.cell[1] - self._position[1],
            )
            self._potential_infestation(observation, center)
        for observation in self.obstacle_observations:
            if not observation.is_obstacle:
                continue
            center = (
                observation.cell[0] - self._position[0],
                observation.cell[1] - self._position[1],
            )
            self._potential_obstacle(center)
        for observation in self.peer_observations:
            center = (
                observation.cell[0] - self._position[0],
                observation.cell[1] - self._position[1],
            )
            self._potential_peer(observation, center)

        self.potential_map.clear()
        for map in maps:
            for cell, potential in map.items():
                if self._in_radius(cell):
                    self.potential_map[cell] = self.potential_map.get(cell, 0.0) + potential
                
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
