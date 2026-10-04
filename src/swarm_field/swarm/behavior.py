"""Composable robot behaviours for the high-level cell model."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from heapq import nlargest
from math import ceil, exp, hypot, isfinite
from typing import Protocol

from swarm_field.contracts import (
    CellId,
    Charge,
    Idle,
    Message,
    Observation,
    RobotSelfState,
    Time,
    Treat,
)
from swarm_field.swarm.field import BoundingBox, GridPotentialField, MixedGraph
from swarm_field.swarm.memory import MemoryView
from swarm_field.swarm.navigator import Navigator
from swarm_field.swarm.seeder import ExploreSeeder, SeededField, TreatSeeder


@dataclass(frozen=True)
class FollowField:
    """Take one step along the grid potential field of a registered seeder.

    Attributes:
        seeder: The name of the seeder whose field to follow.
    """

    seeder: str


Intent = FollowField | Treat | Charge | Idle
"""What a behaviour wants the robot to do if its candidate wins.

The brain's own counterpart of a contract ``Action``; it never crosses into the simulation.
``Treat``, ``Charge`` and ``Idle`` pass through unchanged. ``Move`` is deliberately absent: a
behaviour names the field to follow with ``FollowField``, and only the navigator decides the
step, for the winning candidate.
"""


@dataclass(frozen=True)
class ActionCandidate:
    """A behaviour's bid at a decision point: what it wants done, and how much.

    Attributes:
        action: What the robot should do if this candidate wins.
        score: How much the behaviour wants it; the highest score wins. The scale shared by all
            behaviours is settled together with the arbiter (#39).
        source: The name of the proposing behaviour.
        target: The cell this candidate commits the robot to, such as the field's target at the
            robot's cell, or ``None``. The arbiter uses it to tell a new target from the one
            the robot is already heading for.
    """

    action: Intent
    score: float
    source: str
    target: CellId | None


@dataclass(frozen=True)
class DecisionContext:
    """Read-only view handed to every behaviour at a decision point.

    Attributes:
        time: The time of the decision.
        self_state: The robot's perception of itself.
        sensed: What the robot sensed at this decision point (already merged into memory).
        memory: The robot's facts about the field and its peers.
        fields: Hands out the grid potential fields of the registered seeders by name, through
            ``field_for``. Behaviours read fields; they never move the robot themselves.
        inbox: The messages received at this decision point.
    """

    time: Time
    self_state: RobotSelfState
    sensed: list[Observation]
    memory: MemoryView
    fields: Navigator
    inbox: list[Message] = field(default_factory=list)


class Behavior(Protocol):
    """A single composable piece of decision logic, bidding for what the robot does next.

    At each decision point a behaviour reads the context and returns its candidates, or an
    empty tuple when it has nothing to propose. It decides whether it wants the robot, how much
    (``score``), and what to do if it wins (``action``). It may read the fields of seeders
    through ``ctx.fields``, e.g. to score from the potential at the robot's cell.

    A behaviour never moves the robot itself. To move, it proposes ``FollowField`` with a
    seeder's name, and the navigator steps along that field only if the candidate wins. A
    behaviour standing on its target (``field.target_for(position) == position``) proposes its
    on-the-spot action instead, such as ``Treat`` or ``Charge``.
    """

    name: str

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]: ...


class TreatInfestation:
    """Skeleton of the pest-response behaviour: follow the field of the TREAT seeder."""

    name: str = "treat-infestation"

    def __init__(self, seeder: str = TreatSeeder.name) -> None:
        self._seeder: str = seeder

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        position = ctx.self_state.position
        seeded = ctx.fields.field_for(self._seeder, position, ctx.time)
        if seeded is None:
            return ()
        target = seeded.field.target_for(position)
        if target == position:
            # TODO (#40): standing on a pest: propose Treat(cell=position).
            return ()
        # TODO (#39): a score on the scale shared by all behaviours. Lower potential is better,
        # so negate it for now.
        score = -seeded.field.value_at(position)
        candidate = ActionCandidate(
            action=FollowField(self._seeder), score=score, source=self.name, target=target
        )
        return (candidate,)
    
    
class ExploreBehaviour:
    """Skeleton of the exploration behaviour: follow the field of the EXPLORE seeder."""

    name: str = "explore"

    def __init__(self, seeder: str = ExploreSeeder.name) -> None:
        self._seeder: str = seeder
        self._max_candidates: int = 16 # Maximum number of candidates to propose
        self._dispersion_robot_threshold: int = 15 # Minimum number of robots to include dispersion score
        self._separation_sigma: float = 2.0 # It will only influence robots within 3 sigma
        self._visible_offset: float = 1.0 # Assume robots can see cells within 1 cell in each direction (3x3 area)
        self._peer_forbidden_distance: float = 1.0 # Minimum distance to other robots
        self.weight_visible: float = 0.8 # Weight for visible exploration score
        self.weight_dispersion: float = 0.1 # Weight for dispersion score
        self.weight_separation: float = 0.3 # Weight for separation score

    def propose(self, ctx: DecisionContext) -> tuple[ActionCandidate, ...]:
        position = ctx.self_state.position
        seeded = ctx.fields.field_for(self._seeder, position, ctx.time)
        field = seeded.field
        graph = field.graph
        window = graph.window
        peers = tuple(peer.cell for peer in ctx.memory.peers())
        peers_num = len(peers)
        top_candidates = self._top_candidates(field, graph, window, peers, count=self._max_candidates)
        if not top_candidates:
            spread_target = self._fast_spread(graph, peers)
            if spread_target is None:
                return ()
            return (
                ActionCandidate(
                    action=FollowField(self._seeder),
                    score=0.8,
                    source=self.name,
                    target=spread_target,
                ),
            )
        candidate_cells = [cell for _, cell in top_candidates]
        separation_scores = self._separation_score(candidate_cells, peers)
        visible_scores = self._visible_exploration_score(field, candidate_cells)
        if peers_num > self._dispersion_robot_threshold:
            dispersion_scores = self._dispersion_score(window, candidate_cells, peers)
        final_scores = [
            self.weight_visible * visible - self.weight_separation * separation
            for visible, separation in zip(visible_scores, separation_scores)
        ]
        if 'dispersion_scores' in locals():
            final_scores = [
                score - self.weight_dispersion * dispersion
                for score, dispersion in zip(final_scores, dispersion_scores)
            ]
        return tuple(
            ActionCandidate(
                action=FollowField(self._seeder),
                score=score,
                source=self.name,
                target=cell,
            )
            for score, cell in zip(final_scores, candidate_cells)
        )

    def _visible_exploration_score(
        self,
        field: GridPotentialField,
        candidates: Sequence[CellId],
    ) -> list[float]:
        """Average the field score over each candidate's visible square neighborhood.

        Cells without a finite field value contribute zero.
        """
        radius = ceil(self._visible_offset)
        offsets = tuple(
            (offset_x, offset_y)
            for offset_y in range(-radius, radius + 1)
            for offset_x in range(-radius, radius + 1)
            if abs(offset_x) <= self._visible_offset
            and abs(offset_y) <= self._visible_offset
        )

        scores: list[float] = []
        for candidate in candidates:
            score_total = 0.0
            for offset_x, offset_y in offsets:
                cell = (candidate[0] + offset_x, candidate[1] + offset_y)
                value = field.value_at(cell)
                if isfinite(value):
                    score_total += value
            scores.append(score_total / len(offsets))
        return scores

    def _top_candidates(
        self, field: GridPotentialField, graph: MixedGraph, window: BoundingBox, peers: Sequence[CellId], count: int = 16
    ) -> list[tuple[float, CellId]]:
        """Return the top field values at cell nodes outside peers' exclusion zones."""
        forbidden_cells: set[CellId] = set()
        radius = ceil(self._peer_forbidden_distance)
        for peer in peers:
            for offset_y in range(-radius, radius + 1):
                if abs(offset_y) > self._peer_forbidden_distance:
                    continue
                for offset_x in range(-radius, radius + 1):
                    if abs(offset_x) <= self._peer_forbidden_distance:
                        forbidden_cells.add((peer[0] + offset_x, peer[1] + offset_y))

        def eligible_values() -> Iterator[tuple[float, CellId]]:
            for y in range(window.y_min, window.y_max):
                for x in range(window.x_min, window.x_max):
                    cell = (x, y)
                    if cell in forbidden_cells or graph.node_at(cell) != cell:
                        continue
                    value = field.value_at(cell)
                    if isfinite(value):
                        yield (value, cell)

        return nlargest(count, eligible_values(), key=lambda candidate: candidate[0])

    def _separation_score(self, candidates: Sequence[CellId], peers: Sequence[CellId]) -> list[float]:
        """Generate a separation score when there are few robots nearby"""
        sigma_squared = self._separation_sigma**2
        if not peers or sigma_squared == 0.0:
            return [0.0 for _ in candidates]
        max_distance_squared = 9 * sigma_squared
        scores: list[float] = []
        for candidate in candidates:
            nearby_distances_squared = [
                (candidate[0] - peer[0]) ** 2 + (candidate[1] - peer[1]) ** 2
                for peer in peers
                if (candidate[0] - peer[0]) ** 2 + (candidate[1] - peer[1]) ** 2
                <= max_distance_squared
            ]
            if not nearby_distances_squared:
                scores.append(0.0)
                continue
            proximity = sum(
                exp(-distance_squared / (2 * sigma_squared))
                for distance_squared in nearby_distances_squared
            ) / len(nearby_distances_squared)
            scores.append(1.0 - proximity)
        return scores

    def _dispersion_score(
        self, window: BoundingBox, candidates: Sequence[CellId], peers: Sequence[CellId],
    ) -> list[float]:
        """Generate a dispersion score when there are many robots nearby"""
        map_size = (window.x_max - window.x_min, window.y_max - window.y_min)
        map_diagonal = hypot(map_size[0] - 1, map_size[1] - 1)
        if not peers or map_diagonal == 0.0:
            return [0.0 for _ in candidates]
        return [
            sum(hypot(cell[0] - x, cell[1] - y) for x, y in peers)
            / (len(peers) * map_diagonal)
            for cell in candidates
        ]
        
    def _fast_spread(self, graph: MixedGraph, peers: Sequence[CellId]) -> CellId | None:
        """Choose the least-crowded traversable cell on the field window's boundary."""
        if len(peers) <= self._dispersion_robot_threshold:
            return None

        max_distance_squared = (3 * self._separation_sigma) ** 2
        best_cell: CellId | None = None
        best_nearby_count: int | None = None
        for cell in graph.window.edge_cells():
            if graph.node_at(cell) != cell:
                continue
            nearby_count = sum(
                (cell[0] - peer[0]) ** 2 + (cell[1] - peer[1]) ** 2
                <= max_distance_squared
                for peer in peers
            )
            if best_nearby_count is None or nearby_count < best_nearby_count:
                best_cell = cell
                best_nearby_count = nearby_count
        return best_cell
