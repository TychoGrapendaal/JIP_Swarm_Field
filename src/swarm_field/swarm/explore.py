
"""Score exploration targets from a robot's detailed local map."""

from collections.abc import Sequence
from math import exp, hypot, isfinite

from swarm_field.contracts import CellId
from swarm_field.swarm import ActionCandidate, FollowField

_MAP_SIDE = 16
_CANDIDATE_COUNT = 16
_DISPERSION_ROBOT_THRESHOLD = 10


class ExploreBehaviour:
	"""Rank promising cells and score them for coverage and swarm spacing.

	The local map is indexed as ``values[y][x]``; ``map_origin`` is the world coordinate of
	``values[0][0]``. Visibility offsets describe the sensor footprint relative to a proposed
	target. Peer positions must contain other robots only, not this robot.
	"""

	def __init__(
		self,
		exploration_weight: float = 1.0,
		separation_weight: float = 1.0,
		dispersion_weight: float = 1.0,
		separation_sigma: float = 2.0,
	) -> None:
		if separation_sigma <= 0.0:
			raise ValueError("separation_sigma must be positive")
		self.exploration_weight: float = exploration_weight
		self.separation_weight: float = separation_weight
		self.dispersion_weight: float = dispersion_weight
		self.separation_sigma: float = separation_sigma

	def score_candidates(
		self,
		exploration_values: Sequence[Sequence[float]],
		map_origin: CellId,
		visibility_offsets: Sequence[CellId],
		peer_positions: Sequence[CellId],
	) -> tuple[ActionCandidate, ...]:
		"""Return up to 16 candidates, ordered by weighted utility, highest first."""
		self._validate_inputs(exploration_values, visibility_offsets)
		values = tuple(tuple(row) for row in exploration_values)
		cells = [
			(value, (x, y))
			for y, row in enumerate(values)
			for x, value in enumerate(row)
		]
		cells.sort(key=lambda item: (-item[0], item[1][1], item[1][0]))

		include_dispersion = len(peer_positions) + 1 > _DISPERSION_ROBOT_THRESHOLD
		candidates: list[ActionCandidate] = []
		for _, local_cell in cells[:_CANDIDATE_COUNT]:
			target = (map_origin[0] + local_cell[0], map_origin[1] + local_cell[1])
			exploration_score = self._visible_exploration_score(
				values, local_cell, visibility_offsets
			)
			separation_score = self._separation_score(target, peer_positions)
			dispersion_score = 0.0
			if include_dispersion:
				dispersion_score = self._dispersion_score(target, peer_positions)
			utility = (
				self.exploration_weight * exploration_score
				+ self.separation_weight * separation_score
				+ self.dispersion_weight * dispersion_score
			)
			candidates.append(ActionCandidate(action=FollowField(target), score=utility))

		candidates.sort(key=lambda candidate: (-candidate.score, candidate.action.to))
		return tuple(candidates)

	def _visible_exploration_score(
		self,
		values: tuple[tuple[float, ...], ...],
		candidate: CellId,
		visibility_offsets: Sequence[CellId],
	) -> float:
		visible_values: list[float] = []
		for offset_x, offset_y in visibility_offsets:
			x = candidate[0] + offset_x
			y = candidate[1] + offset_y
			if 0 <= x < _MAP_SIDE and 0 <= y < _MAP_SIDE:
				visible_values.append(values[y][x])
		if not visible_values:
			return 0.0
		return sum(visible_values) / len(visible_values)

	def _separation_score(self, candidate: CellId, peers: Sequence[CellId]) -> float:
		if not peers:
			return 0.0
		sigma_squared = self.separation_sigma**2
		proximity = sum(
			exp(
				-((candidate[0] - peer[0]) ** 2 + (candidate[1] - peer[1]) ** 2)
				/ (2 * sigma_squared)
			)
			for peer in peers
		) / len(peers)
		return 1.0 - proximity

	def _dispersion_score(self, candidate: CellId, peers: Sequence[CellId]) -> float:
		map_diagonal = hypot(_MAP_SIDE - 1, _MAP_SIDE - 1)
		mean_distance = sum(hypot(candidate[0] - x, candidate[1] - y) for x, y in peers)
		return min(1.0, mean_distance / (len(peers) * map_diagonal))

	@staticmethod
	def _validate_inputs(
		exploration_values: Sequence[Sequence[float]], visibility_offsets: Sequence[CellId]
	) -> None:
		if len(exploration_values) != _MAP_SIDE or any(
			len(row) != _MAP_SIDE for row in exploration_values
		):
			raise ValueError("exploration_values must be a 16-by-16 grid")
		if not visibility_offsets:
			raise ValueError("visibility_offsets must not be empty")
		if any(not isfinite(value) for row in exploration_values for value in row):
			raise ValueError("exploration values must be finite")