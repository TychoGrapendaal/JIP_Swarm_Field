"""Keeping every robot in radio contact with at least one other while it moves.

A robot's step is checked before it is taken. It is safe when, from the cell it leads to,

1. some peer is within ``link_cells``, so the robot keeps a neighbour, and
2. every anchor (a peer that reported the objective ``VERIFY``, waiting on a pest) in radio
   range now that has no other neighbour, as far as this robot knows, stays within
   ``link_cells``: an anchor cannot follow, so the robot does not walk away from one that only
   it connects. A moving peer is not protected this way; it keeps a neighbour itself.

``link_cells`` is the radio range less a margin, because the peer may step away too before
the next report. Positions come from memory: the peers' own reports and those relayed in their
memory snapshots, younger than ``fresh_age``.

An unsafe step is replaced by the safe neighbouring cell lowest on the robot's own potential
field, so it still follows the route around obstacles, or by staying; without a field, by the
safe cell nearest to where the robot was going. A robot that has no safe cell, or has heard no
peer within ``fresh_age`` because it is already cut off, steps towards the nearest peer it knows
of, by its last report. Only steps are checked: a robot
that treats, charges or waits stays where it is.
"""

import logging
from dataclasses import dataclass
from math import dist, isfinite

from swarm_field.contracts import (
    Action,
    AgentObjective,
    CellId,
    Idle,
    Move,
    PeerObservation,
    Terrain,
    Time,
    TimeUnit,
)
from swarm_field.swarm.field import GridPotentialField
from swarm_field.swarm.memory import MemoryView

logger = logging.getLogger(__name__)

_STEPS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


@dataclass(frozen=True)
class TetherConfig:
    """How far robots may drift apart.

    Attributes:
        radio_cells: The radio range in cells.
        margin_cells: How far inside the radio range a link must stay, for the peer stepping
            away too before its next report. A robot steps at most one cell, diagonally about
            1.4, per decision.
        fresh_age: Peer positions older than this are not trusted to hold a link.
        return_age: A robot out of contact heads back to where a peer was reported at most
            this long ago; past that the peer has moved on too far for it to be worth it, and
            the robot goes about its work until it meets one.
    """

    radio_cells: float
    margin_cells: float = 1.5
    fresh_age: Time = 30 * TimeUnit.SECOND
    return_age: Time = 2 * TimeUnit.MINUTE

    @property
    def link_cells(self) -> float:
        """The distance within which a link is kept."""
        return self.radio_cells - self.margin_cells


def tethered_action(
    action: Action,
    robot: CellId,
    goal: CellId | None,
    memory: MemoryView,
    now: Time,
    config: TetherConfig,
    potential: GridPotentialField | None = None,
) -> Action:
    """``action``, or a step that keeps the robot and its lone neighbours in radio contact.

    Args:
        action: What the robot was going to do.
        robot: Its cell.
        goal: Where it was heading, to pick the safe cell nearest to it; the step's cell if
            ``None``.
        memory: Where its peers are.
        now: The time of the decision.
        config: The ranges.
        potential: The field the robot follows, to pick the safe cell lowest on it; ``None``
            picks the safe cell nearest to ``goal``.
    """
    if not isinstance(action, Move):
        return action
    peers = _fresh_peers(memory, now, config)
    if not peers:
        # No neighbour has been heard lately: the robot is out of contact, so it heads back.
        return _towards_nearest_peer(robot, memory, now, [], action, config)
    fresh: dict[int, CellId] = {}
    for peer, report in peers.items():
        fresh[peer] = report.cell
    lone = _lone_anchors(robot, peers, config)
    if _is_safe(action.to, fresh, lone, config):
        return action

    aim = action.to
    if goal is not None:
        aim = goal
    safe: list[CellId] = []
    for dx, dy in _STEPS:
        cell = (robot[0] + dx, robot[1] + dy)
        enterable = (dx, dy) == (0, 0) or memory.terrain_at(cell) is Terrain.FREE
        if enterable and _is_safe(cell, fresh, lone, config):
            safe.append(cell)
    if safe:
        best = min(safe, key=lambda cell: _step_cost(cell, robot, aim, potential))
        if best == robot:
            return Idle()
        return Move(to=best)
    return _towards_nearest_peer(robot, memory, now, list(fresh.values()), action, config)


def _step_cost(
    cell: CellId, robot: CellId, aim: CellId, potential: GridPotentialField | None
) -> tuple[int, float, bool]:
    """How good a safe cell is: low on the field first, else near ``aim``; staying last.

    A cell the field does not reach ranks after every cell it does.
    """
    if potential is not None:
        value = potential.value_at(cell)
        if isfinite(value):
            return (0, value, cell != robot)
    return (1, dist(cell, aim), cell != robot)


def _fresh_peers(memory: MemoryView, now: Time, config: TetherConfig) -> dict[int, PeerObservation]:
    """The peers reported within ``fresh_age``, by peer."""
    peers: dict[int, PeerObservation] = {}
    for peer in memory.peers():
        if peer.peer is not None and now - peer.observed_at <= config.fresh_age:
            peers[peer.peer] = peer
    return peers


def _lone_anchors(
    robot: CellId, peers: dict[int, PeerObservation], config: TetherConfig
) -> list[CellId]:
    """The cells of the anchors in radio range now with no other neighbour but this robot."""
    lone: list[CellId] = []
    for peer, report in peers.items():
        if report.objective is not AgentObjective.VERIFY:
            continue
        if dist(report.cell, robot) > config.radio_cells:
            continue
        others = False
        for other, other_report in peers.items():
            if other != peer and dist(other_report.cell, report.cell) <= config.link_cells:
                others = True
                break
        if not others:
            lone.append(report.cell)
    return lone


def _is_safe(
    cell: CellId, fresh: dict[int, CellId], lone: list[CellId], config: TetherConfig
) -> bool:
    """Whether from ``cell`` the robot keeps a neighbour and stays next to every lone one."""
    for lonely in lone:
        if dist(cell, lonely) > config.link_cells:
            return False
    return any(dist(cell, peer_cell) <= config.link_cells for peer_cell in fresh.values())


def _towards_nearest_peer(
    robot: CellId,
    memory: MemoryView,
    now: Time,
    fresh: list[CellId],
    action: Move,
    config: TetherConfig,
) -> Action:
    """A step towards the nearest peer reported within ``return_age``, to win back contact.

    ``action`` if there is none, or if the robot is already within ``link_cells`` of where it
    was reported and has still not heard it: the peer has moved on, so waiting there would
    strand the robot; it goes about its work until it meets one.
    """
    known: list[CellId] = list(fresh)
    for peer in memory.peers():
        recent = now - peer.observed_at <= config.return_age
        if isinstance(peer, PeerObservation) and peer.peer is not None and recent:
            known.append(peer.cell)
    if not known:
        return action
    nearest = min(known, key=lambda cell: dist(cell, robot))
    if dist(nearest, robot) <= config.link_cells:
        return action
    best = robot
    for dx, dy in _STEPS:
        cell = (robot[0] + dx, robot[1] + dy)
        if memory.terrain_at(cell) is Terrain.FREE and dist(cell, nearest) < dist(best, nearest):
            best = cell
    if best == robot:
        logger.warning("Robot at %s is out of contact and cannot step towards %s", robot, nearest)
        return Idle()
    return Move(to=best)
