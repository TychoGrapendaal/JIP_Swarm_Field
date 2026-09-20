"""Navigation seam types — what A.1 hands A.3, and what A.4 gives A.3.

These fix the two boundaries of the navigation module (see ``docs/navigation-module.md``):
``NavGoal`` is the A.1 → A.3 goal; ``HopFieldEntry`` is the decoded hop-field reading A.4
provides A.3 each decision point (the beacon plus the robot's own radio measurement to its
holder). ``NavMode`` distinguishes the three v1 goals, exactly as the module spec defines
them. All shapes are frozen data so the seam stays a contract, not behaviour.
"""

from dataclasses import dataclass
from enum import Enum, auto

from swarm_field.contracts.ids import Time

Vector2 = tuple[float, float]
"""A relative displacement in grid cells, ``(dx, dy)`` with y increasing North.

Matches ``Direction.offset`` and the walkthrough vectors of ``docs/navigation-module.md``
§8.1 (the module's prose at §3 speaks of metres, but the implemented steering works in
cells).
"""


class NavMode(Enum):
    """The three navigation goals A.1 may issue (v1; no more).

    Members:
        HOLD: No useful goal right now — stand still and wait.
        SPREAD: Move along one fixed drift vector; there is no destination.
        CHASE: Commit to following the pest gradient (hop chain) until it resolves.
    """

    HOLD = auto()
    SPREAD = auto()
    CHASE = auto()


@dataclass(frozen=True)
class NavGoal:
    """What A.1 wants A.3 to do this decision point.

    Attributes:
        mode: Which of the three navigation modes to follow.
        drift: SPREAD only — the fixed relative drift vector, in cells; ``None`` otherwise.
        max_hop: CHASE only — pursue while the freshest heard hop is at most ``max_hop``;
            ``None`` otherwise.
    """

    mode: NavMode
    drift: Vector2 | None = None
    max_hop: int | None = None


@dataclass(frozen=True)
class HopFieldEntry:
    """One decoded hop-field reading: a beacon A.3 can steer on.

    These are what A.4 produces every decision point from the in-range explorers' messages
    and the robot's own radio measurements — one entry per qualifying in-range explorer.

    Attributes:
        holder_hop: The holder explorer's hop number; 0 means it sees the pest itself.
        parent_vector: The holder's vector to the next hop down (the beacon's payload).
        to_holder: The robot's measured relative vector to the holder (radio bearing).
        seen_at: Time of the beacon the entry was decoded from, for staleness.
    """

    holder_hop: int
    parent_vector: Vector2
    to_holder: Vector2
    seen_at: Time
