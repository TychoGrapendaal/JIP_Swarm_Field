"""Actions crossing from the robot brain into the high-level world model."""

from dataclasses import dataclass

from swarm_field.contracts.ids import BodyType, CellId


@dataclass(frozen=True)
class Move:
    """Move directly toward a target cell.

    The robot does not turn first or execute one heading-relative step. The target is a
    field cell selected by the navigation / potential-field layer.

    Attributes:
        to: Target field cell.
    """

    to: CellId


@dataclass(frozen=True)
class Treat:
    """Treat an infested field patch.

    Attributes:
        cell: Target field cell to treat.
    """

    cell: CellId


@dataclass(frozen=True)
class Charge:
    """Recharge in place."""


@dataclass(frozen=True)
class SwitchRole:
    """Take on another role: the simulation changes the robot's body and its speed.

    Attributes:
        body: The body the robot switches to.
    """

    body: BodyType


@dataclass(frozen=True)
class Idle:
    """Do nothing until the next decision point."""


Action = Move | Treat | Charge | SwitchRole | Idle
