"""Objectives: which cells a robot wants to reach, and how much, read from its memory."""

from typing import Protocol

from swarm_field.contracts import Time
from swarm_field.swarm.field import Seed
from swarm_field.swarm.memory import MemoryView


class Objective(Protocol):
    """Turns memory into the targets of one potential field (TREAT, EXPLORE, RECHARGE, ...)."""

    name: str

    def seeds(self, memory: MemoryView, now: Time) -> tuple[Seed, ...]: ...


class TreatObjective:
    """Mock of the TREAT objective: known pests, weighted by severity and confidence."""

    name: str = "treat"

    def seeds(self, memory: MemoryView, now: Time) -> tuple[Seed, ...]:
        # TODO: one seed per known pest, with start value -beta * severity * confidence (see
        # docs/robot-memory-and-navigation.md, "How to use it"). Claimed pests are skipped once
        # claims exist. For now there are no targets.
        return ()
