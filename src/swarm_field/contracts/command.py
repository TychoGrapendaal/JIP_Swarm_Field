"""Interaction commands crossing from the visualization (C) into the simulation (B).

``Command`` is the C → B port: the only way the frontend influences a run. World-affecting
commands are applied by the simulation; time-pacing commands describe how the frontend
advances the simulation clock (pause, step once, speed multiplier) and are interpreted by its
render loop.
"""

from dataclasses import dataclass

from swarm_field.contracts.ids import CellId


@dataclass(frozen=True)
class InjectInfestation:
    """Inject a pest outbreak of ``severity`` at ``cell`` (C.1 interactive control).

    Attributes:
        cell: Field cell to infest.
        severity: Starting infestation severity, 0..1.
    """

    cell: CellId
    severity: float

    def __post_init__(self) -> None:
        """Validate the requested severity range."""
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be between 0 and 1")


@dataclass(frozen=True)
class SetTimeScale:
    """Change how fast simulation time advances relative to the render clock.

    Attributes:
        factor: Simulated-time multiplier per rendered frame, strictly positive.
    """

    factor: float

    def __post_init__(self) -> None:
        if self.factor <= 0:
            raise ValueError("time-scale factor must be positive")


@dataclass(frozen=True)
class Pause:
    """Stop advancing simulation time; rendering keeps sampling the current time."""


@dataclass(frozen=True)
class StepOnce:
    """Advance by exactly the next pending simulation event, then stop."""


Command = InjectInfestation | SetTimeScale | Pause | StepOnce
