"""Interaction commands crossing from the visualization (C) into the simulation (B).

``Command`` is the C → B port: the only way the frontend influences a run. World-affecting
commands are applied by the simulation; time-pacing commands describe how the frontend
advances the simulation clock (pause, step once, speed multiplier) and are interpreted by its
render loop. A ``CommandLog`` records the world-affecting commands of a seeded run, so the run
can be replayed exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from swarm_field.contracts.ids import CellId, Time


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


@dataclass(frozen=True)
class TimedCommand:
    """A world-affecting command and the simulation time it was applied at.

    Only world-affecting commands are recorded: time-pacing commands change how fast the
    frontend advances the clock, not what the run produces.

    Attributes:
        time: Simulation time of the application, after every event up to it was handled.
        command: The command that was applied.
    """

    time: Time
    command: InjectInfestation


@dataclass(frozen=True)
class CommandLog:
    """Everything a seeded interactive run needs to be replayed exactly.

    Attributes:
        seed: The seed the run was built from.
        until: Simulation time the recording ended at.
        commands: The applied commands, in the order they were applied.
        scenario: Name of the scenario preset the run was built from, or ``None`` for a run
            that a frontend builds itself and so needs only the seed.
    """

    seed: int
    until: Time
    commands: tuple[TimedCommand, ...]
    scenario: str | None = None

    def to_json(self) -> str:
        """The log as a JSON document, readable by ``from_json``."""
        commands = [
            {
                "time": timed.time,
                "inject_infestation": {
                    "cell": list(timed.command.cell),
                    "severity": timed.command.severity,
                },
            }
            for timed in self.commands
        ]
        document = {
            "scenario": self.scenario,
            "seed": self.seed,
            "until": self.until,
            "commands": commands,
        }
        return json.dumps(document, indent=2)

    @classmethod
    def from_json(cls, text: str) -> CommandLog:
        """Read a log written by ``to_json``.

        Raises:
            KeyError: If a field or command kind is missing from the document.
        """
        document = json.loads(text)
        commands: list[TimedCommand] = []
        for entry in document["commands"]:
            inject = entry["inject_infestation"]
            x, y = inject["cell"]
            command = InjectInfestation(cell=(int(x), int(y)), severity=float(inject["severity"]))
            commands.append(TimedCommand(time=int(entry["time"]), command=command))
        # Recordings from before the scenario was stored have no such key.
        scenario = document.get("scenario")
        if scenario is not None:
            scenario = str(scenario)
        return cls(
            seed=int(document["seed"]),
            until=int(document["until"]),
            commands=tuple(commands),
            scenario=scenario,
        )
