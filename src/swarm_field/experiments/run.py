"""One experiment's run, assembled by ``swarm_field.runner`` and advanced step by step.

A frontend such as ``swarm_field.console`` drives a ``LiveRun`` instead of letting the runner run
it to the end: it advances the clock in slices, applies the commands a user gives, and draws the
world in between. Everything that belongs to the run itself lives here, so every frontend steps,
records and replays a run the same way.
"""

from __future__ import annotations

from swarm_field.contracts import (
    CellId,
    CommandLog,
    InjectInfestation,
    RobotId,
    Time,
    TimedCommand,
    WorldSnapshot,
)
from swarm_field.runner import ExperimentConfig, build_run
from swarm_field.simulation import CommandReplay, MetricsCollector, SimulationContext
from swarm_field.swarm import ComposedAgent


class LiveRun:
    """One experiment's run, advanced in time by the caller, that records the commands applied.

    The clock always advances through a ``CommandReplay``: a new run replays an empty log and a
    replayed run its recording, so both take exactly the same path through the run.

    Attributes:
        config: What was assembled; its ``until`` is the run's horizon.
        context: The simulation's state; its world is the ground truth.
        metrics: The run's metrics, sampled at ``config.metrics.period``.
        replaying: Whether the run plays back a recording, rather than starting fresh.
        recorded: The commands applied to this run so far, in order.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        context: SimulationContext,
        metrics: MetricsCollector,
        log: CommandLog | None = None,
    ) -> None:
        self.config: ExperimentConfig = config
        self.context: SimulationContext = context
        self.metrics: MetricsCollector = metrics
        self.replaying: bool = log is not None
        self.recorded: list[TimedCommand] = []
        if log is None:
            log = CommandLog(seed=config.scenario.seed, until=0, commands=())
        self._replay: CommandReplay = CommandReplay(context, log)

    @classmethod
    def from_config(cls, config: ExperimentConfig, log: CommandLog | None = None) -> LiveRun:
        """Assemble the run of ``config`` with ``runner.build_run``, replaying ``log`` if given.

        The run is built without a renderer: a frontend draws on its own clock, and a renderer
        handed to the run would draw on every metric tick as well.
        """
        context, metrics = build_run(config)
        return cls(config, context, metrics, log)

    @property
    def now(self) -> Time:
        """The simulation time the run has reached."""
        return self.context.now

    @property
    def horizon(self) -> Time:
        """The simulation time the run ends at."""
        return self.config.until

    @property
    def label(self) -> str:
        return self.config.label

    def advance_to(self, time: Time) -> None:
        """Process every event up to ``time``, applying the recorded commands on the way."""
        self._replay.advance_to(time)

    def step(self) -> None:
        """Process the events of the next pending timestamp, and nothing more."""
        self._replay.step()

    def apply_command(self, command: InjectInfestation) -> None:
        """Apply ``command`` to the world now, and record it with the current time."""
        self.context.apply_command(command)
        self.recorded.append(TimedCommand(time=self.now, command=command))

    def command_log(self) -> CommandLog:
        """The recording of this run so far: its scenario, its seed and the commands applied.

        Replaying it reproduces the run exactly; a replayed run records no new commands, so its
        log is the empty log of its seed.
        """
        scenario: str | None = self.config.scenario_name
        if not scenario:
            scenario = None
        return CommandLog(
            seed=self.config.scenario.seed,
            until=self.now,
            commands=tuple(self.recorded),
            scenario=scenario,
        )

    def snapshot(self) -> WorldSnapshot:
        """The ground truth at the time the run has reached, with its latest metrics sample."""
        return self.context.world.snapshot(self.now, self.metrics.latest())

    def snapshot_at(self, time: Time) -> WorldSnapshot:
        """Advance to ``time`` and return the ground truth then, for rendering."""
        self.advance_to(time)
        return self.snapshot()

    def agents(self) -> dict[RobotId, ComposedAgent]:
        """The robots' brains, by id."""
        found: dict[RobotId, ComposedAgent] = {}
        for robot_id, agent in self.context.agents.items():
            if isinstance(agent, ComposedAgent):
                found[robot_id] = agent
        return found

    def targets(self) -> dict[RobotId, CellId]:
        """The cell each robot is heading for, from its last decision."""
        found: dict[RobotId, CellId] = {}
        for robot_id, agent in self.context.agents.items():
            target = agent.inspect().current_target
            if target is not None:
                found[robot_id] = target
        return found
