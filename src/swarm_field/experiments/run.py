"""A run assembled by ``swarm_field.runner`` and advanced step by step, e.g. for a viewer."""

from __future__ import annotations

from dataclasses import dataclass, field

from swarm_field.contracts import CellId, RobotId, Time, WorldSnapshot
from swarm_field.runner import ExperimentConfig, build_run
from swarm_field.simulation import MetricsCollector, SimulationContext, SimulationLoop
from swarm_field.swarm import ComposedAgent


@dataclass
class LiveRun:
    """One experiment's run, advanced in time by the caller.

    Attributes:
        config: What was assembled.
        context: The simulation's state; its world is the ground truth.
        metrics: The run's metrics, sampled at ``config.metrics.period``.
        loop: The event loop that advances the run.
    """

    config: ExperimentConfig
    context: SimulationContext
    metrics: MetricsCollector
    loop: SimulationLoop = field(default_factory=SimulationLoop)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> LiveRun:
        """Assemble the run of ``config`` with ``runner.build_run``."""
        context, metrics = build_run(config)
        return cls(config, context, metrics)

    def advance_to(self, time: Time) -> None:
        """Process every event up to ``time``."""
        self.loop.run(self.context, until=time)

    def snapshot_at(self, time: Time) -> WorldSnapshot:
        """Advance to ``time`` and return the ground truth then, for rendering."""
        self.advance_to(time)
        return self.context.world.snapshot(time)

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
