"""Robustness runs: a scenario of the catalogue under robot failures or position error.

A ``RobustnessCase`` is one condition: a label, how often robots fail and how, and how often
they mislocate themselves. ``RobustnessRunSpec`` names a scenario and a case, and
``measure_robustness`` is a run function for ``Experiment.run``: it makes the run with the run's
seed and returns its last metrics sample, which says how many robots are left and how much of the
field they still cover. Faults are drawn from a random source of their own, so a case and its
baseline run with the same seed see the same pests::

    cases = default_robustness_cases()
    arms = {case.label: RobustnessRunSpec("pests", hours=1.0, case=case) for case in cases}
    results = Experiment("robustness").run(runs_for(arms, seeds=[101]), measure_robustness)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from swarm_field.contracts import (
    FailureMode,
    FailureSpec,
    MetricsSample,
    PositionErrorSpec,
    Time,
    TimeUnit,
)
from swarm_field.experiments.batch import Run
from swarm_field.experiments.catalogue import build_named_experiment
from swarm_field.runner import ExperimentConfig, run_experiment


@dataclass(frozen=True)
class RobustnessCase:
    """One condition a robustness run is made under. Both probabilities at zero is the baseline.

    Attributes:
        label: Name of the condition, used as the arm name.
        failures: How often robots fail, and how.
        position_error: How often robots mislocate themselves.
    """

    label: str
    failures: FailureSpec = field(default_factory=FailureSpec)
    position_error: PositionErrorSpec = field(default_factory=PositionErrorSpec)

    def apply(self, config: ExperimentConfig) -> ExperimentConfig:
        """``config`` under this condition, labelled ``NAME/CASE``; ``config`` is left as it is."""
        name = config.scenario_name
        if not name:
            name = config.label
        scenario = replace(
            config.scenario, failures=self.failures, position_error=self.position_error
        )
        return replace(config, label=f"{name}/{self.label}", scenario=scenario)


def default_robustness_cases(
    failure_probabilities: Iterable[float] = (0.001, 0.005),
    position_error_probabilities: Iterable[float] = (0.1, 0.3),
    modes: Iterable[FailureMode] = (FailureMode.DEAD_BATTERY,),
    failure_period: Time = TimeUnit.MINUTE,
) -> tuple[RobustnessCase, ...]:
    """A baseline, then one axis at a time: each failure mode at each rate, then position error.

    Only one axis is perturbed at a time, so each case answers one question: how much does the
    swarm lose to this fault, at this rate, on its own.
    """
    cases = [RobustnessCase(label="baseline")]
    for mode in modes:
        for probability in failure_probabilities:
            cases.append(
                RobustnessCase(
                    label=f"failure-{mode.name.lower()}-{probability:g}",
                    failures=FailureSpec(period=failure_period, probability=probability, mode=mode),
                )
            )
    for probability in position_error_probabilities:
        cases.append(
            RobustnessCase(
                label=f"position-error-{probability:g}",
                position_error=PositionErrorSpec(probability=probability),
            )
        )
    return tuple(cases)


@dataclass(frozen=True)
class RobustnessRunSpec:
    """A scenario of the catalogue under one condition, for a run function.

    Attributes:
        scenario: The scenario's name in the catalogue.
        hours: How long a run lasts.
        case: The condition.
        sample_period: The time between two metrics samples.
    """

    scenario: str
    hours: float
    case: RobustnessCase
    sample_period: Time = TimeUnit.HOUR

    def into_experiment(self, seed: int) -> ExperimentConfig:
        """The run of this spec with ``seed``, as ``Run`` has it."""
        config = build_named_experiment(self.scenario, seed, int(self.hours * TimeUnit.HOUR))
        config = replace(config, metrics=replace(config.metrics, period=self.sample_period))
        return self.case.apply(config)


def measure_robustness(run: Run[RobustnessRunSpec]) -> MetricsSample:
    """Make the run with the run's seed and return its last metrics sample; a run function."""
    return run_experiment(run.config.into_experiment(run.seed))[-1]
