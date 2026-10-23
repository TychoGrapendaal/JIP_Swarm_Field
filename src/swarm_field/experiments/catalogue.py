"""Every scenario by name: the presets ``S1`` to ``S5`` and the hand-built experiments.

A name and a seed are all a sweep or a viewer needs to build a run: ``build_named_experiment``
turns them into the ``ExperimentConfig`` that ``swarm_field.runner`` assembles and runs, labelled
``NAME/SEED`` and with the name recorded as its ``scenario_name``. Each scenario keeps its own
horizon unless the caller sets one: an hour for the presets, the sweeps' default, and the length
``ExploreScenario`` and ``PestScenario`` define for theirs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace

from swarm_field.contracts import BodyType, ScenarioId, Time, TimeUnit
from swarm_field.experiments.explore import ExploreScenario
from swarm_field.experiments.pests import PestScenario
from swarm_field.runner import BrainSpec, ExperimentConfig, RunSummary, build_experiment, sweep

ScenarioBuilder = Callable[[int, Time | None], ExperimentConfig]
"""Builds one scenario's run from a seed and a horizon; ``None`` keeps the scenario's own."""

PRESET_HORIZON: Time = TimeUnit.HOUR
"""How long a preset runs when the caller does not say."""


PRESET_BRAIN = BrainSpec(behaviors=("explore",), seeders=("explore",))
"""The brain of every preset robot whose body has no brain of its own: it explores."""

PRESET_KILLER_BRAIN = BrainSpec(
    behaviors=("treat-infestation", "explore"), seeders=("treat", "explore")
)
"""A preset killer weighs treating the pests it knows of against exploring."""


def _preset(scenario: ScenarioId) -> ScenarioBuilder:
    def build(seed: int, until: Time | None) -> ExperimentConfig:
        horizon = PRESET_HORIZON
        if until is not None:
            horizon = until
        config = build_experiment(
            f"{scenario.name}/{seed}", scenario, seed, brain=PRESET_BRAIN, until=horizon
        )
        return replace(config, brains_by_body={BodyType.KILLER: PRESET_KILLER_BRAIN})

    return build


def _explore(seed: int, until: Time | None) -> ExperimentConfig:
    scenario = ExploreScenario(seed=seed)
    if until is not None:
        scenario = replace(scenario, until=until)
    return replace(scenario.into_experiment(f"explore/{seed}"), scenario_name="explore")


def _pests(seed: int, until: Time | None) -> ExperimentConfig:
    scenario = PestScenario(seed=seed)
    if until is not None:
        scenario = replace(scenario, until=until)
    return replace(scenario.into_experiment(f"pests/{seed}"), scenario_name="pests")


LARGE_PESTS = PestScenario(
    size=256, scouts=100, killers=5, pests_per_hour=240.0, sample_period=TimeUnit.MINUTE
)
"""``pests`` at scale: 100 robots on a 128 x 128 field (1.28 km a side).

The field is 16 times the area of ``pests``, so the pests spawn 16 times as often to keep their
density; the robots per cell are about the same. Metrics are sampled once a minute, since every
sample reads every cell.
"""


def _large_pests(seed: int, until: Time | None) -> ExperimentConfig:
    scenario = replace(LARGE_PESTS, seed=seed)
    if until is not None:
        scenario = replace(scenario, until=until)
    label = f"pests-large/{seed}"
    return replace(scenario.into_experiment(label), scenario_name="pests-large")


def _build_catalogue() -> dict[str, ScenarioBuilder]:
    catalogue: dict[str, ScenarioBuilder] = {}
    for preset in ScenarioId:
        catalogue[preset.name] = _preset(preset)
    catalogue["explore"] = _explore
    catalogue["pests"] = _pests
    catalogue["pests-large"] = _large_pests
    return catalogue


_CATALOGUE: dict[str, ScenarioBuilder] = _build_catalogue()


def scenario_names() -> tuple[str, ...]:
    """The name of every scenario, the presets first."""
    return tuple(_CATALOGUE)


def build_named_experiment(name: str, seed: int, until: Time | None = None) -> ExperimentConfig:
    """The run of the scenario called ``name``, from ``seed``, labelled ``NAME/SEED``.

    Args:
        name: One of ``scenario_names()``.
        seed: The seed the run is a function of.
        until: How far to run, or ``None`` for the scenario's own horizon.

    Raises:
        KeyError: If no scenario has that name.
    """
    return _CATALOGUE[name](seed, until)


def sweep_scenarios(
    names: Iterable[str],
    seeds: Iterable[int],
    until: Time | None = None,
    sample_period: Time | None = None,
) -> tuple[RunSummary, ...]:
    """Run every named scenario with every seed, and return one summary per run, in that order.

    Args:
        names: Scenarios from ``scenario_names()``.
        seeds: The seeds every scenario is run with.
        until: How far to run every scenario, or ``None`` for each scenario's own horizon.
        sample_period: The time between two metrics samples in every run, or ``None`` for
            each scenario's own. The rest of each scenario's metrics settings is kept.
    """
    every_seed = tuple(seeds)
    configs: list[ExperimentConfig] = []
    for name in names:
        for seed in every_seed:
            config = build_named_experiment(name, seed, until)
            if sample_period is not None:
                config = replace(config, metrics=replace(config.metrics, period=sample_period))
            configs.append(config)
    return sweep(configs)
