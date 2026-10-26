"""Every scenario by name: the presets ``S1`` to ``S5`` and the hand-built experiments.

A name and a seed are all an experiment or a viewer needs to build a run:
``build_named_experiment`` turns them into the ``ExperimentConfig`` that ``swarm_field.runner``
assembles and runs, labelled ``NAME/SEED`` and with the name recorded as its ``scenario_name``.
Each scenario keeps its own horizon unless the caller sets one: an hour for the presets, and the
length its ``SwarmScenario`` defines for the others.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from swarm_field.contracts import BodyType, ScenarioId, Time, TimeUnit
from swarm_field.experiments.maps import MapConfig
from swarm_field.experiments.presets import EXPLORER, SEEDED_KILLER
from swarm_field.experiments.rebalancing import (
    REBALANCING,
    REBALANCING_HOP,
    rebalancing_configurations,
)
from swarm_field.experiments.scenario import (
    EXPLORE,
    FAST_TREATMENT,
    PESTS,
    PestConfig,
    SwarmScenario,
)
from swarm_field.runner import BrainSpec, ExperimentConfig, build_experiment
from swarm_field.simulation import MetricsConfig

ScenarioBuilder = Callable[[int, Time | None], ExperimentConfig]
"""Builds one scenario's run from a seed and a horizon; ``None`` keeps the scenario's own."""

PRESET_HORIZON: Time = TimeUnit.HOUR
"""How long a preset runs when the caller does not say."""


PRESET_BRAIN = BrainSpec(behaviors=("explore", "return-to-dock"), seeders=("explore", "dock"))
"""The brain of every preset robot whose body has no brain of its own: it explores, and
recharges at the preset's dock when its battery runs low."""

PRESET_KILLER_BRAIN = BrainSpec(
    behaviors=("treat-infestation", "explore", "return-to-dock"),
    seeders=("treat", "explore", "dock"),
)
"""A preset killer weighs treating the pests it knows of against exploring, and recharges at
the preset's dock when its battery runs low."""


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
    scenario = replace(EXPLORE, seed=seed)
    if until is not None:
        scenario = replace(scenario, until=until)
    return replace(scenario.into_experiment(f"explore/{seed}"), scenario_name="explore")


def _pests(seed: int, until: Time | None) -> ExperimentConfig:
    scenario = replace(PESTS, seed=seed)
    if until is not None:
        scenario = replace(scenario, until=until)
    return replace(scenario.into_experiment(f"pests/{seed}"), scenario_name="pests")


LARGE_PESTS = (
    SwarmScenario(
        map=MapConfig(size=256),
        pests=PestConfig(pests_per_hour=240.0, treatment_rate=FAST_TREATMENT),
        metrics=MetricsConfig(period=TimeUnit.MINUTE),
    )
    .add_robots(EXPLORER, 100)
    .add_robots(SEEDED_KILLER, 5)
)
"""``pests`` at scale: 100 explorers and 5 killers on a 256 x 256 field (2.56 km a side).

The field is 64 times the area of ``pests`` and the pests spawn 16 times as often, so they are a
quarter as dense. Metrics are sampled once a minute, since every sample reads every cell.
"""


def _large_pests(seed: int, until: Time | None) -> ExperimentConfig:
    scenario = replace(LARGE_PESTS, seed=seed)
    if until is not None:
        scenario = replace(scenario, until=until)
    label = f"pests-large/{seed}"
    return replace(scenario.into_experiment(label), scenario_name="pests-large")


def _rebalancing(arm: str, name: str, base: SwarmScenario = REBALANCING) -> ScenarioBuilder:
    """The ``base`` swarm of arm ``fixed`` or ``rebalancing`` (#64), called ``name``."""

    def build(seed: int, until: Time | None) -> ExperimentConfig:
        scenario = replace(rebalancing_configurations(base)[arm], seed=seed)
        if until is not None:
            scenario = replace(scenario, until=until)
        return replace(scenario.into_experiment(f"{name}/{seed}"), scenario_name=name)

    return build


def _build_catalogue() -> dict[str, ScenarioBuilder]:
    catalogue: dict[str, ScenarioBuilder] = {}
    for preset in ScenarioId:
        catalogue[preset.name] = _preset(preset)
    catalogue["explore"] = _explore
    catalogue["pests"] = _pests
    catalogue["pests-large"] = _large_pests
    catalogue["rebalancing"] = _rebalancing("rebalancing", "rebalancing")
    catalogue["rebalancing-fixed"] = _rebalancing("fixed", "rebalancing-fixed")
    catalogue["rebalancing-hop"] = _rebalancing("rebalancing", "rebalancing-hop", REBALANCING_HOP)
    catalogue["rebalancing-hop-fixed"] = _rebalancing(
        "fixed", "rebalancing-hop-fixed", REBALANCING_HOP
    )
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
