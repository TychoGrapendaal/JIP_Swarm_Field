"""The pest experiment with hop-based pest navigation, next to the memory-sharing baseline.

``HopPestScenario`` is ``PestScenario`` with one switch, ``navigation``:

- ``"hop"``: every robot is a ``HopAgent``. News of a pest travels only as ``PestBeacon``s,
  passed on hop by hop; killers follow the beacon with the fewest hops (``FollowBeacon`` on
  the beacon field) and explore otherwise. Peers' memory snapshots no longer carry pests.
- ``"gradient"``: every robot is a ``GradientAgent``. A scout that finds a pest stays on it
  as an anchor and sends hop 0; every robot passes on its hop count, and killers drive to the
  neighbour one hop nearer, down the gradient, with no pest position in any message.
- ``"memory"``: the existing pest experiment, where killers learn pests from peers' memory
  snapshots and follow the TREAT field.

Everything else is the same, so the three can be compared run for run. Killers drive at 0.5 m/s
here instead of the pest experiment's 0.1 m/s, which otherwise limits clearing far more than
how they learn of pests (#60).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import override

from swarm_field.contracts import BodyType, CellId
from swarm_field.experiments.pests import PestScenario
from swarm_field.runner import BrainSpec, ExperimentConfig
from swarm_field.swarm import GradientConfig, HopConfig, TetherConfig

NAVIGATIONS: tuple[str, ...] = ("gradient", "hop", "memory")
"""How killers learn of pests and go to them: down a hop gradient, by beacons, or from memory."""


@dataclass(frozen=True)
class HopPestScenario(PestScenario):
    """The pest experiment, with killers navigating to pests by a hop gradient, beacons or memory.

    Attributes:
        navigation: ``"gradient"``, ``"hop"`` or ``"memory"``, see ``NAVIGATIONS``.
        hop: How far and how long beacons travel, and how much killers want to follow them.
        gradient: How hop counts spread and age, how long anchors stay, and how much killers
            want to follow the gradient.
        speed_mps_ground: The killers' speed.
        layout: Where the robots start: ``"edge"`` side by side along the bottom row, or
            ``"centre"`` on the free cells nearest the middle of the field.
        keep_connected: Whether every robot checks its steps to stay in radio contact with a
            peer (``swarm.tether``), whatever the navigation. Off by default, so the three
            navigations compare as before; it costs memory sharing many treatments.
    """

    navigation: str = "hop"
    hop: HopConfig = field(default_factory=HopConfig)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    speed_mps_ground: float = 0.5
    keep_connected: bool = False
    layout: str = "edge"

    @override
    def starts(self) -> list[CellId]:
        """The robots' starting cells, by ``layout``.

        Raises:
            ValueError: If ``layout`` is unknown or the field has too few free cells.
        """
        if self.layout == "edge":
            return super().starts()
        if self.layout != "centre":
            raise ValueError(f"unknown layout: {self.layout}")
        blocked = set(self.blocked_cells())
        middle = (self.size - 1) / 2
        free: list[CellId] = []
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) not in blocked:
                    free.append((x, y))
        free.sort(key=lambda cell: ((cell[0] - middle) ** 2 + (cell[1] - middle) ** 2, cell))
        robots = self.scouts + self.killers
        if len(free) < robots:
            raise ValueError(f"the field has room for {len(free)} robots, not {robots}")
        return free[:robots]

    @override
    def into_experiment(self, label: str = "pests-hop") -> ExperimentConfig:
        """The run as an experiment for ``swarm_field.runner``.

        Raises:
            ValueError: If ``navigation`` is not one of ``NAVIGATIONS``.
        """
        if self.navigation not in NAVIGATIONS:
            raise ValueError(f"unknown navigation: {self.navigation}")
        config = self._navigation_experiment(super().into_experiment(label))
        if not self.keep_connected:
            return config
        tether = TetherConfig(radio_cells=self.radio_range_m / self.cell_size)
        explore = replace(self.explore, reach_cells=tether.link_cells)
        connected = {"tether": tether, "explore": explore}
        brains_by_body: dict[BodyType, BrainSpec] = {}
        for body, brain in config.brains_by_body.items():
            brains_by_body[body] = replace(brain, params={**brain.params, **connected})
        brain = replace(config.brain, params={**config.brain.params, **connected})
        return replace(config, brain=brain, brains_by_body=brains_by_body)

    def _navigation_experiment(self, config: ExperimentConfig) -> ExperimentConfig:
        """``config``, the plain pest experiment, with the brains of ``navigation``."""
        if self.navigation == "memory":
            return config
        if self.navigation == "gradient":
            return self._gradient_experiment(config)
        params = {"explore": self.explore, "hop": self.hop}
        scout = BrainSpec(
            behaviors=("explore",), seeders=("explore",), params=params, arbiter=self.arbiter
        )
        killer_behaviours: tuple[str, ...] = ("follow-beacon",)
        killer_seeders: tuple[str, ...] = ("beacon",)
        if self.killers_explore:
            killer_behaviours = ("follow-beacon", "explore")
            killer_seeders = ("beacon", "explore")
        killer = BrainSpec(
            behaviors=killer_behaviours, seeders=killer_seeders, params=params, arbiter=self.arbiter
        )
        return replace(config, brain=scout, brains_by_body={BodyType.KILLER: killer})

    def _gradient_experiment(self, config: ExperimentConfig) -> ExperimentConfig:
        """``config`` with scouts that anchor pests and killers that follow the hop gradient."""
        params = {"explore": self.explore, "gradient": self.gradient}
        scout = BrainSpec(
            behaviors=("anchor", "explore"),
            seeders=("explore",),
            params=params,
            arbiter=self.arbiter,
        )
        killer_behaviours: tuple[str, ...] = ("follow-gradient",)
        killer_seeders: tuple[str, ...] = ("gradient",)
        if self.killers_explore:
            killer_behaviours = ("follow-gradient", "explore")
            killer_seeders = ("gradient", "explore")
        killer = BrainSpec(
            behaviors=killer_behaviours, seeders=killer_seeders, params=params, arbiter=self.arbiter
        )
        return replace(config, brain=scout, brains_by_body={BodyType.KILLER: killer})
