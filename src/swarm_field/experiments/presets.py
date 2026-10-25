"""Robot presets: the kinds of robot a swarm is built from, and how each one's brain is built.

A preset names a kind of robot, gives its body, and builds its brain. Four are ready:

- ``EXPLORER``: a scout that explores the field on the EXPLORE field and shares everything it
  knows, so explorers spread over the field and every robot that keeps a map learns from them;
- ``RELAY``: a scout without behaviours. It stands still and only passes on what the swarm's
  killers read;
- ``SEEDED_KILLER``: treats remembered pests on the TREAT field of its map. With claims it
  leaves a pest to a closer killer heading for it and does not join one another killer treats;
- ``HOP_KILLER``: treats on hop beacons (``TreatByBeacon`` on the hop field). It keeps no map
  and shares no memory.

Each is an instance of its preset class with the default parameters, so a variant is the class
with other ones, e.g. ``HopKillerPreset(beacons=...)``. ``CustomPreset`` takes any ``BrainSpec``.

What a robot broadcasts depends on who listens. Every preset says what the swarm has to relay
for it (``Relaying``): hop killers need every robot to relay their beacons, seeded killers need
the relays to pass on the pests. ``SwarmScenario`` merges what its presets need, and hands the
result to every preset that builds its brain (``SwarmContext``), with the time the swarm's
explorers need to sweep the field, from which explorers set their revisit age.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Protocol

from swarm_field.contracts import BodyType, Time
from swarm_field.runner import BrainSpec
from swarm_field.swarm import (
    ArbiterConfig,
    BeaconConfig,
    DockBeacons,
    DockSeeder,
    EscapeConfig,
    ExploreConfig,
    ExploreSeeder,
    HopDockField,
    HopField,
    KnowledgeSharing,
    Resupply,
    ReturnToDock,
    SharingConfig,
    TreatByBeacon,
    TreatInfestation,
    TreatSeeder,
)

REVISIT_SWEEPS = 0.5
"""An explorer's revisit age in sweep times of its swarm. From 0.25 to 0.8 the mean idleness
stays within 15 % of its best for 5 to 40 explorers, on 32 x 32 and 64 x 64 fields; see
``docs/explore.md``."""

_PEST_FACTS_ORDER = (
    KnowledgeSharing.NOTHING,
    KnowledgeSharing.INFESTATIONS,
    KnowledgeSharing.INFESTATIONS_AND_TARGETS,
)
"""The levels of pest facts a relay can pass on, from least to most."""


@dataclass(frozen=True)
class Relaying:
    """What the robots of a swarm relay, so that its killers hear of pests.

    Attributes:
        beacons: The hop beacons every robot keeps and relays, or ``None`` for no beacons.
        pest_facts: What relays pass on of the pests: nothing, the pest facts, or the pest facts
            and the killers' claims.
    """

    beacons: BeaconConfig | None = None
    pest_facts: KnowledgeSharing = KnowledgeSharing.NOTHING

    def merge(self, other: Relaying) -> Relaying:
        """What both need: the beacons of either, and the more of the two pest facts.

        Raises:
            ValueError: If both need beacons, but different ones: robots relay one kind.
        """
        beacons = self.beacons
        if other.beacons is not None:
            if beacons is not None and beacons != other.beacons:
                raise ValueError("two presets need different beacons; robots relay one kind")
            beacons = other.beacons
        pest_facts = self.pest_facts
        if _PEST_FACTS_ORDER.index(other.pest_facts) > _PEST_FACTS_ORDER.index(pest_facts):
            pest_facts = other.pest_facts
        return Relaying(beacons, pest_facts)


@dataclass(frozen=True)
class SwarmContext:
    """What a swarm tells each of its presets when the preset builds a robot's brain.

    Attributes:
        relaying: What the swarm's robots relay, merged from what all its presets need.
        arbiter: The overrides and hysteresis of the presets' ``UtilityArbiter``.
        sweep_time: How long the swarm's explorers need to visit every free cell once, if they
            split the field evenly; 0 without explorers. ``ExplorerPreset`` sets its revisit
            age from it.
    """

    relaying: Relaying
    arbiter: ArbiterConfig
    sweep_time: Time


class RobotPreset(Protocol):
    """A kind of robot: its name, its body, what the swarm relays for it, and its brain."""

    @property
    def name(self) -> str:
        """The preset's name in tables and arm names, e.g. ``explorer``."""
        ...

    @property
    def body(self) -> BodyType: ...

    def needs(self, hop_cells: float) -> Relaying:
        """What the swarm has to relay for this robot, with hops of ``hop_cells`` cells."""
        ...

    def brain(self, swarm: SwarmContext) -> BrainSpec:
        """The robot's brain in ``swarm``."""
        ...


@dataclass(frozen=True)
class ExplorerPreset:
    """An explorer: the explore behaviour on the EXPLORE field, sharing all it knows.

    In a swarm with beacons it relays them too, and is the source of the pests it sees. It
    returns to a dock of the field when its battery runs low (``ReturnToDock``). Its revisit
    age is ``revisit_sweeps`` times the swarm's sweep time, so a swarm of any size on any field
    keeps enough of the field stale to spread over.

    Attributes:
        explore: The parameters of its EXPLORE field.
        revisit_sweeps: Its revisit age in sweep times of its swarm (``SwarmContext``), or
            ``None`` to keep the revisit age of ``explore``.
        name: The preset's name.
    """

    explore: ExploreConfig = field(default_factory=ExploreConfig)
    revisit_sweeps: float | None = REVISIT_SWEEPS
    name: str = "explorer"

    @property
    def body(self) -> BodyType:
        return BodyType.SCOUT

    def needs(self, hop_cells: float) -> Relaying:
        return Relaying()

    def brain(self, swarm: SwarmContext) -> BrainSpec:
        return BrainSpec(
            behaviors=("explore", ReturnToDock.name),
            seeders=(ExploreSeeder.name, DockSeeder.name),
            params={"explore": self.explore_config(swarm.sweep_time)},
            arbiter=swarm.arbiter,
            beacons=swarm.relaying.beacons,
        )

    def explore_config(self, sweep_time: Time) -> ExploreConfig:
        """Its EXPLORE parameters in a swarm that sweeps its field in ``sweep_time``."""
        if self.revisit_sweeps is None:
            return self.explore
        return replace(self.explore, revisit_age=round(self.revisit_sweeps * sweep_time))


@dataclass(frozen=True)
class RelayPreset:
    """A relay: no behaviour, so it stands still and only broadcasts what the swarm relays.

    It keeps a map only to hold the pest facts it passes on.

    Attributes:
        name: The preset's name.
    """

    name: str = "relay"

    @property
    def body(self) -> BodyType:
        return BodyType.SCOUT

    def needs(self, hop_cells: float) -> Relaying:
        return Relaying()

    def brain(self, swarm: SwarmContext) -> BrainSpec:
        return BrainSpec(
            behaviors=(),
            seeders=(),
            arbiter=swarm.arbiter,
            sharing=SharingConfig(facts=swarm.relaying.pest_facts),
            beacons=swarm.relaying.beacons,
            keeps_map=swarm.relaying.pest_facts is not KnowledgeSharing.NOTHING,
        )


@dataclass(frozen=True)
class SeededKillerPreset:
    """A seeded killer: ``TreatInfestation`` on the TREAT field of its map, returning to a dock
    of the field when its battery runs low (``ReturnToDock``) or its doses do (``Resupply``).

    With ``claims`` it leaves a pest to a closer killer that claimed it and waits on a pest a
    lower-id killer already treats, and the swarm passes on the killers' claims with the pests.

    Attributes:
        claims: Whether it claims pests.
        window_blocks: How many blocks of the quadtree around its own its TREAT field is searched
            cell by cell; beyond them, leaf by leaf.
        name: The preset's name.
    """

    claims: bool = True
    window_blocks: int = 2
    name: str = "seeded-killer"

    @property
    def body(self) -> BodyType:
        return BodyType.KILLER

    def needs(self, hop_cells: float) -> Relaying:
        return Relaying(pest_facts=self._pest_facts())

    def brain(self, swarm: SwarmContext) -> BrainSpec:
        return BrainSpec(
            behaviors=(TreatInfestation.name, ReturnToDock.name, Resupply.name),
            seeders=(TreatSeeder.name, DockSeeder.name),
            params={"treat_window_blocks": self.window_blocks, "treat_yield": self.claims},
            arbiter=swarm.arbiter,
            sharing=SharingConfig(facts=self._pest_facts()),
            beacons=swarm.relaying.beacons,
        )

    def _pest_facts(self) -> KnowledgeSharing:
        """The pest facts it shares, and needs passed on: with the claims if it claims."""
        if self.claims:
            return KnowledgeSharing.INFESTATIONS_AND_TARGETS
        return KnowledgeSharing.INFESTATIONS


@dataclass(frozen=True)
class HopKillerPreset:
    """A hop killer: ``TreatByBeacon`` on the hop field, with no map and no shared memory.

    When its swarm relays DOCK beacons (``BeaconConfig.dock``), it returns to a dock down them
    (``HopDockField``) when its battery runs low (``ReturnToDock``) or its doses do
    (``Resupply``), still without a map.

    Attributes:
        beacons: How it and every robot of its swarm keep, relay and weigh beacons, or ``None``
            for ``hop-tuned`` of ``hop_beacon_variants``.
        name: The preset's name.
    """

    beacons: BeaconConfig | None = None
    name: str = "hop-killer"

    @property
    def body(self) -> BodyType:
        return BodyType.KILLER

    def needs(self, hop_cells: float) -> Relaying:
        beacons = self.beacons
        if beacons is None:
            beacons = hop_beacon_variants(hop_cells)["hop-tuned"]
        return Relaying(beacons=beacons)

    def brain(self, swarm: SwarmContext) -> BrainSpec:
        """Its brain, on the beacons the swarm relays, which are the ones it needs.

        Raises:
            ValueError: If the swarm relays no beacons, which ``SwarmScenario`` never does.
        """
        beacons = swarm.relaying.beacons
        if beacons is None:
            raise ValueError("a hop killer needs a swarm that relays beacons")
        behaviors = [TreatByBeacon.name]
        seeders = [HopField.name]
        if beacons.dock is not None:
            behaviors.extend([ReturnToDock.name, Resupply.name])
            seeders.append(HopDockField.name)
        return BrainSpec(
            behaviors=tuple(behaviors),
            seeders=tuple(seeders),
            arbiter=swarm.arbiter,
            sharing=SharingConfig(facts=KnowledgeSharing.NOTHING),
            beacons=beacons,
            keeps_map=False,
        )


@dataclass(frozen=True)
class CustomPreset:
    """A robot with a brain of its own, e.g. a new behaviour under test.

    The brain is used as it is, with its own arbiter. If it has beacons, every robot of the
    swarm relays them; ``pest_facts`` is what it needs the relays to pass on.

    Attributes:
        name: The preset's name.
        body: The robot's body.
        brain_spec: The robot's brain.
        pest_facts: What of the pests it needs the relays to pass on.
    """

    name: str
    body: BodyType
    brain_spec: BrainSpec
    pest_facts: KnowledgeSharing = KnowledgeSharing.NOTHING

    def needs(self, hop_cells: float) -> Relaying:
        return Relaying(beacons=self.brain_spec.beacons, pest_facts=self.pest_facts)

    def brain(self, swarm: SwarmContext) -> BrainSpec:
        return self.brain_spec


EXPLORER = ExplorerPreset()
RELAY = RelayPreset()
SEEDED_KILLER = SeededKillerPreset()
HOP_KILLER = HopKillerPreset()


def hop_beacon_variants(hop_cells: float) -> dict[str, BeaconConfig]:
    """The hop method's configurations, by name, for hops of ``hop_cells`` cells.

    ``hop`` counts metric hops of one radio range, capped where a beacon's pull falls below
    5 %, and relays DOCK beacons, so hop killers find a dock. ``hop-repel`` adds killers
    repelling each other, and ``hop-tuned`` pulls weighted by the square root of the distance
    and the escape: the best hop configuration so far.
    """
    hop = BeaconConfig(
        hop_cells=hop_cells, max_value=math.log(0.05) / math.log(0.5), dock=DockBeacons()
    )
    repel = replace(hop, killer_repulsion=2.0)
    tuned = replace(repel, distance_power=0.5, escape=EscapeConfig())
    return {"hop": hop, "hop-repel": repel, "hop-tuned": tuned}
