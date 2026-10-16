"""Body-specific assembly for the heterogeneous swarm.

This is the default constructor for the issue's SCOUT, KILLER and BATTERY body models. The
simulation can still construct custom ``ComposedAgent`` instances for targeted experiments.
"""

from swarm_field.contracts import BodyType, CellId, FieldGeometry, RobotId
from swarm_field.swarm.behavior import (
    Behavior,
    ExploreField,
    ReturnToDock,
    SupplyBattery,
    TreatInfestation,
)
from swarm_field.swarm.brain import ComposedAgent
from swarm_field.swarm.capabilities import CapabilitySet, capabilities_for
from swarm_field.swarm.explore import ExploreSeeder
from swarm_field.swarm.memory import AgentMemory
from swarm_field.swarm.navigator import Navigator
from swarm_field.swarm.provider import FieldProvider
from swarm_field.swarm.seeder import BatteryRequestSeeder, DockSeeder, TreatSeeder


def build_agent(
    robot_id: RobotId,
    body: BodyType,
    field: FieldGeometry,
    *,
    docks: tuple[CellId, ...] = (),
) -> ComposedAgent:
    """Build a brain whose navigation and behaviours match ``body``.

    SCOUTs explore unknown cells and fly over ground obstacles. KILLERs follow shared infestation
    knowledge and treat a cell once they reach it. BATTERY robots seek recent low-battery peer
    sightings and replace a battery when they share the peer's cell.
    """
    capabilities: CapabilitySet = capabilities_for(body)
    memory = AgentMemory(robot_id, field)

    dock_seeder = DockSeeder(docks)
    # Keep one shared protocol-typed list so branch-specific behaviour types do not infer an
    # invariant union such as list[ExploreField | ReturnToDock].
    behaviors: list[Behavior] = []
    if body is BodyType.SCOUT:
        seeders = [ExploreSeeder(), dock_seeder]
        behaviors.extend((ExploreField(), ReturnToDock()))
    elif body is BodyType.KILLER:
        seeders = [TreatSeeder(), dock_seeder]
        behaviors.extend((TreatInfestation(), ReturnToDock()))
    else:
        seeders = [BatteryRequestSeeder(), dock_seeder]
        behaviors.extend((SupplyBattery(), ReturnToDock()))

    fields = FieldProvider(memory, flies=capabilities.can_fly, seeders=seeders)
    return ComposedAgent(
        robot_id=robot_id,
        behaviors=behaviors,
        memory=memory,
        fields=fields,
        navigator=Navigator(fields.quadtree),
        capabilities=capabilities,
    )
