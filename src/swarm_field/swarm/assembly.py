"""Body-specific assembly for the heterogeneous swarm.

This is the default constructor for the issue's SCOUT, KILLER and BATTERY body models. The
simulation can still construct custom ``ComposedAgent`` instances for targeted experiments.
"""

from swarm_field.contracts import BodyType, CellId, FieldGeometry, RobotId
from swarm_field.swarm.behavior import (
    ExploreField,
    ReturnToDock,
    SupplyBattery,
    TreatInfestation,
)
from swarm_field.swarm.brain import ComposedAgent
from swarm_field.swarm.capabilities import CapabilitySet, capabilities_for
from swarm_field.swarm.memory import AgentMemory
from swarm_field.swarm.navigator import Navigator
from swarm_field.swarm.provider import FieldProvider
from swarm_field.swarm.seeder import (
    BatteryRequestSeeder,
    DockSeeder,
    ExploreSeeder,
    TreatSeeder,
)


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
    if body is BodyType.SCOUT:
        seeders = [ExploreSeeder(), dock_seeder]
        behaviors = [ExploreField(), ReturnToDock()]
    elif body is BodyType.KILLER:
        seeders = [TreatSeeder(), dock_seeder]
        behaviors = [TreatInfestation(), ReturnToDock()]
    else:
        seeders = [BatteryRequestSeeder(), dock_seeder]
        behaviors = [SupplyBattery(), ReturnToDock()]

    fields = FieldProvider(memory, flies=capabilities.can_fly, seeders=seeders)
    return ComposedAgent(
        robot_id=robot_id,
        behaviors=behaviors,
        memory=memory,
        fields=fields,
        navigator=Navigator(fields.quadtree),
        capabilities=capabilities,
    )
