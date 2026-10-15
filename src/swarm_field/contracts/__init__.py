"""Shared boundary types used by the swarm, simulation and visualization modules."""

from swarm_field.contracts.actions import Action, Charge, Idle, Move, Treat
from swarm_field.contracts.agent import (
    AgentInspection,
    Decision,
    Perception,
    RobotPublicState,
    RobotSelfState,
    SwarmAgent,
)
from swarm_field.contracts.ids import (
    AgentObjective,
    BodyType,
    CellId,
    CropType,
    Direction,
    PestType,
    RobotId,
    Terrain,
    Time,
    TimeUnit,
)
from swarm_field.contracts.messages import Message, MessageKind
from swarm_field.contracts.metrics import MetricsSample
from swarm_field.contracts.navigation import Vector2
from swarm_field.contracts.observations import (
    CropObservation,
    InfestationObservation,
    Observation,
    ObservationBase,
    PeerObservation,
    PeerTargetObservation,
    TerrainObservation,
)
from swarm_field.contracts.scenario import (
    EnergySpec,
    FailureMode,
    FailureSpec,
    PestDynamicsSpec,
    PositionErrorSpec,
    RobotSpec,
    ScenarioConfig,
    ScenarioId,
)
from swarm_field.contracts.snapshot import WorldSnapshot
from swarm_field.contracts.world import CellState, FieldGeometry

__all__ = [
    "Action",
    "AgentInspection",
    "AgentObjective",
    "BodyType",
    "CellId",
    "CellState",
    "Charge",
    "CropObservation",
    "CropType",
    "Decision",
    "Direction",
    "EnergySpec",
    "FailureMode",
    "FailureSpec",
    "FieldGeometry",
    "Idle",
    "InfestationObservation",
    "Message",
    "MessageKind",
    "MetricsSample",
    "Move",
    "Observation",
    "ObservationBase",
    "PeerObservation",
    "PeerTargetObservation",
    "Perception",
    "PestDynamicsSpec",
    "PestType",
    "PositionErrorSpec",
    "RobotId",
    "RobotPublicState",
    "RobotSelfState",
    "RobotSpec",
    "ScenarioConfig",
    "ScenarioId",
    "SwarmAgent",
    "Terrain",
    "TerrainObservation",
    "Time",
    "TimeUnit",
    "Treat",
    "Vector2",
    "WorldSnapshot",
]
