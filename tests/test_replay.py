"""Tests for recording the commands of an interactive run and replaying them."""

from random import Random

from swarm_field.contracts import (
    CellId,
    CommandLog,
    FieldGeometry,
    InjectInfestation,
    TimedCommand,
)
from swarm_field.simulation import (
    CommandReplay,
    PestDynamicsConfig,
    SimulationContext,
    SimulationLoop,
    WorldState,
    WorldTick,
)

_PESTS = PestDynamicsConfig(
    period=1_000,
    growth_rate=0.05,
    spread_probability=0.3,
    spawn_probability=0.02,
    spawn_severity=0.1,
)


def _spreading_context(seed: int) -> SimulationContext:
    """A small field whose pests spread, so an injected pest changes every later random draw."""
    field = FieldGeometry(
        width=8,
        height=8,
        cell_size=1.0,
        row_sp=1.0,
        in_row_sp=1.0,
        headland=0.0,
        connectivity=4,
    )
    ctx = SimulationContext(now=0, rng=Random(seed), agents={}, world=WorldState(field))
    ctx.schedule(WorldTick(time=0, config=_PESTS))
    return ctx


def _infestation(ctx: SimulationContext) -> dict[CellId, float]:
    return {cell: state.infestation for cell, state in sorted(ctx.world.cells.items())}


def test_a_command_log_round_trips_through_json() -> None:
    log = CommandLog(
        seed=7,
        until=12_000,
        commands=(
            TimedCommand(time=2_500, command=InjectInfestation(cell=(3, 4), severity=0.8)),
            TimedCommand(time=9_000, command=InjectInfestation(cell=(1, 0), severity=0.25)),
        ),
    )

    assert CommandLog.from_json(log.to_json()) == log


def test_a_replay_reproduces_the_live_run_it_recorded() -> None:
    live = _spreading_context(seed=3)
    loop = SimulationLoop()
    recorded: list[TimedCommand] = []
    # The live frontend advances in uneven chunks and applies commands between them.
    for until, inject in ((1_500, None), (4_000, (5, 5)), (4_000, (6, 1)), (9_500, None)):
        loop.run(live, until=until)
        if inject is not None:
            command = InjectInfestation(cell=inject, severity=0.9)
            live.apply_command(command)
            recorded.append(TimedCommand(time=live.now, command=command))
    loop.run(live, until=12_000)

    replay = CommandReplay(
        _spreading_context(seed=3), CommandLog(seed=3, until=12_000, commands=tuple(recorded))
    )
    replay.advance_to(12_000)

    assert _infestation(replay.ctx) == _infestation(live)
    assert replay.ctx.world.cell((5, 5)).infestation > 0.0
