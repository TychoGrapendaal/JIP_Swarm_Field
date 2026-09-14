"""Tests for the minimal simulation wiring: the staged loop."""

from random import Random

from swarm_field.sim import SimContext, SimLoop


def _empty_context() -> SimContext:
    return SimContext(tick=0, rng=Random(0), agents={})


class _RecordingStage:
    name: str = "recording"

    def __init__(self) -> None:
        self.seen_ticks: list[int] = []

    def run(self, ctx: SimContext) -> None:
        self.seen_ticks.append(ctx.tick)


def test_loop_advances_clock_and_runs_stage_each_tick() -> None:
    stage = _RecordingStage()
    ctx = _empty_context()

    SimLoop([stage]).run(ctx, until=3)

    assert ctx.tick == 3
    assert stage.seen_ticks == [0, 1, 2]


def test_loop_runs_stages_in_order() -> None:
    order: list[str] = []

    class _Tagged:
        def __init__(self, tag: str) -> None:
            self.name: str = tag

        def run(self, ctx: SimContext) -> None:
            order.append(self.name)

    SimLoop([_Tagged("a"), _Tagged("b")]).step(_empty_context())

    assert order == ["a", "b"]
