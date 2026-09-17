"""Tests for the minimal visualization adapter."""

from pytest import CaptureFixture

from swarm_field.contracts import FieldGeometry, WorldSnapshot
from swarm_field.viz import TextRenderer


def _snapshot() -> WorldSnapshot:
    return WorldSnapshot(
        time=5,
        field=FieldGeometry(
            width=3,
            height=2,
            cell_size=1.0,
            row_sp=1.0,
            in_row_sp=1.0,
            headland=1.0,
        ),
        cells={},
        robots=[],
    )


def test_text_renderer_summarizes_snapshot() -> None:
    assert TextRenderer.summarize(_snapshot()) == "t=5: 3x2 grid, 0 robot(s)"


def test_text_renderer_prints_summary(capsys: CaptureFixture[str]) -> None:
    TextRenderer().render(_snapshot())
    assert "t=5" in capsys.readouterr().out
