"""Tests for filming runs of a pest batch into an experiment's folder."""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from swarm_field.contracts import TimeUnit
from swarm_field.experiments import (
    PestRunSpec,
    PestScenario,
    Run,
    measure_pest_response,
    record_video,
    record_videos,
    runs_for,
    video_path,
)

pytest.importorskip("matplotlib")


def _tiny_spec() -> PestRunSpec:
    """One hour of a tiny field, deciding every five minutes: a video of twelve frames."""
    scenario = PestScenario(
        size=6,
        obstacles=(),
        scouts=1,
        killers=1,
        robots_know_terrain=True,
        decision_interval=5 * TimeUnit.MINUTE,
    )
    return PestRunSpec(scenario, hours=1, pests_per_hour=20.0, warmup_hours=0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="an mp4 needs ffmpeg")
def test_videos_are_named_after_their_run_in_the_video_folder(tmp_path: Path) -> None:
    run = Run("tiny/hop", _tiny_spec(), seed=3)

    paths = record_videos(tmp_path, [run], processes=1, fps=50)

    assert paths == [tmp_path / "video" / "tiny_hop_seed3.mp4"]
    assert paths[0] == video_path(tmp_path, run)
    assert paths[0].stat().st_size > 0


def test_a_video_measures_what_the_headless_run_measures(tmp_path: Path) -> None:
    run = runs_for({"tiny": _tiny_spec()}, seeds=[1])[0]

    filmed = record_video(run, tmp_path / "frames.gif", fps=50)

    assert replace(filmed, wall=0.0) == replace(measure_pest_response(run), wall=0.0)
