"""Run a batch of experiment runs in parallel, and keep everything they produce in ``output/``.

A batch is a list of ``Run``s: a configuration under a name, its *arm*, and a seed. ``runs_for``
builds one per arm and seed. ``Experiment`` makes a fresh folder ``output/<date>_<time>_<name>``,
runs the batch in parallel processes with any function that turns a ``Run`` into a result, and
writes every result to ``results.json`` there, with what was run in ``experiment.json``. The
folder also takes whatever else the experiment makes, e.g. a table or a video::

    experiment = Experiment("obstacle_comparison")
    runs = runs_for({"small/seeded": seeded, "small/hop": hop}, seeds=range(12))
    results = experiment.run(runs, measure, processes=22)
    rows = []
    for arm, arm_results in by_arm(results).items():
        rows.append([arm, mean_and_error([result.infested for result in arm_results])])
    experiment.write_text("tables.md", markdown_table(["arm", "infested"], rows))

A result is a dataclass, or anything else ``json`` can write. A configuration only has to
pickle, since it is sent to a worker process; it is not written yet, since the configurations
have no JSON form so far.
"""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

logger = logging.getLogger(__name__)

OUTPUT = Path("output")
"""Where every experiment gets its folder, relative to the working directory."""


@dataclass(frozen=True)
class Run[C]:
    """One run of a batch: a configuration and a seed.

    Attributes:
        arm: The configuration's name, e.g. ``river-small/hop-tuned``; the runs of one arm differ
            only in their seed.
        config: Whatever the run function needs, e.g. a ``PestRunSpec``. It must pickle.
        seed: The seed of this run.
        cost: A rough relative cost; the costliest runs start first, so that no process is left
            with a long run at the end. Optional.
    """

    arm: str
    config: C
    seed: int
    cost: float = 1.0

    @property
    def name(self) -> str:
        """The run's unique name, ``ARM/seedSEED``."""
        return f"{self.arm}/seed{self.seed}"


@dataclass(frozen=True)
class RunResult[C, R]:
    """What one run produced.

    Attributes:
        run: The run.
        result: What the run function returned.
        wall: Seconds the run function took.
    """

    run: Run[C]
    result: R
    wall: float


def runs_for[C](
    arms: Mapping[str, C], seeds: Iterable[int], cost: Callable[[C], float] | None = None
) -> list[Run[C]]:
    """One run per arm and seed, every arm with the same seeds, in the order of ``arms``.

    Args:
        arms: Each configuration by its arm name; names must be unique, which a ``dict`` makes
            them. ``arm_name`` builds a name from the parameters an arm differs in.
        seeds: The seeds every arm is run with.
        cost: The rough cost of a configuration, to start the costliest runs first. Optional.
    """
    every_seed = tuple(seeds)
    runs: list[Run[C]] = []
    for arm, config in arms.items():
        run_cost = 1.0
        if cost is not None:
            run_cost = cost(config)
        for seed in every_seed:
            runs.append(Run(arm, config, seed, run_cost))
    return runs


def arm_name(*parts: str, **params: object) -> str:
    """An arm name from its parts and its parameters, e.g. ``small/hop/killers=6``.

    Parameters are written as ``key=value``, numbers in their shortest form, so that every
    point of a parameter sweep gets its own name.
    """
    pieces = list(parts)
    for key, value in params.items():
        if isinstance(value, float):
            pieces.append(f"{key}={value:g}")
        else:
            pieces.append(f"{key}={value}")
    return "/".join(pieces)


class Experiment:
    """One experiment's folder in ``output/``, and the batch of runs that fills it.

    The folder is ``<root>/<YYYY-MM-DD_HH-MM-SS>_<name>``, so a listing sorts by time. It is made
    when the experiment is created, and holds ``experiment.json`` (when, what and from which
    commit), ``results.json`` (one record per run) and any file the caller adds.

    Attributes:
        name: The experiment's name, with every run of characters other than letters, digits,
            ``-`` and ``_`` turned into one underscore.
        directory: The experiment's folder.
    """

    def __init__(self, name: str, root: Path = OUTPUT) -> None:
        started = datetime.now()
        self.name: str = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
        self.directory: Path = root / f"{started:%Y-%m-%d_%H-%M-%S}_{self.name}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self._started: datetime = started

    def run[C, R](
        self,
        runs: Sequence[Run[C]],
        function: Callable[[Run[C]], R],
        processes: int | None = None,
    ) -> list[RunResult[C, R]]:
        """Run every run in parallel processes, write the results, and return them in order.

        Args:
            runs: The batch, e.g. from ``runs_for``.
            function: Turns one run into its result. It runs in a worker process, so it must be
                a module-level function.
            processes: How many processes, or ``None`` for one per CPU.

        Raises:
            ValueError: If two runs share a name.
        """
        _refuse_duplicate_names(runs)
        self.write_json("experiment.json", self._description(runs))
        logger.info("Running %d runs of %s in %s", len(runs), self.name, self.directory)
        results = run_in_parallel(runs, function, processes)
        self.write_json("results.json", [_record(result) for result in results])
        return results

    def path(self, filename: str) -> Path:
        """Where a file of this experiment goes, e.g. a video; its folders are made."""
        path = self.directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_text(self, filename: str, text: str) -> Path:
        """Write a text file, e.g. a Markdown table, into the experiment's folder."""
        path = self.path(filename)
        path.write_text(text, encoding="utf-8")
        return path

    def write_json(self, filename: str, data: object) -> Path:
        """Write ``data`` as JSON into the experiment's folder; dataclasses become objects."""
        return self.write_text(filename, json.dumps(_jsonable(data), indent=1) + "\n")

    def _description(self, runs: Sequence[Run[Any]]) -> dict[str, object]:
        """What ``experiment.json`` says: the name, when, the command, the commit and the runs."""
        arms: dict[str, list[int]] = {}
        for run in runs:
            arms.setdefault(run.arm, []).append(run.seed)
        return {
            "name": self.name,
            "started": self._started.isoformat(timespec="seconds"),
            "command": sys.argv,
            "commit": _git_commit(),
            "arms": arms,
        }


def run_in_parallel[C, R](
    runs: Sequence[Run[C]],
    function: Callable[[Run[C]], R],
    processes: int | None = None,
) -> list[RunResult[C, R]]:
    """Run every run in parallel processes, costliest first, and return the results in run order.

    ``Experiment.run`` runs a batch with it, and ``record_videos`` the runs it films.

    Args:
        runs: The runs.
        function: Turns one run into its result; a module-level function, or an object that
            pickles, since it runs in a worker process.
        processes: How many processes, or ``None`` for one per CPU.

    Raises:
        ValueError: If two runs share a name.
    """
    _refuse_duplicate_names(runs)
    order = sorted(range(len(runs)), key=lambda index: runs[index].cost, reverse=True)
    costliest_first = [runs[index] for index in order]
    with ProcessPoolExecutor(max_workers=processes) as pool:
        outcomes = list(pool.map(_Timed(function), costliest_first))
    by_index: dict[int, RunResult[C, R]] = {}
    for index, run, (result, wall) in zip(order, costliest_first, outcomes, strict=True):
        by_index[index] = RunResult(run, result, wall)
    results: list[RunResult[C, R]] = []
    for index in range(len(runs)):
        results.append(by_index[index])
    return results


def by_arm[C, R](results: Iterable[RunResult[C, R]]) -> dict[str, list[R]]:
    """The results grouped by arm, in the order the arms first appear, seeds in run order."""
    grouped: dict[str, list[R]] = {}
    for result in results:
        grouped.setdefault(result.run.arm, []).append(result.result)
    return grouped


def mean_and_error(values: Sequence[float | None]) -> str:
    """The mean of ``values`` and its standard error, as ``mean ± error``.

    ``-`` if there are no values or any is ``None``, e.g. a response time of a run that
    treated nothing: a mean over the other runs would hide it.
    """
    present: list[float] = []
    for value in values:
        if value is None:
            return "-"
        present.append(value)
    if not present:
        return "-"
    error = 0.0
    if len(present) > 1:
        error = stdev(present) / math.sqrt(len(present))
    return f"{mean(present):.2f} ± {error:.2f}"


def paired_difference(values: Sequence[float], baseline: Sequence[float]) -> str:
    """The mean of ``values`` minus ``baseline`` seed by seed, as ``+mean ± error``.

    Both must come from the same seeds in the same order, as ``by_arm`` gives them.
    """
    differences: list[float] = []
    for value, base in zip(values, baseline, strict=True):
        differences.append(value - base)
    error = 0.0
    if len(differences) > 1:
        error = stdev(differences) / math.sqrt(len(differences))
    return f"{mean(differences):+.2f} ± {error:.2f}"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """A Markdown table: the headers, then one line per row."""
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


class _Timed[C, R]:
    """The run function, timed; it is sent to the worker processes, so it must pickle."""

    def __init__(self, function: Callable[[Run[C]], R]) -> None:
        self._function: Callable[[Run[C]], R] = function

    def __call__(self, run: Run[C]) -> tuple[R, float]:
        """Call the run function on one run, in a worker, and time it."""
        started = time.perf_counter()
        result = self._function(run)
        return result, time.perf_counter() - started


def _refuse_duplicate_names(runs: Sequence[Run[Any]]) -> None:
    """Raise ``ValueError`` if two runs share a name, so no result or file overwrites another."""
    names = [run.name for run in runs]
    if len(set(names)) != len(names):
        raise ValueError("every run of a batch needs a unique arm and seed")


def _record(result: RunResult[Any, Any]) -> dict[str, object]:
    """One run's line of ``results.json``."""
    return {
        "name": result.run.name,
        "arm": result.run.arm,
        "seed": result.run.seed,
        "wall": result.wall,
        "result": result.result,
    }


def _jsonable(data: object) -> object:
    """``data`` with every dataclass turned into a dict, so that ``json`` can write it."""
    if is_dataclass(data) and not isinstance(data, type):
        return _jsonable(asdict(data))
    if isinstance(data, Mapping):
        converted: dict[str, object] = {}
        for key, value in data.items():
            converted[str(key)] = _jsonable(value)
        return converted
    if isinstance(data, (list, tuple)):
        return [_jsonable(item) for item in data]
    return data


def _git_commit() -> str | None:
    """The commit the code was run from, with ``+dirty`` for uncommitted changes, if known."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        logger.warning("Could not read the git commit for experiment.json: %s", error)
        return None
    if status.strip():
        return f"{commit}+dirty"
    return commit
