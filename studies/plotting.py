"""Figures and LaTeX fragments for the studies, so the report can be rebuilt from a rerun.

Every figure is saved as a PDF in the experiment's folder, next to a ``.tex`` fragment that
includes it with its caption and label; a table is written as a ``.tex`` fragment as well. The
report then ``\\input``s the fragments, and running a study again replaces figures and tables
together. Methods keep their colours across all figures: shades of blue for hop-based killers,
of orange for seeded killers, and grey for the grid-search baseline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import mean, stdev
from typing import TYPE_CHECKING

from swarm_field.experiments import Experiment

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

METHOD_COLORS: dict[str, str] = {
    "hop": "#1f5fa8",
    "seeded": "#e07b28",
    "grid": "#7f7f7f",
}
"""The colour of every method, the same in every figure."""

METHOD_LABELS: dict[str, str] = {
    "hop": "hop-based",
    "seeded": "seeded",
    "grid": "grid search",
}
"""How every method is named in a legend."""

METHOD_MARKERS: dict[str, str] = {"hop": "s", "seeded": "o", "grid": "^"}


def mean_and_standard_error(values: Sequence[float]) -> tuple[float, float]:
    """The mean of ``values`` and its standard error over the seeds; 0 error for one value."""
    if len(values) < 2:
        return mean(values), 0.0
    return mean(values), stdev(values) / math.sqrt(len(values))


def plot_method(
    axes: Axes,
    method: str,
    xs: Sequence[float],
    values: Sequence[Sequence[float]],
    linestyle: str = "-",
    label: str | None = None,
    color: str | None = None,
) -> None:
    """Plot one method's mean over the seeds at every x, with its standard error as error bars.

    Args:
        axes: Where to draw.
        method: ``hop``, ``seeded`` or ``grid``, which sets the colour, marker and label.
        xs: The x of every point.
        values: Every point's values, one per seed; a point without values is left out.
        linestyle: E.g. ``"--"`` for a variant of the method.
        label: The legend entry, the method's name by default.
        color: A shade of the method's colour for a variant of it, the method's by default.
    """
    points_x: list[float] = []
    means: list[float] = []
    errors: list[float] = []
    for x, point in zip(xs, values, strict=True):
        if not point:
            continue
        average, error = mean_and_standard_error(point)
        points_x.append(x)
        means.append(average)
        errors.append(error)
    axes.errorbar(
        points_x,
        means,
        yerr=errors,
        color=color or METHOD_COLORS[method],
        marker=METHOD_MARKERS[method],
        linestyle=linestyle,
        capsize=3,
        label=label or METHOD_LABELS[method],
    )


def save_figure(
    experiment: Experiment, figure: Figure, stem: str, caption: str, label: str
) -> None:
    """Save ``figure`` as ``STEM.pdf`` with a ``STEM.tex`` fragment that includes it.

    The fragment is a LaTeX ``figure`` with the caption below the image and the label
    ``fig:LABEL``, so the report ``\\input``s it and refers to it with ``\\autoref``.
    """
    import matplotlib.pyplot as plt

    figure.tight_layout()
    figure.savefig(experiment.path(f"{stem}.pdf"))
    plt.close(figure)
    fragment = "\n".join(
        [
            r"\begin{figure}[h]",
            r"    \centering",
            rf"    \includegraphics[width=\linewidth]{{{stem}.pdf}}",
            rf"    \caption{{{latex_escape(caption)}}}",
            rf"    \label{{fig:{label}}}",
            r"\end{figure}",
            "",
        ]
    )
    experiment.write_text(f"{stem}.tex", fragment)


def save_table(
    experiment: Experiment,
    stem: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    caption: str,
    label: str,
) -> None:
    """Write a table as a ``STEM.tex`` fragment in the format of the report's tables.

    A ``booktabs`` table with the caption below it and the label ``tab:LABEL``.
    """
    columns = "l" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        r"    \small",
        rf"    \begin{{tabular}}{{{columns}}}",
        r"        \toprule",
        "        " + " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"        \midrule",
    ]
    for row in rows:
        lines.append("        " + " & ".join(latex_escape(cell) for cell in row) + r" \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            rf"    \caption{{{latex_escape(caption)}}}",
            rf"    \label{{tab:{label}}}",
            r"\end{table}",
            "",
        ]
    )
    experiment.write_text(f"{stem}.tex", "\n".join(lines))


def latex_escape(text: str) -> str:
    """``text`` with the characters LaTeX treats specially escaped, and ± as math."""
    escaped: list[str] = []
    for character in text:
        escaped.append(_LATEX_ESCAPES.get(character, character))
    return "".join(escaped)


_LATEX_ESCAPES: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "±": r"$\pm$",
}
"""What every special character becomes, replaced one character at a time."""
