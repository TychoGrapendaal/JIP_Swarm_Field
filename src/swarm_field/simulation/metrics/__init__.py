"""Ground-truth measures of a run: they only read the world, and never change the run.

``MetricsCollector`` records the run's metrics series (B.5), sampled by ``MetricSample``, and
exports it. ``CoverageTracker`` and ``InfestationTracker`` each accumulate one measure, sampled
by their own event, and summarise the run so far.
"""

from swarm_field.simulation.metrics.collector import (
    MetricsCollector,
    MetricsConfig,
    write_metrics_csv,
    write_metrics_json,
)
from swarm_field.simulation.metrics.coverage import (
    CoverageSample,
    CoverageSummary,
    CoverageTracker,
)
from swarm_field.simulation.metrics.infestation import (
    ClearedPest,
    InfestationSample,
    InfestationSummary,
    InfestationTracker,
)

__all__ = [
    "ClearedPest",
    "CoverageSample",
    "CoverageSummary",
    "CoverageTracker",
    "InfestationSample",
    "InfestationSummary",
    "InfestationTracker",
    "MetricsCollector",
    "MetricsConfig",
    "write_metrics_csv",
    "write_metrics_json",
]
