"""Statistics over a series returned by vehicle-server.

`compute_signal_stats` prefers the exact `/stats` route. This module computes
statistics from `/series` when an older server lacks that route. vehicle-server
downsamples long windows with min/max bucketing; in that case the returned
extremes are still exact, but mean, standard deviation and percentiles would be
wrong, so they are reported as null with a reason instead of guessed.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

STATS_MAX_POINTS = 20_000
DOWNSAMPLED_REASON = (
    "vehicle-server downsampled this window with min/max bucketing; "
    "min and max are exact, but mean, std and percentiles need the full-resolution series"
)


def _finite_values(values: Sequence[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def compute_stats(series: Mapping[str, Any]) -> dict[str, Any]:
    """Compute stats for one series entry from GET /sessions/{id}/series."""
    values = _finite_values(series.get("v") or [])
    times = [float(t) for t in (series.get("t_ms") or []) if t is not None]
    downsampled = bool(series.get("downsampled", False))

    result: dict[str, Any] = {
        "count": series.get("points_total", len(values)),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": None,
        "std": None,
        "p05": None,
        "p50": None,
        "p95": None,
        "duration_ms": (max(times) - min(times)) if times else 0,
        "unit": series.get("unit"),
        "approximate_reason": None,
    }

    if downsampled:
        result["approximate_reason"] = DOWNSAMPLED_REASON
    elif values:
        ordered = sorted(values)
        result["mean"] = statistics.fmean(ordered)
        result["std"] = statistics.pstdev(ordered)
        result["p05"] = _percentile(ordered, 0.05)
        result["p50"] = _percentile(ordered, 0.50)
        result["p95"] = _percentile(ordered, 0.95)

    return result
