"""Pure helpers for `explain_context` (D-131 item 2): topic tables, history ranking, direction.

Nothing here knows a signal scale or CAN id (invariant 2). The related-quantity table lists
names of ride-summary scalars and substrings of series names; a quantity is used only when
the server reports it for the ride, and every number is computed from server values.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

MAX_HISTORY_RIDES = 10
MAX_HISTORY_SCAN = 25  # candidate rides tried, so a long ride list cannot cause many calls
MIN_HISTORY_FOR_DIRECTION = 2  # previous rides needed before history decides up/down
MAX_CATALOG_ENTRIES = 3
HISTORY_MEASURES = ("min", "mean", "max")

# Priority when a signal name matches several topics (engine speed is in `idle` and `rpm`).
TOPIC_PRIORITY = (
    "coolant",
    "battery",
    "rpm",
    "speed",
    "throttle",
    "fuel",
    "gears",
    "stops",
    "idle",
    "ride",
)

# topic -> (ride-summary scalar names, series-name substrings whose mean and max are added)
_RIDE_BASICS = ("ride_time", "ride_distance", "ride_idle_time", "ride_stop_count")
_STYLE = ("ride_shift_count", "ride_launch_count", "ride_aggressiveness_score")
RELATED: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "coolant": (
        (
            "ride_stop_count",
            "ride_idle_time",
            "ride_time",
            "ride_distance",
            "ride_warmup_time",
            "ride_start_coolant_temp",
            "ride_end_coolant_temp",
            "ride_start_intake_temp",
            "ride_heat_soak_max",
            "ride_coolant_rate_max",
            "ride_fan_on_temperature",
            "ride_thermal_cycle_count",
        ),
        ("vehicle_speed", "intake_air_temp", "ambient_air_temp"),
    ),
    "battery": (
        (
            "ride_idle_time",
            "ride_time",
            "ride_stop_count",
            "ride_cranking_dip",
            "ride_rest_voltage",
        ),
        ("engine_speed",),
    ),
    "idle": (
        (
            "ride_idle_time",
            "ride_stop_count",
            "ride_time",
            "ride_warmup_time",
            "ride_start_coolant_temp",
            "ride_idle_rpm_std",
            "ride_idle_rpm_drift",
        ),
        ("engine_speed", "coolant_temp"),
    ),
    "rpm": (
        (*_STYLE, "ride_over_rev_time", "ride_idle_time", "ride_time"),
        ("vehicle_speed",),
    ),
    "speed": (
        (*_STYLE, "ride_stop_count", "ride_distance", "ride_time"),
        ("engine_speed",),
    ),
    "throttle": (
        (*_STYLE, "ride_time"),
        ("vehicle_speed", "engine_speed"),
    ),
    "fuel": (
        (
            "ride_distance",
            "ride_idle_time",
            "ride_stop_count",
            "ride_aggressiveness_score",
            "ride_time",
        ),
        ("vehicle_speed",),
    ),
    "gears": ((*_STYLE, "ride_time"), ("engine_speed", "vehicle_speed")),
    "stops": (_RIDE_BASICS, ("vehicle_speed",)),
    "ride": (_RIDE_BASICS, ("vehicle_speed",)),
}


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def compare(this: float, previous: Sequence[float]) -> dict[str, Any]:
    """Rank of this ride among all (1 = highest) and where it sits against the previous range."""
    low, high = min(previous), max(previous)
    if this > high:
        position = "above"
    elif this < low:
        position = "below"
    else:
        position = "within"
    return {
        "this": this,
        "rank": 1 + sum(1 for value in previous if value > this),
        "of": len(previous) + 1,
        "previous_min": low,
        "previous_median": median(previous),
        "previous_max": high,
        "outside_previous_range": position,
    }


def directions_shown(trend: str, comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Which catalog directions (up / down / unstable) the data shows, each with its basis.

    With at least two previous rides, history decides up and down: `up` when no previous
    ride's maximum is higher and this ride's maximum is above the previous median (the top
    of the usual range, ties included); `down` likewise for the minimum. Without enough
    history the ride's own trend decides (a coolant trend alone always rises while warming
    up, which is why history wins). An unstable trend always counts.
    """
    shown: dict[str, str] = {}
    if trend == "unstable":
        shown["unstable"] = "ride trend"
    has_history = bool(comparisons) and all(
        c["of"] - 1 >= MIN_HISTORY_FOR_DIRECTION for c in comparisons.values()
    )
    if has_history:
        high, low = comparisons["max"], comparisons["min"]
        if high["rank"] == 1 and high["this"] > high["previous_median"]:
            shown["up"] = "this ride's maximum is at the top of the previous rides' range"
        if low["this"] < low["previous_median"] and low["this"] <= low["previous_min"]:
            shown["down"] = "this ride's minimum is at the bottom of the previous rides' range"
    elif trend == "rising":
        shown["up"] = "ride trend (too few previous rides)"
    elif trend == "falling":
        shown["down"] = "ride trend (too few previous rides)"
    return shown


def entry_matches(entry: Mapping[str, Any], signal: str, shown: Mapping[str, str]) -> bool:
    """A catalog entry has a symptom on this signal whose direction the data shows."""
    symptoms = entry.get("symptoms")
    if not isinstance(symptoms, list):
        return False
    return any(
        isinstance(s, dict) and s.get("signal") == signal and s.get("direction") in shown
        for s in symptoms
    )
