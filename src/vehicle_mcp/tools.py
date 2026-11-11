"""The read-only MCP tools.

Series names, kinds, units and counts always come from vehicle-server; this
module never hard-codes a DID, signal name, unit or scale. Every method issues
GET requests only and turns expected failures into ToolError messages.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime, time
from typing import Any, Literal

from mcp.server.mcpserver.exceptions import ToolError

from .source import (
    FaultNotAvailable,
    HistorySource,
    SourceError,
    StatsRouteUnavailable,
)
from .stats import STATS_MAX_POINTS, compute_stats

MIN_SERIES_POINTS = 10
MAX_SLICE_POINTS = 2_000
DEFAULT_SLICE_POINTS = 500

# A horizontal component anywhere in the name; a vertical one only as the last token,
# so a relative quantity such as derived.altitude_change (barometric, since key-on) stays.
_HORIZONTAL_TOKENS = frozenset({"lat", "lon", "lng", "latitude", "longitude"})
_VERTICAL_TOKENS = frozenset({"height", "alt", "altitude"})
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def is_position_name(name: str) -> bool:
    """True when a series name looks like a GPS position component."""
    normalized = _CAMEL_BOUNDARY.sub("_", name).lower()
    tokens = [token for token in _TOKEN_SPLIT.split(normalized) if token]
    if any(token in _HORIZONTAL_TOKENS for token in tokens):
        return True
    return bool(tokens) and tokens[-1] in _VERTICAL_TOKENS


def _position_error(signal: str) -> ToolError:
    return ToolError(
        f"series {signal!r} looks like a GPS position; vehicle-server does not expose "
        "position yet (D-076 item 3 is not implemented), so vehicle-mcp has no position tool"
    )


_RIDE_SUMMARY_WARNING = "stored bin edges differ from the current defs; values are as stored"
_POSITION_NOTE = (
    "position-looking names are hidden: vehicle-server does not expose GPS "
    "position yet (D-076 item 3 is not implemented)"
)


def _summary_position_names(entry: dict[str, Any]) -> list[str]:
    """Names in one summary entry that look like a GPS position component."""
    names: list[str] = []
    name = entry.get("name")
    if isinstance(name, str) and is_position_name(name):
        names.append(name)
    axes = entry.get("axes")
    if isinstance(axes, list):
        for axis in axes:
            input_name = axis.get("input") if isinstance(axis, dict) else None
            if isinstance(input_name, str) and is_position_name(input_name):
                names.append(input_name)
    return names


def _without_position_keys(mapping: Any) -> tuple[Any, list[str]]:
    """A /rides `values` or `flags` mapping without position-looking keys, and those keys."""
    if not isinstance(mapping, dict):
        return mapping, []
    dropped = [name for name in mapping if isinstance(name, str) and is_position_name(name)]
    kept = {name: value for name, value in mapping.items() if name not in dropped}
    return kept, dropped


def _as_utc(parsed: datetime) -> datetime:
    """A timestamp without an offset is taken as UTC."""
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_bound(value: str, *, end: bool) -> datetime:
    """Parse an ISO 8601 date or timestamp; a date-only bound covers the whole UTC day."""
    text = value.strip()
    try:
        day = date.fromisoformat(text)
    except ValueError:
        pass
    else:
        return datetime.combine(day, time.max if end else time.min, tzinfo=UTC)

    try:
        return _as_utc(datetime.fromisoformat(text))
    except ValueError:
        name = "end_date" if end else "start_date"
        raise ToolError(f"{name} must be an ISO 8601 date or timestamp") from None


def _parse_created(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return _as_utc(datetime.fromisoformat(value.strip()))
    except ValueError:
        return None


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


_FAULT_NOTE = (
    "Catalog lookup only, not a diagnosis. If advise_stop is true, tell the rider to stop "
    "safely and have the motorcycle inspected; never say that riding on is safe (D-084)."
)


def _symptom_summary(symptom: Any) -> str:
    if not isinstance(symptom, dict):
        return str(symptom)
    text = f"{symptom.get('signal')} {symptom.get('direction')}"
    if symptom.get("condition"):
        text += f" ({symptom['condition']})"
    return text


def _compact_fault(entry: dict[str, Any]) -> dict[str, Any]:
    rating = entry.get("rating")
    symptoms = entry.get("symptoms")
    return {
        "id": entry.get("id"),
        "subsystem": entry.get("subsystem"),
        "root_cause": entry.get("root_cause"),
        "severity": rating.get("severity") if isinstance(rating, dict) else None,
        "advise_stop": entry.get("advise_stop"),
        "detectable_now": entry.get("detectable_now"),
        "symptoms": (
            "; ".join(_symptom_summary(s) for s in symptoms) if isinstance(symptoms, list) else ""
        ),
    }


class VehicleTools:
    """Tool implementations bound to one HistorySource."""

    def __init__(self, source: HistorySource) -> None:
        self._source = source

    def _call(self, fetch: Callable[[], Any]) -> Any:
        try:
            return fetch()
        except SourceError as exc:
            raise ToolError(str(exc)) from None

    def _series_index(
        self, session_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        payload = self._call(
            lambda: self._source.series(session_id, signals=None, max_points=MIN_SERIES_POINTS)
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("series"), list):
            raise ToolError(
                "vehicle-server returned an unexpected series response "
                "(expected an object with a 'series' list)"
            )
        kept: list[dict[str, Any]] = []
        dropped: list[str] = []
        for entry in payload["series"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ToolError("vehicle-server returned a series entry without a name")
            if is_position_name(entry["name"]):
                dropped.append(entry["name"])
            else:
                kept.append(entry)
        return payload, kept, dropped

    def _check_window(self, start_ms: int | None, end_ms: int | None) -> None:
        if start_ms is not None and end_ms is not None and start_ms > end_ms:
            raise ToolError("start_ms must be less than or equal to end_ms")

    def _ensure_known(self, session_id: str, signals: list[str]) -> None:
        _, kept, _ = self._series_index(session_id)
        available = {entry["name"] for entry in kept}
        for signal in signals:
            if is_position_name(signal):
                raise _position_error(signal)
        unknown = [signal for signal in signals if signal not in available]
        if unknown:
            names = ", ".join(repr(name) for name in unknown)
            raise ToolError(
                f"unknown series in session {session_id!r}: {names}; "
                "call list_series to see the available names"
            )

    def _one_series(self, payload: Any, signal: str) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("series"), list):
            raise ToolError(
                "vehicle-server returned an unexpected series response "
                "(expected an object with a 'series' list)"
            )
        for entry in payload["series"]:
            if isinstance(entry, dict) and entry.get("name") == signal:
                return entry
        raise ToolError(f"vehicle-server returned no series for {signal!r}")

    def list_sessions(self) -> dict[str, Any]:
        """List recorded sessions with their index metadata."""
        payload = self._call(self._source.sessions)
        if not isinstance(payload, list):
            raise ToolError("vehicle-server returned an unexpected sessions response")
        return {"sessions": payload}

    def get_session_report(
        self, session_id: str, format: Literal["json", "text"] = "json"
    ) -> dict[str, Any] | str:
        """Read one session report as structured JSON or as pre-rendered text."""
        return self._call(lambda: self._source.report(session_id, text=format == "text"))

    def get_ride_summary(self, session_id: str) -> dict[str, Any]:
        """Read the stored class-B ride summary of one session.

        One session is one ride. The scalars and binned signals are computed once by
        vehicle-server from vehicle-defs definitions and returned here exactly as stored:
        this tool never recomputes, rebins, converts or aggregates a value. A null value
        carries the server's status/flag reason in the same entry. Binned signals keep
        their server-provided axes and cells; `defs_mismatch` is passed through and adds
        a warning when the stored bin edges differ from the current defs.
        """
        payload = self._call(lambda: self._source.ride_summary(session_id))
        if not isinstance(payload, dict) or not isinstance(payload.get("signals"), list):
            raise ToolError(
                "vehicle-server returned an unexpected ride summary response "
                "(expected an object with a 'signals' list)"
            )

        kept: list[dict[str, Any]] = []
        dropped: list[str] = []
        for entry in payload["signals"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ToolError("vehicle-server returned a ride summary signal without a name")
            position_names = _summary_position_names(entry)
            if position_names:
                dropped.extend(position_names)
            else:
                kept.append(entry)

        result = dict(payload)
        result["signals"] = kept
        if dropped:
            result["dropped_position_names"] = dropped
            result["note"] = _POSITION_NOTE

        defs_mismatch = payload.get("defs_mismatch") is True or any(
            entry.get("defs_mismatch") is True for entry in kept
        )
        warnings = list(payload["warnings"]) if isinstance(payload.get("warnings"), list) else []
        if defs_mismatch and _RIDE_SUMMARY_WARNING not in warnings:
            warnings.append(_RIDE_SUMMARY_WARNING)
        if warnings:
            result["warnings"] = warnings
        return result

    def list_fault_catalog(self, subsystem: str | None = None) -> dict[str, Any]:
        """List the sourced fault catalog (compact), optionally for one subsystem.

        This only looks up the catalog that vehicle-defs ships; it never diagnoses a
        motorcycle and never says that a fault is present. Only displayable (confirmed)
        entries are listed. Each entry has id, subsystem, root_cause, severity,
        advise_stop, detectable_now and a one-line symptom summary; call get_fault for
        the full entry. If advise_stop is true, any answer to the rider must tell them
        to stop safely and have the motorcycle inspected, and must never say that
        riding on is safe. If the catalog is not available the result has
        available=false and a reason; relay the reason and do not guess.
        """
        try:
            payload = self._source.fault_catalog(subsystem=subsystem)
        except FaultNotAvailable as exc:
            return {"available": False, "reason": str(exc), "note": _FAULT_NOTE}
        except SourceError as exc:
            raise ToolError(str(exc)) from None
        if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
            raise ToolError(
                "vehicle-server returned an unexpected fault catalog response "
                "(expected an object with an 'entries' list)"
            )
        entries = [_compact_fault(e) for e in payload["entries"] if isinstance(e, dict)]
        return {
            "available": True,
            "defs_version": payload.get("defs_version"),
            "vehicle": payload.get("vehicle"),
            "subsystem": subsystem,
            "count": len(entries),
            "entries": entries,
            "note": _FAULT_NOTE,
        }

    def get_fault(self, fault_id: str) -> dict[str, Any]:
        """Read one displayable fault catalog entry in full: root cause, symptoms,
        rating, rider_text, action and sources.

        This only looks up the sourced catalog; it never diagnoses and never says the
        fault is present on this motorcycle. If advise_stop is true, the answer must tell
        the rider to stop safely and have the motorcycle inspected, never that riding on
        is safe. If the entry is not available (unknown id, unconfirmed entry or an older
        server) the result has available=false and a reason; relay it, do not guess.
        """
        try:
            payload = self._source.fault(fault_id)
        except FaultNotAvailable as exc:
            return {"available": False, "reason": str(exc), "note": _FAULT_NOTE}
        except SourceError as exc:
            raise ToolError(str(exc)) from None
        if not isinstance(payload, dict) or not isinstance(payload.get("entry"), dict):
            raise ToolError(
                "vehicle-server returned an unexpected fault response "
                "(expected an object with an 'entry' object)"
            )
        return {
            "available": True,
            "defs_version": payload.get("defs_version"),
            "vehicle": payload.get("vehicle"),
            "entry": payload["entry"],
            "note": _FAULT_NOTE,
        }

    def list_series(self, session_id: str) -> dict[str, Any]:
        """List a session's series names, kinds, units and point counts (no samples)."""
        payload, kept, dropped = self._series_index(session_id)
        result: dict[str, Any] = {
            "session_id": payload.get("session_id", session_id),
            "series": [
                {
                    "name": entry["name"],
                    "kind": entry.get("kind"),
                    "unit": entry.get("unit"),
                    "points_total": entry.get("points_total"),
                }
                for entry in kept
            ],
        }
        if dropped:
            result["dropped_position_series"] = dropped
            result["note"] = (
                "position-looking series are hidden: vehicle-server does not expose GPS "
                "position yet (D-076 item 3 is not implemented)"
            )
        return result

    def get_timeseries_slice(
        self,
        session_id: str,
        signals: list[str],
        start_ms: int | None = None,
        end_ms: int | None = None,
        max_points: int = DEFAULT_SLICE_POINTS,
    ) -> dict[str, Any]:
        """Read time-series samples for a window, capped at 2000 points per series."""
        if not signals:
            raise ToolError("signals must contain at least one series name")
        if max_points < MIN_SERIES_POINTS:
            raise ToolError(f"max_points must be at least {MIN_SERIES_POINTS}")
        self._check_window(start_ms, end_ms)

        unique = list(dict.fromkeys(signals))
        self._ensure_known(session_id, unique)

        payload = self._call(
            lambda: self._source.series(
                session_id,
                signals=unique,
                max_points=min(max_points, MAX_SLICE_POINTS),
                from_ms=start_ms,
                to_ms=end_ms,
            )
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("series"), list):
            raise ToolError(
                "vehicle-server returned an unexpected series response "
                "(expected an object with a 'series' list)"
            )
        return payload

    def _series_stats(
        self,
        session_id: str,
        signal: str,
        start_ms: int | None,
        end_ms: int | None,
    ) -> dict[str, Any]:
        """The pre-SV-2 path: compute statistics from the (possibly downsampled) series."""
        payload = self._call(
            lambda: self._source.series(
                session_id,
                signals=[signal],
                max_points=STATS_MAX_POINTS,
                from_ms=start_ms,
                to_ms=end_ms,
            )
        )
        return compute_stats(self._one_series(payload, signal))

    def _stats_from_server(self, payload: Any, signal: str) -> dict[str, Any]:
        entry = self._one_series(payload, signal)
        first = entry.get("t_first_ms")
        last = entry.get("t_last_ms")
        duration = (
            last - first
            if isinstance(first, (int, float)) and isinstance(last, (int, float))
            else 0
        )
        return {
            "count": entry.get("count", 0),
            "min": entry.get("min"),
            "max": entry.get("max"),
            "mean": entry.get("mean"),
            "std": entry.get("std"),
            "p05": entry.get("p05"),
            "p50": entry.get("p50"),
            "p95": entry.get("p95"),
            "duration_ms": duration,
            "unit": entry.get("unit"),
            "approximate_reason": None,
        }

    def compute_signal_stats(
        self,
        session_id: str,
        signal: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> dict[str, Any]:
        """Compute count, min, max, mean, std, p05, p50, p95, duration_ms and unit."""
        self._check_window(start_ms, end_ms)
        self._ensure_known(session_id, [signal])

        try:
            payload = self._source.stats(
                session_id, signals=[signal], start_ms=start_ms, end_ms=end_ms
            )
        except StatsRouteUnavailable:
            return self._series_stats(session_id, signal, start_ms, end_ms)
        except SourceError as exc:
            raise ToolError(str(exc)) from None
        return self._stats_from_server(payload, signal)

    def compare_sessions(self, session_a: str, session_b: str, signal: str) -> dict[str, Any]:
        """Compare one signal across two sessions and report the b - a deltas.

        Each side covers its whole session; the two rides are not aligned in time.
        """
        stats_a = self.compute_signal_stats(session_a, signal)
        stats_b = self.compute_signal_stats(session_b, signal)

        if stats_a["unit"] != stats_b["unit"]:
            raise ToolError(
                f"cannot compare {signal!r}: units differ "
                f"({stats_a['unit']!r} vs {stats_b['unit']!r})"
            )

        delta: dict[str, float | int] = {}
        for key in ("count", "min", "max", "mean", "std", "p05", "p50", "p95", "duration_ms"):
            value_a = stats_a[key]
            value_b = stats_b[key]
            if value_a is not None and value_b is not None:
                delta[key] = value_b - value_a

        return {
            "signal": signal,
            "session_a": {"session_id": session_a, **stats_a},
            "session_b": {"session_id": session_b, **stats_b},
            "delta": delta,
        }

    def _with_ride_scalars(
        self,
        result: dict[str, Any],
        sessions: list[dict[str, Any]],
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Join the server's /rides scalars onto the listed dated sessions.

        Any source error (a missing route included) keeps the session part, which is
        still valid, and states why the ride scalars are unavailable.
        """
        try:
            payload = self._source.rides(start=start_date, end=end_date, signals=None)
        except SourceError as exc:  # RidesRouteUnavailable is one of them
            result["ride_scalars"] = "unavailable"
            result["ride_scalars_reason"] = str(exc)
            return result

        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("rows"), list)
            or not isinstance(payload.get("columns"), list)
        ):
            result["ride_scalars"] = "unavailable"
            result["ride_scalars_reason"] = (
                "vehicle-server returned an unexpected ride-scalars response"
            )
            return result

        rows_by_session = {
            row["session_id"]: row
            for row in payload["rows"]
            if isinstance(row, dict) and isinstance(row.get("session_id"), str)
        }
        columns: list[Any] = []
        dropped: list[str] = []
        for column in payload["columns"]:
            name = column.get("name") if isinstance(column, dict) else None
            if isinstance(name, str) and is_position_name(name):
                dropped.append(name)
            else:
                columns.append(column)

        joined: list[dict[str, Any]] = []
        for session in sessions:
            item = dict(session)
            row = rows_by_session.get(session.get("session_id"))
            if row is None:
                item["ride"] = None
                item["ride_reason"] = "vehicle-server has no stored ride summary for this session"
            else:
                values, dropped_values = _without_position_keys(row.get("values"))
                flags, dropped_flags = _without_position_keys(row.get("flags"))
                dropped.extend(dropped_values + dropped_flags)
                item["ride"] = {"values": values, "flags": flags, "columns": columns}
            joined.append(item)

        result["sessions"] = joined
        result["skipped"] = payload.get("skipped", 0)
        if result["undated"]:
            result["undated_ride_reason"] = (
                "undated sessions have no ride scalars: /rides selects sessions by created_utc"
            )
        if dropped:
            result["dropped_position_names"] = list(dict.fromkeys(dropped))
            result["note"] = _POSITION_NOTE
        return result

    def query_ride_history(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Summarise the sessions whose created_utc is inside an inclusive date range.

        Each bound is an ISO 8601 date (the whole UTC day) or a timestamp (without an
        offset it is UTC). mean_loss_percent is the unweighted mean of the per-session
        loss_percent values. Sessions without a parseable created_utc are listed under
        `undated`, whatever the range, and are not summarised.

        The server's stored class-B ride scalars are joined onto every dated session as
        `ride` (values, flags and column units); a session the server skipped has
        `ride: null` and a reason; undated sessions get none (`undated_ride_reason`). If
        /rides is missing or fails, the output keeps the session data and states that
        ride scalars are unavailable.
        """
        start = _parse_bound(start_date, end=False)
        end = _parse_bound(end_date, end=True)
        if start > end:
            raise ToolError("start_date must be less than or equal to end_date")

        payload = self._call(self._source.sessions)
        if not isinstance(payload, list):
            raise ToolError("vehicle-server returned an unexpected sessions response")

        dated: list[tuple[datetime, dict[str, Any]]] = []
        undated: list[Any] = []
        for session in payload:
            created = (
                _parse_created(session.get("created_utc")) if isinstance(session, dict) else None
            )
            if created is None:
                undated.append(session)
            elif start <= created <= end:
                dated.append((created, session))
        dated.sort(key=lambda item: item[0])
        sessions = [session for _, session in dated]

        packets = [_number(session.get("packet_count")) for session in sessions]
        losses = [_number(session.get("loss_percent")) for session in sessions]
        known_losses = [loss for loss in losses if loss is not None]
        statuses = Counter(
            status if isinstance(status := session.get("status"), str) and status else "unknown"
            for session in sessions
        )
        result = {
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "count": len(sessions),
            "total_packet_count": sum(count for count in packets if count is not None),
            "mean_loss_percent": statistics.fmean(known_losses) if known_losses else None,
            "status_histogram": dict(sorted(statuses.items())),
            "sessions": sessions,
            "undated": undated,
        }
        return self._with_ride_scalars(result, sessions, start_date, end_date)
