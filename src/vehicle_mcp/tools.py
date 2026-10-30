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

from .source import HistorySource, SourceError
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

    def query_ride_history(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Summarise the sessions whose created_utc is inside an inclusive date range.

        Each bound is an ISO 8601 date (the whole UTC day) or a timestamp (without an
        offset it is UTC). mean_loss_percent is the unweighted mean of the per-session
        loss_percent values. Sessions without a parseable created_utc are listed under
        `undated`, whatever the range, and are not summarised.
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
        return {
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "count": len(sessions),
            "total_packet_count": sum(count for count in packets if count is not None),
            "mean_loss_percent": statistics.fmean(known_losses) if known_losses else None,
            "status_histogram": dict(sorted(statuses.items())),
            "sessions": sessions,
            "undated": undated,
        }
