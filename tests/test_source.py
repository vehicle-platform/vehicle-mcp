from __future__ import annotations

from typing import Any

import pytest

from vehicle_mcp.source import HistorySource
from vehicle_mcp.stats import compute_stats
from vehicle_mcp.tools import VehicleTools

SERIES = {
    "name": "derived.sample",
    "kind": "derived",
    "unit": "u",
    "points_total": 4,
    "downsampled": False,
    "t_ms": [0, 100, 200, 300],
    "v": [1.0, 2.0, 3.0, 4.0],
}


class InMemorySource:
    """A non-HTTP HistorySource: the tools must not depend on vehicle-server."""

    def __init__(self, sessions: list[dict[str, Any]] | None = None) -> None:
        self._sessions = sessions if sessions is not None else [{"session_id": "mem-1"}]

    def sessions(self) -> Any:
        return self._sessions

    def report(self, session_id: str, *, text: bool = False) -> Any:
        return "synthetic" if text else {"session_id": session_id}

    def series(
        self,
        session_id: str,
        *,
        signals: list[str] | None,
        max_points: int,
        from_ms: int | None = None,
        to_ms: int | None = None,
    ) -> Any:
        return {"session_id": session_id, "series": [SERIES]}

    def stats(
        self,
        session_id: str,
        *,
        signals: list[str] | None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> Any:
        summary = compute_stats(SERIES)
        entry = {
            "name": SERIES["name"],
            "kind": SERIES["kind"],
            "unit": SERIES["unit"],
            "count": summary["count"],
            "mean": summary["mean"],
            "std": summary["std"],
            "min": summary["min"],
            "max": summary["max"],
            "p05": summary["p05"],
            "p50": summary["p50"],
            "p95": summary["p95"],
            "t_first_ms": SERIES["t_ms"][0],
            "t_last_ms": SERIES["t_ms"][-1],
        }
        return {"session_id": session_id, "series": [entry]}


def test_tools_run_on_any_history_source() -> None:
    source: HistorySource = InMemorySource()
    tools = VehicleTools(source)

    assert tools.list_sessions() == {"sessions": [{"session_id": "mem-1"}]}
    assert tools.get_session_report("mem-1", format="text") == "synthetic"
    assert [s["name"] for s in tools.list_series("mem-1")["series"]] == ["derived.sample"]
    stats = tools.compute_signal_stats("mem-1", "derived.sample")
    assert (stats["mean"], stats["min"], stats["max"], stats["duration_ms"]) == (2.5, 1.0, 4.0, 300)


def test_query_ride_history_runs_on_any_history_source() -> None:
    source: HistorySource = InMemorySource(
        [
            {
                "session_id": "a",
                "created_utc": "2026-10-01T10:00:00Z",
                "packet_count": 10,
                "loss_percent": 0.5,
                "status": "ok",
            },
            {
                "session_id": "b",
                "created_utc": "2026-10-02T10:00:00Z",
                "packet_count": 20,
                "loss_percent": 1.5,
                "status": "fail",
            },
        ]
    )
    result = VehicleTools(source).query_ride_history("2026-10-01", "2026-10-01")
    assert [session["session_id"] for session in result["sessions"]] == ["a"]
    assert result["count"] == 1
    assert result["total_packet_count"] == 10
    assert result["mean_loss_percent"] == pytest.approx(0.5)
