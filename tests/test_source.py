from __future__ import annotations

from typing import Any

import httpx
import pytest

from tests.support import TOKEN, FakeVehicleServer
from vehicle_mcp.client import VehicleServerClient, VehicleServerError
from vehicle_mcp.config import Settings
from vehicle_mcp.source import (
    HistorySource,
    NoRideSummary,
    RidesRouteUnavailable,
    SourceError,
)
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

    def ride_summary(self, session_id: str) -> Any:
        return {"session_id": session_id, "signals": []}

    def rides(
        self,
        *,
        start: str | None,
        end: str | None,
        signals: list[str] | None,
    ) -> Any:
        return {"columns": [], "rows": [], "skipped": 0}


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


def test_ride_summary_uses_the_summary_route(
    client: VehicleServerClient, fake_server: FakeVehicleServer
) -> None:
    payload = client.ride_summary("s1")
    assert payload["session_id"] == "s1"
    request = fake_server.requests[-1]
    assert request.method == "GET"
    assert request.url.path == "/sessions/s1/summary"
    assert not request.url.params


def test_rides_passes_the_same_bounds_and_signal_names(
    client: VehicleServerClient, fake_server: FakeVehicleServer
) -> None:
    payload = client.rides(
        start="2026-10-01T20:00:00",
        end="2026-10-02",
        signals=["ride_distance", "ride_time"],
    )
    assert payload["skipped"] == 1
    request = fake_server.requests[-1]
    assert request.method == "GET"
    assert request.url.path == "/rides"
    assert request.url.params["from"] == "2026-10-01T20:00:00"
    assert request.url.params["to"] == "2026-10-02"
    assert request.url.params["names"] == "ride_distance,ride_time"


def test_ride_summary_distinguishes_unknown_session_from_no_summary(
    client: VehicleServerClient,
) -> None:
    with pytest.raises(VehicleServerError, match="unknown session"):
        client.ride_summary("nope")
    with pytest.raises(NoRideSummary, match="no ride summary"):
        client.ride_summary("clean")


def test_rides_missing_route_is_detected() -> None:
    fake = FakeVehicleServer(supports_rides=False)
    client = VehicleServerClient(
        Settings(server_url="http://server.test", api_token=TOKEN),
        transport=httpx.MockTransport(fake.handler),
    )
    try:
        with pytest.raises(RidesRouteUnavailable, match="ride scalars route"):
            client.rides(start=None, end=None, signals=None)
    finally:
        client.close()


def test_ride_errors_never_contain_the_token() -> None:
    fake = FakeVehicleServer(rides_error=(503, TOKEN))
    client = VehicleServerClient(
        Settings(server_url="http://server.test", api_token=TOKEN),
        transport=httpx.MockTransport(fake.handler),
    )
    try:
        with pytest.raises(SourceError) as excinfo:
            client.rides(start=None, end=None, signals=None)
    finally:
        client.close()
    assert TOKEN not in str(excinfo.value)
    assert "***" in str(excinfo.value)


def test_no_summary_error_never_contains_the_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": TOKEN})

    client = VehicleServerClient(
        Settings(server_url="http://server.test", api_token=TOKEN),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(NoRideSummary) as excinfo:
            client.ride_summary("s1")
    finally:
        client.close()
    assert TOKEN not in str(excinfo.value)
    assert "***" in str(excinfo.value)
