from __future__ import annotations

import json

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from tests.support import (
    ENGINE_SPEED,
    TOKEN,
    FakeVehicleServer,
)
from vehicle_mcp.client import VehicleServerClient
from vehicle_mcp.config import Settings
from vehicle_mcp.tools import VehicleTools, is_position_name


def test_list_sessions_returns_index_metadata(tools: VehicleTools) -> None:
    result = tools.list_sessions()
    assert [session["session_id"] for session in result["sessions"]] == ["s1", "s2"]
    assert result["sessions"][0]["packet_count"] == 1200
    assert result["sessions"][0]["defs_version"] == "0.19.1"


def test_list_series_returns_names_without_samples(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    result = tools.list_series("s1")
    assert [entry["name"] for entry in result["series"]] == [
        "did.ENGINE_SPEED",
        "derived.LEAN_ANGLE",
        "did.HAND",
    ]
    for entry in result["series"]:
        assert set(entry) == {"name", "kind", "unit", "points_total"}
    assert fake_server.requests[0].url.params["max_points"] == "10"


def test_list_series_drops_position_looking_names(tools: VehicleTools) -> None:
    result = tools.list_series("s1")
    assert result["dropped_position_series"] == ["sensor.gps_latitude"]
    assert "D-076" in result["note"]
    assert "sensor.gps_latitude" not in [entry["name"] for entry in result["series"]]


def test_list_series_has_no_position_note_when_nothing_was_dropped(tools: VehicleTools) -> None:
    result = tools.list_series("clean")
    assert "dropped_position_series" not in result
    assert "note" not in result


def test_get_session_report_json_passthrough(tools: VehicleTools) -> None:
    result = tools.get_session_report("s1")
    assert isinstance(result, dict)
    assert result["session_id"] == "s1"
    assert result["summary"]["packet_count"] == 1200


def test_get_session_report_text(tools: VehicleTools, fake_server: FakeVehicleServer) -> None:
    result = tools.get_session_report("s1", format="text")
    assert result.startswith("# Session s1")
    assert fake_server.requests[0].url.params["format"] == "text"


def test_get_timeseries_slice_passes_window_and_caps_points(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    result = tools.get_timeseries_slice(
        "s1", ["did.ENGINE_SPEED"], start_ms=100, end_ms=300, max_points=5000
    )
    assert result["series"][0]["v"] == ENGINE_SPEED["v"]
    params = fake_server.requests[-1].url.params
    assert params["signals"] == "did.ENGINE_SPEED"
    assert params["max_points"] == "2000"
    assert params["from_ms"] == "100"
    assert params["to_ms"] == "300"


def test_get_timeseries_slice_keeps_smaller_cap(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    tools.get_timeseries_slice("s1", ["did.ENGINE_SPEED"], max_points=250)
    assert fake_server.requests[-1].url.params["max_points"] == "250"


def test_get_timeseries_slice_deduplicates_signals(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    tools.get_timeseries_slice("s1", ["did.ENGINE_SPEED", "did.ENGINE_SPEED"])
    assert fake_server.requests[-1].url.params["signals"] == "did.ENGINE_SPEED"


def test_get_timeseries_slice_multiple_signals(tools: VehicleTools) -> None:
    result = tools.get_timeseries_slice("s1", ["did.ENGINE_SPEED", "derived.LEAN_ANGLE"])
    assert [entry["name"] for entry in result["series"]] == [
        "did.ENGINE_SPEED",
        "derived.LEAN_ANGLE",
    ]


def test_get_timeseries_slice_requires_signals(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    with pytest.raises(ToolError, match="at least one"):
        tools.get_timeseries_slice("s1", [])
    assert fake_server.requests == []


def test_get_timeseries_slice_rejects_small_max_points(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    with pytest.raises(ToolError, match="at least 10"):
        tools.get_timeseries_slice("s1", ["did.ENGINE_SPEED"], max_points=5)
    assert fake_server.requests == []


def test_get_timeseries_slice_rejects_reversed_window(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    with pytest.raises(ToolError, match="start_ms"):
        tools.get_timeseries_slice("s1", ["did.ENGINE_SPEED"], start_ms=300, end_ms=100)
    assert fake_server.requests == []


def test_get_timeseries_slice_rejects_unknown_signal(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    with pytest.raises(ToolError, match="unknown series"):
        tools.get_timeseries_slice("s1", ["did.NOPE"])
    assert len(fake_server.requests) == 1


def test_get_timeseries_slice_rejects_position_signal(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    with pytest.raises(ToolError, match="D-076"):
        tools.get_timeseries_slice("s1", ["sensor.gps_latitude"])
    assert len(fake_server.requests) == 1


def test_compute_signal_stats_hand_computed(tools: VehicleTools) -> None:
    stats = tools.compute_signal_stats("s1", "did.HAND")
    assert stats["count"] == 8
    assert stats["min"] == 2
    assert stats["max"] == 9
    assert stats["mean"] == pytest.approx(5.0)
    assert stats["std"] == pytest.approx(2.0)
    assert stats["p05"] == pytest.approx(2.7)
    assert stats["p50"] == pytest.approx(4.5)
    assert stats["p95"] == pytest.approx(8.3)
    assert stats["duration_ms"] == 700
    assert stats["unit"] == "count"
    assert stats["approximate_reason"] is None


def test_compute_signal_stats_requests_server_max(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    tools.compute_signal_stats("s1", "did.ENGINE_SPEED", start_ms=100, end_ms=300)
    params = fake_server.requests[-1].url.params
    assert params["max_points"] == "20000"
    assert params["from_ms"] == "100"
    assert params["to_ms"] == "300"


def test_compute_signal_stats_downsampled_keeps_exact_extremes(tools: VehicleTools) -> None:
    stats = tools.compute_signal_stats("down", "did.DOWNSAMPLED")
    assert stats["count"] == 10_000
    assert stats["min"] == 1
    assert stats["max"] == 9
    assert stats["mean"] is None
    assert stats["std"] is None
    assert stats["p05"] is None
    assert stats["p50"] is None
    assert stats["p95"] is None
    assert "downsampled" in stats["approximate_reason"]


def test_compute_signal_stats_empty_series(tools: VehicleTools) -> None:
    stats = tools.compute_signal_stats("empty", "did.EMPTY")
    assert stats["count"] == 0
    assert stats["min"] is None
    assert stats["max"] is None
    assert stats["mean"] is None
    assert stats["duration_ms"] == 0


def test_compute_signal_stats_rejects_position(tools: VehicleTools) -> None:
    with pytest.raises(ToolError, match="D-076"):
        tools.compute_signal_stats("s1", "sensor.gps_latitude")


def test_compare_sessions_reports_b_minus_a_deltas(tools: VehicleTools) -> None:
    result = tools.compare_sessions("s1", "s2", "did.ENGINE_SPEED")
    assert result["session_a"]["session_id"] == "s1"
    assert result["session_b"]["session_id"] == "s2"
    assert result["delta"]["min"] == 500
    assert result["delta"]["max"] == 500
    assert result["delta"]["mean"] == pytest.approx(500)
    assert result["delta"]["p50"] == pytest.approx(500)
    assert result["delta"]["std"] == pytest.approx(0)
    assert result["delta"]["count"] == 0
    assert result["delta"]["duration_ms"] == 0


def test_compare_sessions_omits_approximate_fields(tools: VehicleTools) -> None:
    result = tools.compare_sessions("down", "down", "did.DOWNSAMPLED")
    assert result["delta"]["min"] == 0
    assert result["delta"]["max"] == 0
    assert "mean" not in result["delta"]
    assert result["session_a"]["mean"] is None


def test_compare_sessions_rejects_unit_mismatch(tools: VehicleTools) -> None:
    with pytest.raises(ToolError, match="units differ"):
        tools.compare_sessions("s1", "rad", "derived.LEAN_ANGLE")


def test_compare_sessions_rejects_unknown_signal_in_second_session(tools: VehicleTools) -> None:
    with pytest.raises(ToolError, match="unknown series"):
        tools.compare_sessions("s1", "s2", "did.HAND")


@pytest.mark.parametrize(
    ("session_id", "status"),
    [
        ("missing", "404"),
        ("bad-request", "422"),
        ("unavailable", "503"),
        ("unauthorized", "401"),
        ("forbidden", "403"),
    ],
)
def test_http_status_errors_are_explanatory(
    tools: VehicleTools, session_id: str, status: str
) -> None:
    with pytest.raises(ToolError) as excinfo:
        tools.list_series(session_id)
    message = str(excinfo.value)
    assert status in message
    assert TOKEN not in message
    assert "Traceback" not in message


def test_report_404_is_explanatory(tools: VehicleTools) -> None:
    with pytest.raises(ToolError, match="404"):
        tools.get_session_report("missing")


def test_connection_error_is_explanatory() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = VehicleServerClient(
        Settings(server_url="http://server.test", api_token=TOKEN),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ToolError) as excinfo:
            VehicleTools(client).list_sessions()
    finally:
        client.close()

    message = str(excinfo.value)
    assert "cannot reach vehicle-server" in message
    assert "http://server.test" in message
    assert TOKEN not in message


def test_every_request_is_an_authenticated_get(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    tools.list_sessions()
    tools.get_session_report("s1")
    tools.list_series("s1")
    tools.get_timeseries_slice("s1", ["did.ENGINE_SPEED"])
    tools.compute_signal_stats("s1", "did.ENGINE_SPEED")
    tools.compare_sessions("s1", "s2", "did.ENGINE_SPEED")

    assert fake_server.requests
    for request in fake_server.requests:
        assert request.method == "GET"
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        assert TOKEN not in str(request.url)


def test_tool_results_never_contain_the_token(tools: VehicleTools) -> None:
    results = [
        tools.list_sessions(),
        tools.get_session_report("s1"),
        tools.get_session_report("s1", format="text"),
        tools.list_series("s1"),
        tools.get_timeseries_slice("s1", ["did.ENGINE_SPEED"]),
        tools.compute_signal_stats("s1", "did.ENGINE_SPEED"),
        tools.compare_sessions("s1", "s2", "did.ENGINE_SPEED"),
    ]
    assert TOKEN not in json.dumps(results, default=str)


@pytest.mark.parametrize(
    "name",
    [
        "sensor.gps_latitude",
        "did.LAT",
        "derived.longitude",
        "vehicle.height",
        "gpsLatitude",
        "GPSLongitude",
        "did.ALTITUDE",
        "sensor.alt",
        "sensor.gps_height",
    ],
)
def test_is_position_name_matches_position_components(name: str) -> None:
    assert is_position_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "did.ENGINE_SPEED",
        "derived.LEAN_ANGLE",
        "sensor.related",
        "did.PLATE",
        "sensor.velocity",
        "did.ALTIMETER_OK",
        "derived.altitude_change",
        "derived.road_grade",
    ],
)
def test_is_position_name_ignores_other_names(name: str) -> None:
    assert not is_position_name(name)
