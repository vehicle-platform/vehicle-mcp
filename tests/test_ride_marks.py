"""Tests for get_ride_marks read-only MCP tool (G5-3, D-133 item 4)."""

from __future__ import annotations

import json

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from tests.support import TOKEN, FakeVehicleServer
from vehicle_mcp.client import VehicleServerClient
from vehicle_mcp.config import Settings
from vehicle_mcp.tools import VehicleTools


def _tools(server: FakeVehicleServer) -> tuple[VehicleTools, VehicleServerClient]:
    client = VehicleServerClient(
        Settings(server_url="http://server.test", api_token=TOKEN),
        transport=httpx.MockTransport(server.handler),
    )
    return VehicleTools(client), client


def test_get_ride_marks_normal_returns_labelled_quantities_and_offsets() -> None:
    fake = FakeVehicleServer()
    tools, client = _tools(fake)
    try:
        result = tools.get_ride_marks("s1")
    finally:
        client.close()

    assert [req.method for req in fake.requests] == ["GET"]
    assert fake.requests[0].url.path == "/rides/s1/marks"

    assert result["available"] is True
    assert result["kind"] == "ride_marks"
    assert result["session_id"] == "s1"

    values_by_name = {item["name"]: item for item in result["values"]}
    assert values_by_name["mark_count"] == {
        "name": "mark_count",
        "unit": "count",
        "value": 2,
        "label": "ride mark count",
    }
    assert values_by_name["mark_count_misfire"]["value"] == 1
    assert values_by_name["mark_count_rough_road"]["value"] == 1

    marks = result["marks"]
    assert len(marks) == 2
    assert marks[0]["offset_s"] == 65.5
    assert marks[0]["offset"] == {
        "name": "mark_offset",
        "unit": "s",
        "value": 65.5,
        "label": "time offset from ride start",
    }
    assert marks[0]["kind"] == "misfire"
    assert marks[0]["note"] == {"untrusted_text": "stumbled under throttle"}
    assert marks[1]["offset_s"] == 180.0
    assert marks[1]["kind"] == "rough_road"
    assert marks[1]["note"] is None

    serialized = json.dumps(result)
    for forbidden in ("lat", "lon", "latitude", "longitude", "position", TOKEN):
        assert forbidden not in serialized


def test_get_ride_marks_empty_ride() -> None:
    fake = FakeVehicleServer()
    tools, client = _tools(fake)
    try:
        result = tools.get_ride_marks("s2")
    finally:
        client.close()

    assert result["available"] is True
    assert result["session_id"] == "s2"
    assert result["marks"] == []
    assert result["values"] == [
        {
            "name": "mark_count",
            "unit": "count",
            "value": 0,
            "label": "ride mark count",
        }
    ]


def test_get_ride_marks_unknown_ride_returns_safe_message() -> None:
    fake = FakeVehicleServer()
    tools, client = _tools(fake)
    try:
        result = tools.get_ride_marks("missing")
    finally:
        client.close()

    assert result["available"] is False
    assert result["session_id"] == "missing"
    assert result["marks"] == []
    assert "missing" in result["reason"]
    assert TOKEN not in result["reason"]
    assert "Traceback" not in result["reason"]


def test_get_ride_marks_older_server_without_route_returns_safe_message() -> None:
    fake = FakeVehicleServer(supports_marks=False)
    tools, client = _tools(fake)
    try:
        result = tools.get_ride_marks("s1")
    finally:
        client.close()

    assert result["available"] is False
    assert "older server" in result["reason"]


def test_get_ride_marks_server_error_raises_safe_tool_error() -> None:
    fake = FakeVehicleServer()
    tools, client = _tools(fake)
    try:
        with pytest.raises(ToolError) as excinfo:
            tools.get_ride_marks("unavailable")
    finally:
        client.close()

    message = str(excinfo.value)
    assert "503" in message
    assert TOKEN not in message
    assert "Traceback" not in message


def test_get_ride_marks_ignores_unexpected_position_fields_and_rejects_empty_id() -> None:
    fake = FakeVehicleServer()
    fake.marks_by_session["s1"]["marks"][0]["lat"] = 41.0
    fake.marks_by_session["s1"]["marks"][0]["lon"] = 29.0
    fake.marks_by_session["s1"]["marks"][0]["position"] = {"lat": 41.0}
    tools, client = _tools(fake)
    try:
        with pytest.raises(ToolError):
            tools.get_ride_marks("   ")
        result = tools.get_ride_marks("s1")
    finally:
        client.close()

    entry = result["marks"][0]
    assert set(entry.keys()) == {"offset_s", "offset", "kind", "note"}
