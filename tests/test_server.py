from __future__ import annotations

import asyncio

import httpx
import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from tests.support import TOKEN, FakeVehicleServer
from vehicle_mcp.client import VehicleServerClient
from vehicle_mcp.config import Settings
from vehicle_mcp.server import build_server, main


def make_server(fake_server: FakeVehicleServer) -> tuple[MCPServer, VehicleServerClient]:
    client = VehicleServerClient(
        Settings(server_url="http://server.test", api_token=TOKEN),
        transport=httpx.MockTransport(fake_server.handler),
    )
    return build_server(client), client


def test_registers_exactly_the_read_only_tools() -> None:
    server, client = make_server(FakeVehicleServer())
    try:
        tools = asyncio.run(server.list_tools())
    finally:
        client.close()

    assert {tool.name for tool in tools} == {
        "list_sessions",
        "get_session_report",
        "list_series",
        "get_timeseries_slice",
        "compute_signal_stats",
        "compare_sessions",
        "query_ride_history",
    }
    assert all("position" not in tool.name for tool in tools)


def test_missing_token_exits_before_network(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("VEHICLE_API_TOKEN", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
    assert "VEHICLE_API_TOKEN" in capsys.readouterr().err


def test_tool_errors_are_explanatory_without_traceback_or_token() -> None:
    server, client = make_server(FakeVehicleServer())
    try:
        with pytest.raises(ToolError) as excinfo:
            asyncio.run(server.call_tool("list_series", {"session_id": "missing"}))
    finally:
        client.close()

    message = str(excinfo.value)
    assert "404" in message
    assert "Traceback" not in message
    assert TOKEN not in message


def test_happy_path_through_mcp() -> None:
    server, client = make_server(FakeVehicleServer())
    try:
        sessions = asyncio.run(server.call_tool("list_sessions", {}))
        report_json = asyncio.run(server.call_tool("get_session_report", {"session_id": "s1"}))
        report_text = asyncio.run(
            server.call_tool("get_session_report", {"session_id": "s1", "format": "text"})
        )
        history = asyncio.run(
            server.call_tool(
                "query_ride_history", {"start_date": "2026-10-01", "end_date": "2026-10-02"}
            )
        )
    finally:
        client.close()

    assert sessions.is_error is False
    assert '"s1"' in "\n".join(block.text for block in sessions.content)
    assert report_json.is_error is False
    assert '"summary"' in "\n".join(block.text for block in report_json.content)
    assert report_text.is_error is False
    assert "# Session s1" in "\n".join(block.text for block in report_text.content)
    assert history.is_error is False
    assert '"undated"' in "\n".join(block.text for block in history.content)
