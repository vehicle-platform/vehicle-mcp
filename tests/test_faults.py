from __future__ import annotations

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


def test_list_is_compact_and_never_asks_for_unconfirmed(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    result = tools.list_fault_catalog()
    assert result["available"] is True
    assert result["count"] == 1
    entry = result["entries"][0]
    assert entry["id"] == "F-TEST-01"
    assert entry["severity"] == 8
    assert entry["advise_stop"] is True
    assert "did.COOLANT_TEMP rising (at idle)" in entry["symptoms"]
    assert "rider_text" not in entry
    assert "stop safely" in result["note"]
    url = fake_server.requests[-1].url
    assert url.path == "/faults"
    assert "include_unconfirmed" not in str(url)


def test_list_passes_the_subsystem_filter(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    result = tools.list_fault_catalog("brakes")
    assert result["entries"] == []
    assert fake_server.requests[-1].url.params["subsystem"] == "brakes"
    assert "include_unconfirmed" not in str(fake_server.requests[-1].url)


def test_unknown_subsystem_is_a_clear_error(tools: VehicleTools) -> None:
    with pytest.raises(ToolError, match="not a known fault subsystem"):
        tools.list_fault_catalog("warp-drive")


def test_get_fault_returns_the_full_entry(
    tools: VehicleTools, fake_server: FakeVehicleServer
) -> None:
    result = tools.get_fault("F-TEST-01")
    assert result["available"] is True
    assert result["entry"]["rider_text"].startswith("Stop safely")
    assert result["entry"]["sources"][0]["ref"] == "synthetic"
    assert "include_unconfirmed" not in str(fake_server.requests[-1].url)


def test_get_fault_404_is_not_available_with_the_reason(tools: VehicleTools) -> None:
    result = tools.get_fault("F-NOPE")
    assert result["available"] is False
    assert "unknown fault: F-NOPE" in result["reason"]


def test_older_server_404_on_faults_is_not_available() -> None:
    tools, client = _tools(FakeVehicleServer(supports_faults=False))
    try:
        listed = tools.list_fault_catalog()
        one = tools.get_fault("F-TEST-01")
    finally:
        client.close()
    assert listed["available"] is False
    assert "older server" in listed["reason"]
    assert one["available"] is False
    assert "older server" in one["reason"]
