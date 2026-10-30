from __future__ import annotations

import httpx
import pytest

from tests.support import RIDE_SESSIONS, TOKEN, FakeVehicleServer
from vehicle_mcp.client import VehicleServerClient
from vehicle_mcp.config import Settings
from vehicle_mcp.tools import VehicleTools


@pytest.fixture
def fake_server() -> FakeVehicleServer:
    return FakeVehicleServer()


@pytest.fixture
def client(fake_server: FakeVehicleServer) -> VehicleServerClient:
    client = VehicleServerClient(
        Settings(server_url="http://server.test", api_token=TOKEN),
        transport=httpx.MockTransport(fake_server.handler),
    )
    yield client
    client.close()


@pytest.fixture
def tools(client: VehicleServerClient) -> VehicleTools:
    return VehicleTools(client)


@pytest.fixture
def ride_tools(client: VehicleServerClient, fake_server: FakeVehicleServer) -> VehicleTools:
    fake_server.sessions = RIDE_SESSIONS
    return VehicleTools(client)
