from __future__ import annotations

import os

import httpx
import pytest
from hypothesis import HealthCheck, settings

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


@pytest.fixture
def old_server() -> FakeVehicleServer:
    """A pre-SV-2 server: /series works, /stats is a route-level 404."""
    return FakeVehicleServer(supports_stats=False)


@pytest.fixture
def old_tools(old_server: FakeVehicleServer) -> VehicleTools:
    client = VehicleServerClient(
        Settings(server_url="http://server.test", api_token=TOKEN),
        transport=httpx.MockTransport(old_server.handler),
    )
    yield VehicleTools(client)
    client.close()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", help="run tests that need the real model")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow: needs the real embedding model (--run-slow)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


# Hypothesis profiles (T31): bounded for PR CI; HYPOTHESIS_PROFILE=deep for the deep run.
settings.register_profile(
    "ci", max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "deep", max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
