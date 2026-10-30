from __future__ import annotations

import pytest

from vehicle_mcp.config import (
    DEFAULT_SERVER_URL,
    ConfigError,
    Settings,
    load_settings,
)


def test_defaults_to_local_server() -> None:
    settings = load_settings({"VEHICLE_API_TOKEN": "secret"})
    assert settings.server_url == DEFAULT_SERVER_URL
    assert settings.api_token == "secret"
    assert settings.timeout_s == 10.0


def test_reads_url_and_strips_trailing_slash() -> None:
    settings = load_settings(
        {"VEHICLE_API_TOKEN": "secret", "VEHICLE_SERVER_URL": "https://telemetry.example/"}
    )
    assert settings.server_url == "https://telemetry.example"


def test_blank_url_falls_back_to_default() -> None:
    settings = load_settings({"VEHICLE_API_TOKEN": "secret", "VEHICLE_SERVER_URL": "  "})
    assert settings.server_url == DEFAULT_SERVER_URL


def test_missing_token_is_a_startup_error() -> None:
    with pytest.raises(ConfigError, match="VEHICLE_API_TOKEN"):
        load_settings({})


def test_blank_token_is_a_startup_error() -> None:
    with pytest.raises(ConfigError, match="VEHICLE_API_TOKEN"):
        load_settings({"VEHICLE_API_TOKEN": "   "})


@pytest.mark.parametrize("url", ["ftp://example", "http://", "not-a-url"])
def test_non_http_url_is_rejected(url: str) -> None:
    with pytest.raises(ConfigError, match="http"):
        load_settings({"VEHICLE_API_TOKEN": "secret", "VEHICLE_SERVER_URL": url})


def test_repr_hides_the_token() -> None:
    settings = Settings(server_url="http://server.test", api_token="super-secret")
    assert "super-secret" not in repr(settings)
    assert "api_token='***'" in repr(settings)
