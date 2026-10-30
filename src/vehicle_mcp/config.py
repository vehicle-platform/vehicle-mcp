"""Environment-only configuration for vehicle-mcp.

The bearer token is read once from VEHICLE_API_TOKEN, kept in memory and never
logged or returned. VEHICLE_SERVER_URL defaults to the local vehicle-server.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse

DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_S = 10.0


class ConfigError(RuntimeError):
    """The environment cannot start the server; the message is safe to print."""


@dataclass(frozen=True)
class Settings:
    server_url: str
    api_token: str = field(repr=False)
    timeout_s: float = REQUEST_TIMEOUT_S

    def __repr__(self) -> str:
        return (
            f"Settings(server_url={self.server_url!r}, api_token='***', "
            f"timeout_s={self.timeout_s!r})"
        )


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from the environment, raising ConfigError when it is incomplete."""
    source = os.environ if env is None else env
    token = source.get("VEHICLE_API_TOKEN", "").strip()
    if not token:
        raise ConfigError(
            "VEHICLE_API_TOKEN is required; set it to the bearer token of vehicle-server"
        )

    url = source.get("VEHICLE_SERVER_URL", DEFAULT_SERVER_URL).strip() or DEFAULT_SERVER_URL
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError("VEHICLE_SERVER_URL must be a valid http:// or https:// URL")

    return Settings(server_url=url.rstrip("/"), api_token=token)
