"""Read-only HTTP client for the vehicle-server API.

Only GET requests are issued. Errors are translated into messages that are safe
to show to the model: no stack traces and never the bearer token.
"""

from __future__ import annotations

from typing import Any

import httpx

from .config import Settings

_MAX_DETAIL_CHARS = 300

_STATUS_HINTS = {
    401: "vehicle-server rejected the bearer token (401); check VEHICLE_API_TOKEN",
    403: "vehicle-server refused this read (403); the token is not allowed to access it",
    503: "vehicle-server cannot serve authenticated reads right now (503); "
    "VEHICLE_API_TOKEN may not be configured on the server",
}


class VehicleServerError(RuntimeError):
    """A vehicle-server call failed; the message is safe to show to the model."""


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:_MAX_DETAIL_CHARS] if text else "no detail"
    if isinstance(payload, dict) and "detail" in payload:
        text = str(payload["detail"])
    else:
        text = str(payload)
    return text[:_MAX_DETAIL_CHARS]


class VehicleServerClient:
    """Thin synchronous client; every method performs exactly one GET."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._client = httpx.Client(
            base_url=settings.server_url,
            timeout=settings.timeout_s,
            headers={"Authorization": f"Bearer {settings.api_token}"},
            transport=transport,
        )

    @property
    def base_url(self) -> str:
        return self._settings.server_url

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        try:
            response = self._client.get(path, params=params)
        except httpx.TimeoutException:
            raise VehicleServerError(
                f"vehicle-server at {self.base_url} did not answer within "
                f"{self._settings.timeout_s:g} s"
            ) from None
        except httpx.HTTPError as exc:
            raise VehicleServerError(
                f"cannot reach vehicle-server at {self.base_url} ({type(exc).__name__})"
            ) from None

        if response.status_code >= 400:
            raise VehicleServerError(self._status_message(response))
        return response

    def _status_message(self, response: httpx.Response) -> str:
        status = response.status_code
        # A server that echoes the request headers must not leak the token to the model.
        detail = _detail(response).replace(self._settings.api_token, "***")
        hint = _STATUS_HINTS.get(status)
        if hint is not None:
            return f"{hint}: {detail}"
        return f"vehicle-server returned HTTP {status}: {detail}"

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = self._get(path, params)
        try:
            return response.json()
        except ValueError:
            raise VehicleServerError(
                f"vehicle-server returned a non-JSON response for {path}"
            ) from None

    def get_text(self, path: str, params: dict[str, Any] | None = None) -> str:
        return self._get(path, params).text
