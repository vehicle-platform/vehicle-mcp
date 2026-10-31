"""Synthetic vehicle-server responses for the tests (no network, no real ride data)."""

from __future__ import annotations

from typing import Any

import httpx

from vehicle_mcp.stats import compute_stats

TOKEN = "test-token"

ENGINE_SPEED: dict[str, Any] = {
    "name": "did.ENGINE_SPEED",
    "kind": "did",
    "unit": "rpm",
    "points_total": 5,
    "downsampled": False,
    "t_ms": [0, 100, 200, 300, 400],
    "v": [1000, 2000, 3000, 4000, 5000],
}

ENGINE_SPEED_B: dict[str, Any] = {
    **ENGINE_SPEED,
    "v": [1500, 2500, 3500, 4500, 5500],
}

LEAN_ANGLE: dict[str, Any] = {
    "name": "derived.LEAN_ANGLE",
    "kind": "derived",
    "unit": "deg",
    "points_total": 5,
    "downsampled": False,
    "t_ms": [0, 100, 200, 300, 400],
    "v": [10, 20, 30, 20, 10],
}

LEAN_ANGLE_RAD: dict[str, Any] = {
    **LEAN_ANGLE,
    "unit": "rad",
    "v": [0.17, 0.35, 0.52, 0.35, 0.17],
}

HAND: dict[str, Any] = {
    "name": "did.HAND",
    "kind": "did",
    "unit": "count",
    "points_total": 8,
    "downsampled": False,
    "t_ms": [0, 100, 200, 300, 400, 500, 600, 700],
    "v": [2, 4, 4, 4, 5, 5, 7, 9],
}

DOWNSAMPLED: dict[str, Any] = {
    "name": "did.DOWNSAMPLED",
    "kind": "did",
    "unit": "rpm",
    "points_total": 10_000,
    "downsampled": True,
    "t_ms": [0, 1000],
    "v": [1, 9],
}

EMPTY: dict[str, Any] = {
    "name": "did.EMPTY",
    "kind": "did",
    "unit": "count",
    "points_total": 0,
    "downsampled": False,
    "t_ms": [],
    "v": [],
}

GPS_LATITUDE: dict[str, Any] = {
    "name": "sensor.gps_latitude",
    "kind": "sensor",
    "unit": "deg",
    "points_total": 3,
    "downsampled": False,
    "t_ms": [0, 100, 200],
    "v": [41.0, 41.1, 41.2],
}

SERIES_BY_SESSION: dict[str, list[dict[str, Any]]] = {
    "s1": [ENGINE_SPEED, LEAN_ANGLE, HAND, GPS_LATITUDE],
    "s2": [ENGINE_SPEED_B, GPS_LATITUDE],
    "rad": [LEAN_ANGLE_RAD, GPS_LATITUDE],
    "down": [DOWNSAMPLED, GPS_LATITUDE],
    "empty": [EMPTY],
    "clean": [ENGINE_SPEED],
}

RIDE_SESSIONS: list[dict[str, Any]] = [
    {
        "session_id": "ride-early",
        "created_utc": "2026-10-01T09:00:00Z",
        "packet_count": 100,
        "loss_percent": 0.1,
        "status": "ok",
    },
    {
        "session_id": "ride-middle",
        "created_utc": "2026-10-01T22:00:00+02:00",
        "packet_count": 200,
        "loss_percent": 0.2,
        "status": "ok",
    },
    {
        "session_id": "ride-failed",
        "created_utc": "2026-10-02T12:00:00Z",
        "packet_count": 400,
        "loss_percent": 0.4,
        "status": "fail",
    },
    {
        "session_id": "ride-late",
        "created_utc": "2026-10-03T09:00:00Z",
        "packet_count": 300,
        "loss_percent": 0.3,
        "status": "ok",
    },
    {
        "session_id": "ride-september",
        "created_utc": "2026-09-30T23:59:59Z",
        "packet_count": 999,
        "loss_percent": 0.9,
        "status": "warn",
    },
    {
        "session_id": "ride-undated",
        "packet_count": 50,
        "loss_percent": 0.0,
        "status": "warn",
    },
    {
        "session_id": "ride-bad-date",
        "created_utc": "not-a-date",
        "packet_count": 25,
        "loss_percent": 0.2,
        "status": "ok",
    },
]

SESSION_INDEX: list[dict[str, Any]] = [
    {
        "session_id": "s1",
        "created_utc": "2026-10-01T10:00:00Z",
        "imported_utc": "2026-10-01T10:05:00Z",
        "content_hash": "sha256:aaaa",
        "app_version": "0.5.0",
        "ble_schema_version": "1",
        "packet_count": 1200,
        "loss_percent": 0.0,
        "status": "ok",
        "has_imu": True,
        "defs_version": "0.19.1",
        "has_gps": True,
        "has_vehicle_signals": True,
        "has_derived": True,
    },
    {
        "session_id": "s2",
        "created_utc": "2026-10-02T10:00:00Z",
        "imported_utc": "2026-10-02T10:05:00Z",
        "content_hash": "sha256:bbbb",
        "app_version": "0.5.0",
        "ble_schema_version": "1",
        "packet_count": 1300,
        "loss_percent": 0.1,
        "status": "ok",
        "has_imu": True,
        "defs_version": "0.19.1",
        "has_gps": True,
        "has_vehicle_signals": True,
        "has_derived": True,
    },
]

FAILURES = {
    "missing": 404,
    "bad-request": 422,
    "unavailable": 503,
    "unauthorized": 401,
    "forbidden": 403,
}

DETAILS = {
    401: "bad bearer token",
    403: "read not allowed",
    404: "unknown session",
    422: "unknown series",
    503: "authenticated reads are not configured",
}


class FakeVehicleServer:
    """A routed httpx.MockTransport handler with recorded requests."""

    def __init__(
        self,
        sessions: list[dict[str, Any]] | None = None,
        *,
        supports_stats: bool = True,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.sessions = sessions if sessions is not None else SESSION_INDEX
        self.supports_stats = supports_stats

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.method == "GET", f"unexpected {request.method} {request.url}"
        self.requests.append(request)

        parts = [part for part in request.url.path.split("/") if part]
        if parts == ["sessions"]:
            return httpx.Response(200, json=self.sessions)

        if len(parts) == 3 and parts[0] == "sessions":
            session_id, leaf = parts[1], parts[2]
            if session_id in FAILURES:
                status = FAILURES[session_id]
                return httpx.Response(status, json={"detail": f"{DETAILS[status]}: {session_id}"})
            if session_id not in SERIES_BY_SESSION:
                return httpx.Response(404, json={"detail": f"unknown session: {session_id}"})
            if leaf == "report":
                return self._report(request, session_id)
            if leaf == "series":
                return self._series(request, session_id)
            if leaf == "stats":
                if not self.supports_stats:
                    return httpx.Response(404, json={"detail": "Not Found"})
                return self._stats(request, session_id)
        return httpx.Response(404, json={"detail": f"unknown path: {request.url.path}"})

    def _report(self, request: httpx.Request, session_id: str) -> httpx.Response:
        if request.url.params.get("format") == "text":
            return httpx.Response(200, text=f"# Session {session_id}\n\nsynthetic report\n")
        return httpx.Response(
            200,
            json={
                "session_id": session_id,
                "summary": {"packet_count": 1200, "loss_percent": 0.0},
                "notes": ["synthetic test data"],
            },
        )

    def _stats(self, request: httpx.Request, session_id: str) -> httpx.Response:
        params = request.url.params
        available = SERIES_BY_SESSION[session_id]
        by_name = {entry["name"]: entry for entry in available}

        requested = params.get("signals")
        if requested is None:
            entries = available
        else:
            names = requested.split(",")
            unknown = [name for name in names if name not in by_name]
            if unknown:
                return httpx.Response(
                    404,
                    json={"detail": f"unknown or unavailable series: {', '.join(unknown)}"},
                )
            entries = [by_name[name] for name in names]

        series = []
        for entry in entries:
            summary = compute_stats({**entry, "downsampled": False})
            if entry.get("downsampled"):
                summary["count"] = entry.get("points_total", summary["count"])
            times = entry["t_ms"]
            series.append(
                {
                    "name": entry["name"],
                    "kind": entry["kind"],
                    "unit": entry["unit"],
                    "count": summary["count"],
                    "mean": summary["mean"],
                    "std": summary["std"],
                    "min": summary["min"],
                    "max": summary["max"],
                    "p05": summary["p05"],
                    "p50": summary["p50"],
                    "p95": summary["p95"],
                    "t_first_ms": min(times) if times else None,
                    "t_last_ms": max(times) if times else None,
                }
            )
        return httpx.Response(
            200,
            json={
                "session_id": session_id,
                "t0_device_ms": 0,
                "start_ms": params.get("start_ms"),
                "end_ms": params.get("end_ms"),
                "series": series,
            },
        )

    def _series(self, request: httpx.Request, session_id: str) -> httpx.Response:
        params = request.url.params
        max_points = int(params.get("max_points", "2000"))
        available = SERIES_BY_SESSION[session_id]
        by_name = {entry["name"]: entry for entry in available}

        requested = params.get("signals")
        if requested is None:
            entries = available
        else:
            names = requested.split(",")
            unknown = [name for name in names if name not in by_name]
            if unknown:
                return httpx.Response(
                    422,
                    json={"detail": f"unknown or unavailable series: {', '.join(unknown)}"},
                )
            entries = [by_name[name] for name in names]

        return httpx.Response(
            200,
            json={
                "session_id": session_id,
                "t0_device_ms": 0,
                "max_points": max_points,
                "series": entries,
            },
        )
