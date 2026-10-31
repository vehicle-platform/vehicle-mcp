"""The data-source interface behind the history tools.

The tools never build URLs or know which backend serves them: they ask a
HistorySource for sessions, reports, series and exact statistics. vehicle-server's HTTP API
is the first implementation
(`client.VehicleServerClient`); another project plugs in its own source, and a
live source (Kuksa/VSS) gets its own interface when the live tools are built.
"""

from __future__ import annotations

from typing import Any, Protocol


class SourceError(RuntimeError):
    """A source call failed; the message is safe to show to the model."""


class StatsRouteUnavailable(SourceError):
    """The server has no exact-statistics route (an older server without SV-2)."""


class HistorySource(Protocol):
    """Read-only access to recorded sessions."""

    def sessions(self) -> Any:
        """The session index: a list of objects with at least `session_id`."""

    def report(self, session_id: str, *, text: bool = False) -> Any:
        """A session report as a JSON object, or as rendered text if `text`."""

    def series(
        self,
        session_id: str,
        *,
        signals: list[str] | None,
        max_points: int,
        from_ms: int | None = None,
        to_ms: int | None = None,
    ) -> Any:
        """Series in the GET /sessions/{id}/series shape: `series` entries with
        `name`, `kind`, `unit`, `points_total`, `downsampled`, `t_ms` and `v`."""

    def stats(
        self,
        session_id: str,
        *,
        signals: list[str] | None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> Any:
        """Exact statistics in the GET /sessions/{id}/stats shape: `series` entries
        with `name`, `kind`, `unit`, `count`, `mean`, `std`, `min`, `max`, `p05`,
        `p50`, `p95`, `t_first_ms` and `t_last_ms`."""
