"""The data-source interface behind the history tools.

The tools never build URLs or know which backend serves them: they ask a
HistorySource for sessions, reports and series and compute every number
themselves. vehicle-server's HTTP API is the first implementation
(`client.VehicleServerClient`); another project plugs in its own source, and a
live source (Kuksa/VSS) gets its own interface when the live tools are built.
"""

from __future__ import annotations

from typing import Any, Protocol


class SourceError(RuntimeError):
    """A source call failed; the message is safe to show to the model."""


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
