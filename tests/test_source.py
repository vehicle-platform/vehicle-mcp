from __future__ import annotations

from typing import Any

from vehicle_mcp.source import HistorySource
from vehicle_mcp.tools import VehicleTools

SERIES = {
    "name": "derived.sample",
    "kind": "derived",
    "unit": "u",
    "points_total": 4,
    "downsampled": False,
    "t_ms": [0, 100, 200, 300],
    "v": [1.0, 2.0, 3.0, 4.0],
}


class InMemorySource:
    """A non-HTTP HistorySource: the tools must not depend on vehicle-server."""

    def sessions(self) -> Any:
        return [{"session_id": "mem-1"}]

    def report(self, session_id: str, *, text: bool = False) -> Any:
        return "synthetic" if text else {"session_id": session_id}

    def series(
        self,
        session_id: str,
        *,
        signals: list[str] | None,
        max_points: int,
        from_ms: int | None = None,
        to_ms: int | None = None,
    ) -> Any:
        return {"session_id": session_id, "series": [SERIES]}


def test_tools_run_on_any_history_source() -> None:
    source: HistorySource = InMemorySource()
    tools = VehicleTools(source)

    assert tools.list_sessions() == {"sessions": [{"session_id": "mem-1"}]}
    assert tools.get_session_report("mem-1", format="text") == "synthetic"
    assert [s["name"] for s in tools.list_series("mem-1")["series"]] == ["derived.sample"]
    stats = tools.compute_signal_stats("mem-1", "derived.sample")
    assert (stats["mean"], stats["min"], stats["max"], stats["duration_ms"]) == (2.5, 1.0, 4.0, 300)
