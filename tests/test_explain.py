"""explain_context (D-131 item 2): reading, conditions, history, references, catalog."""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from tests import support
from tests.support import COOLANT_TEMP, ENGINE_SPEED, FAULT_ENTRY, TOKEN, FakeVehicleServer
from vehicle_mcp import explain
from vehicle_mcp.client import VehicleServerClient
from vehicle_mcp.config import Settings
from vehicle_mcp.server import build_server
from vehicle_mcp.tools import VehicleTools


def _coolant(values: list[float]) -> dict[str, Any]:
    return {**COOLANT_TEMP, "v": values}


# Newest first: hot (108.123 max), then three cooler rides, then a ride without coolant.
SESSIONS = [
    {"session_id": "s1", "created_utc": "2026-10-01T10:00:00Z"},
    {"session_id": "hotA", "created_utc": "2026-10-02T10:00:00Z"},
    {"session_id": "hotB", "created_utc": "2026-10-03T10:00:00Z"},
    {"session_id": "hotC", "created_utc": "2026-10-04T10:00:00Z"},
    {"session_id": "hot", "created_utc": "2026-10-05T10:00:00Z"},
]
SERIES = {
    "hotA": [ENGINE_SPEED, _coolant([70, 72, 75, 80, 85, 88, 90, 91])],
    "hotB": [ENGINE_SPEED, _coolant([72, 74, 78, 82, 86, 90, 92, 93])],
    "hotC": [ENGINE_SPEED, _coolant([75, 78, 80, 84, 88, 92, 95, 96])],
}
BINNED = {
    "name": "ride_coolant_time_at_temp",
    "status": "computed",
    "unit": "s",
    "kind": "binned",
    "axes": [
        {"input": "did.COOLANT_TEMP", "unit": "degC", "edges": [[80.0, 100.0], [100.0, 120.0]]}
    ],
    "cells": [
        {"index": 0, "axes": [0], "value": 30.0, "flag": None},
        {"index": 1, "axes": [1], "value": 10.0, "flag": None},
    ],
    "total_s": 40.0,
    "outside_s": 0.0,
}


@pytest.fixture(autouse=True)
def _rides(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, series in SERIES.items():
        monkeypatch.setitem(support.SERIES_BY_SESSION, name, series)


def _tools(server: FakeVehicleServer | None = None) -> tuple[VehicleTools, FakeVehicleServer]:
    server = server or FakeVehicleServer(sessions=SESSIONS)
    summary = copy.deepcopy(support.SUMMARY_BY_SESSION["hot"])
    summary["signals"].append(BINNED)
    server.summaries["hot"] = summary
    client = VehicleServerClient(
        Settings(server_url="http://server.test", api_token=TOKEN),
        transport=httpx.MockTransport(server.handler),
    )
    return VehicleTools(client), server


def _values(items: list[dict[str, Any]]) -> dict[str, float]:
    return {i["name"]: i["value"] for i in items}


def test_full_context_for_a_topic() -> None:
    tools, _ = _tools()
    result = tools.explain_context("coolant")
    assert result["kind"] == "explain_context"
    assert result["diagnosis"] is False
    assert "not a diagnosis" in result["note"]
    assert result["available"] is True
    assert (result["topic"], result["signal"], result["session_id"]) == (
        "coolant",
        "did.COOLANT_TEMP",
        "hot",
    )
    reading = result["reading"]
    assert reading["unit"] == "degC"
    assert reading["trend"] == "rising"
    values = _values(reading["values"])
    assert values["coolant_temp_max"] == 108.12
    assert values["coolant_temp_min"] == 80.0
    assert values["coolant_temp_last"] == 108.12
    assert all(set(v) == {"name", "unit", "value", "label"} for v in reading["values"])
    band = reading["time_in_bands"][0]
    assert (band["from"], band["to"], band["unit"]) == (80.0, 100.0, "degC")
    assert band["time"]["value"] == 30.0
    assert band["share"]["value"] == 75.0  # 30 s of the stored 40 s total

    # conditions: only what the server reports for the ride
    conditions = _values(result["conditions"])
    assert conditions["ride_stop_count"] == 3
    assert conditions["ride_time"] == 3601.46
    assert "ride_warmup_time" not in conditions
    assert "ride_fuel_used" not in conditions
    assert result["references"] == []
    assert "not expose reference values" in result["references_reason"]


def test_history_ranks_this_ride_against_previous_ones_newest_first() -> None:
    tools, _ = _tools()
    history = tools.explain_context("coolant")["history"]
    # s1 has no coolant series and is skipped; the others are newest first
    assert [r["session_id"] for r in history["rides"]] == ["hotC", "hotB", "hotA"]
    assert history["count"] == 3
    by_measure = {c["measure"]: c for c in history["comparison"]}
    high = by_measure["max"]
    assert (high["rank"], high["of"]) == (1, 4)
    assert high["previous_min"]["value"] == 91.0
    assert high["previous_median"]["value"] == 93.0
    assert high["previous_max"]["value"] == 96.0
    assert high["this_ride"]["value"] == 108.12
    assert high["outside_previous_range"] == "above"
    assert by_measure["min"]["outside_previous_range"] == "above"


def test_history_only_uses_rides_before_the_chosen_one() -> None:
    tools, _ = _tools()
    result = tools.explain_context("coolant", "hotB")
    assert result["session_id"] == "hotB"
    assert [r["session_id"] for r in result["history"]["rides"]] == ["hotA"]


def test_newest_ride_is_chosen_by_start_time_not_list_order() -> None:
    tools, _ = _tools(FakeVehicleServer(sessions=list(reversed(SESSIONS))))
    assert tools.explain_context("coolant")["session_id"] == "hot"


def test_history_is_capped_at_ten_rides(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = [{"session_id": "hot", "created_utc": "2026-11-01T10:00:00Z"}]
    for i in range(14):
        sid = f"old{i:02d}"
        monkeypatch.setitem(support.SERIES_BY_SESSION, sid, [_coolant([70 + i] * 8)])
        sessions.append({"session_id": sid, "created_utc": f"2026-10-{i + 1:02d}T10:00:00Z"})
    tools, _ = _tools(FakeVehicleServer(sessions=sessions))
    rides = tools.explain_context("coolant")["history"]["rides"]
    assert len(rides) == 10
    assert rides[0]["session_id"] == "old13"


def test_no_previous_ride_falls_back_to_the_trend() -> None:
    tools, _ = _tools(FakeVehicleServer(sessions=[SESSIONS[-1]]))
    result = tools.explain_context("coolant")
    assert result["history"]["count"] == 0
    assert "no previous ride" in result["history"]["reason"]
    assert "too few previous rides" in result["directions_shown"]["up"]
    assert [c["id"] for c in result["catalog"]] == [FAULT_ENTRY["id"]]


def test_catalog_entry_is_a_full_displayable_summary() -> None:
    tools, _ = _tools()
    result = tools.explain_context("coolant")
    assert list(result["directions_shown"]) == ["up"]
    entry = result["catalog"][0]
    assert entry["id"] == FAULT_ENTRY["id"]
    assert entry["advise_stop"] is True
    assert entry["rider_text"] == FAULT_ENTRY["rider_text"]
    assert entry["action"] == FAULT_ENTRY["action"]
    assert "match_score" not in entry
    assert entry["measurable"] is True
    # the explained signal's numbers live in `reading`, not repeated in the entry
    assert entry["signals"] == []
    assert entry["explained_signal_in_reading"] == "did.COOLANT_TEMP"


def test_catalog_needs_the_direction_the_data_shows() -> None:
    server = FakeVehicleServer(sessions=SESSIONS)
    server.fault_entries[0]["symptoms"] = [{"signal": "did.COOLANT_TEMP", "direction": "down"}]
    tools, _ = _tools(server)
    result = tools.explain_context("coolant")
    assert result["catalog"] == []
    assert "no catalog entry" in result["catalog_reason"]


def test_unconfirmed_entries_appear_only_as_data_only() -> None:
    server = FakeVehicleServer(sessions=SESSIONS)
    server.fault_entries.append(
        {
            **FAULT_ENTRY,
            "id": "U-1",
            "displayable": False,
            "root_cause": "SECRET-CAUSE",
            "rider_text": "SECRET-TEXT",
            "action": "SECRET-ACTION",
            "needs": ["did.COOLANT_TEMP", "did.ENGINE_SPEED"],
        }
    )
    tools, _ = _tools(server)
    result = tools.explain_context("coolant")
    assert [c.get("id") or c["kind"] for c in result["catalog"]] == [FAULT_ENTRY["id"], "data_only"]
    text = str(result)
    assert "SECRET" not in text
    assert "U-1" not in text
    data_only = result["catalog"][1]
    assert [c["signal"] for c in data_only["signals"]] == ["did.ENGINE_SPEED"]


def test_refused_unconfirmed_still_gives_displayable_entries() -> None:
    tools, _ = _tools(FakeVehicleServer(sessions=SESSIONS, refuse_unconfirmed=403))
    result = tools.explain_context("coolant")
    assert [c["id"] for c in result["catalog"]] == [FAULT_ENTRY["id"]]
    assert "did not allow unconfirmed" in result["catalog_reason"]


def test_old_server_without_catalog_or_summary_degrades() -> None:
    server = FakeVehicleServer(sessions=SESSIONS, supports_faults=False)
    tools, _ = _tools(server)
    server.summaries.clear()
    result = tools.explain_context("coolant")
    assert result["available"] is True
    assert result["catalog"] == []
    assert "older server" in result["catalog_reason"]
    assert result["conditions"] == [] or all("ride_" not in c["name"] for c in result["conditions"])
    assert result["reading"]["time_in_bands"] == []


def test_signal_names_resolve_like_topic_words() -> None:
    tools, _ = _tools()
    for query in ("did.COOLANT_TEMP", "coolant temp", "COOLANT_TEMP", "overheating"):
        assert tools.explain_context(query)["signal"] == "did.COOLANT_TEMP", query
    assert tools.explain_context("engine speed")["signal"] == "did.ENGINE_SPEED"


def test_unknown_topic_and_bad_input() -> None:
    tools, _ = _tools()
    result = tools.explain_context("flux capacitor")
    assert result["available"] is False
    assert result["diagnosis"] is False
    assert "coolant" in result["reason"]
    with pytest.raises(ToolError, match="must not be empty"):
        tools.explain_context("  ")
    with pytest.raises(ToolError, match="position"):
        tools.explain_context("sensor.gps_latitude")
    empty, _ = _tools(FakeVehicleServer(sessions=[]))
    assert empty.explain_context("coolant")["available"] is False


def test_tool_is_registered_and_only_reads() -> None:
    tools, server = _tools()
    mcp = build_server(tools._source)  # noqa: SLF001
    result = asyncio.run(mcp.call_tool("explain_context", {"topic_or_signal": "coolant"}))
    assert result.is_error is False
    assert result.structured_content["kind"] == "explain_context"
    assert all(r.method == "GET" for r in server.requests)


def test_direction_rules() -> None:
    def cmp(this: float, previous: list[float]) -> dict[str, Any]:
        return explain.compare(this, previous)

    hot = {"max": cmp(9, [5, 6, 7]), "min": cmp(5, [4, 5, 6])}
    assert list(explain.directions_shown("flat", hot)) == ["up"]
    # a tie at the top of a varied history still counts, a flat history does not
    tie = {"max": cmp(7, [5, 6, 7]), "min": cmp(5, [4, 5, 6])}
    assert list(explain.directions_shown("flat", tie)) == ["up"]
    same = {"max": cmp(7, [7, 7, 7]), "min": cmp(7, [7, 7, 7])}
    assert explain.directions_shown("rising", same) == {}  # history wins over the trend
    low = {"max": cmp(5, [7, 8, 9]), "min": cmp(1, [4, 5, 6])}
    assert list(explain.directions_shown("flat", low)) == ["down"]
    assert "unstable" in explain.directions_shown("unstable", same)
    assert list(explain.directions_shown("falling", {})) == ["down"]
    one_previous = {"max": cmp(9, [5]), "min": cmp(5, [4])}
    assert list(explain.directions_shown("rising", one_previous)) == ["up"]  # trend fallback
    assert explain.compare(5, [5, 7])["outside_previous_range"] == "within"
    assert explain.compare(3, [5, 7])["outside_previous_range"] == "below"


def test_related_table_names_are_ride_summary_names_not_signals() -> None:
    for scalars, fragments in explain.RELATED.values():
        assert all(name.startswith("ride_") for name in scalars)
        assert all(fragment.islower() for fragment in fragments)
    assert set(explain.RELATED) >= {"coolant", "battery", "speed", "rpm", "idle", "fuel"}
