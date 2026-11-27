"""G6-6 (T58 follow-up): no internal identifiers (`snake_case` or `did.`/`sensor.`) in
rider-facing text fields returned by vehicle-mcp tools.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx

from tests.knowledge_support import FakeEmbedder, catalog_payload, write_docs
from tests.support import FAULT_ENTRY, HOT_SESSIONS, TOKEN, FakeVehicleServer
from vehicle_mcp.client import VehicleServerClient
from vehicle_mcp.config import Settings
from vehicle_mcp.knowledge.search import KnowledgeSearch
from vehicle_mcp.knowledge.sources import catalog_chunks, docs_chunks
from vehicle_mcp.knowledge.store import create_index
from vehicle_mcp.server import build_server
from vehicle_mcp.tools import VehicleTools

INTERNAL_ID_RE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b|\b(?:did|sensor)\.[A-Za-z0-9_]+")

# Rider-facing text fields inspected recursively in tool outputs.
RIDER_TEXT_FIELDS = frozenset(
    {
        "label",
        "title",
        "section",
        "summary",
        "text",
        "snippet",
        "symptoms",
        "root_cause",
        "rider_text",
        "action",
        "matched_phrase",
        "available_signals",
        "measurement_note",
        "rank_note",
        "trend_basis",
        "warnings",
        "reason",
        "catalog_reason",
        "references_reason",
        "ride_reason",
        "ride_scalars_reason",
        "approximate_reason",
        "directions_shown",
    }
)

# Machine, identifier, protocol, and untrusted-device fields exempt from rider-text checks.
EXEMPT_FIELDS = frozenset(
    {
        "name",
        "kind",
        "id",
        "fault_id",
        "session_id",
        "device_id",
        "ride_id",
        "signal",
        "explained_signal_in_reading",
        "input",
        "topic",
        "subsystem",
        "source",
        "url_or_path",
        "license",
        "locale",
        "unit",
        "status",
        "flag",
        "measure",
        "direction",
        "outside_previous_range",
        "trend",
        "trend_over_ride",
        "catalog_direction",
        "ride_definition",
        "defs_version",
        "app_version",
        "vehicle",
        "content_hash",
        "created_utc",
        "imported_utc",
        "ride_date",
        "start_utc",
        "end_utc",
        "needs",
        "missing",
        "missing_in_ride",
        "dropped_position_names",
        "dropped_position_series",
        "topics",
        "undated_ride_reason",  # API-schema diagnostic reason naming created_utc
        "note",  # LLM protocol instructions (e.g. advise_stop, untrusted_text guidance)
        "query",  # echoes the caller's own query string
    }
)


def _extract_field_texts(child: str, value: Any) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(child, value)]
    if isinstance(value, dict):
        if isinstance(value.get("untrusted_text"), str):
            return [(child, value["untrusted_text"])]
        return [(f"{child}.{k}", v) for k, v in value.items() if isinstance(v, str)]
    if isinstance(value, list):
        return [(f"{child}[{i}]", v) for i, v in enumerate(value) if isinstance(v, str)]
    return []


def _collect_rider_texts(node: Any, path: str = "") -> list[tuple[str, str]]:
    """Collect `(json_path, text)` pairs for every rider-facing field in `node`."""
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in EXEMPT_FIELDS:
                continue
            child = f"{path}.{key}" if path else str(key)
            if key in RIDER_TEXT_FIELDS:
                found.extend(_extract_field_texts(child, value))
            else:
                found.extend(_collect_rider_texts(value, child))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            found.extend(_collect_rider_texts(item, f"{path}[{idx}]"))
    return found


def _build_knowledge(tmp_path: Path) -> KnowledgeSearch:
    db = tmp_path / "knowledge.sqlite"
    docs_dir = write_docs(tmp_path / "docs")
    payload = catalog_payload()
    # Include a multi-word snake_case subsystem matching real vehicle-defs entries (ENG-001).
    payload["entries"][0]["subsystem"] = "engine_mechanical"
    fake = FakeEmbedder()
    chunks = [*catalog_chunks(payload), *docs_chunks(docs_dir)]
    index = create_index(db, dim=fake.dim, model_id=fake.model_id)
    index.add_chunks(chunks, fake.embed_passages([c.text for c in chunks]))
    index.close()
    return KnowledgeSearch(db, embedder_factory=lambda: fake)


def _all_tool_outputs(tmp_path: Path) -> dict[str, list[Any]]:
    server = FakeVehicleServer(sessions=HOT_SESSIONS)
    client = VehicleServerClient(
        Settings(server_url="http://vehicle.test", api_token=TOKEN),
        transport=httpx.MockTransport(server.handler),
    )
    knowledge = _build_knowledge(tmp_path)
    tools = VehicleTools(client, knowledge=knowledge)

    # Verify the test covers every tool registered on the MCP server.
    mcp_server = build_server(client, knowledge)
    registered = {t.name for t in mcp_server._tool_manager.list_tools()}

    outputs: dict[str, list[Any]] = {
        "list_sessions": [tools.list_sessions()],
        "get_session_report": [tools.get_session_report("s1", format="json")],
        "list_series": [tools.list_series("hot")],
        "get_timeseries_slice": [tools.get_timeseries_slice("s1", ["did.ENGINE_SPEED"])],
        "compute_signal_stats": [tools.compute_signal_stats("hot", "did.COOLANT_TEMP")],
        "compare_sessions": [tools.compare_sessions("s1", "s2", "did.ENGINE_SPEED")],
        "query_ride_history": [tools.query_ride_history("2026-10-01", "2026-10-06")],
        "get_ride_summary": [tools.get_ride_summary("s1"), tools.get_ride_summary("hot")],
        "get_ride_marks": [tools.get_ride_marks("s1")],
        "list_fault_catalog": [tools.list_fault_catalog()],
        "get_fault": [tools.get_fault(FAULT_ENTRY["id"])],
        "get_data_card": [
            tools.get_data_card("coolant", "hot"),
            tools.get_data_card("ride", "hot"),
            tools.get_data_card("gears", "hot"),
        ],
        "check_symptom": [
            tools.check_symptom("coolant overheating", "hot"),
            tools.check_symptom("valves making noise", "hot"),
            tools.check_symptom("unrelated complaint xyz", "hot"),
        ],
        "explain_context": [
            tools.explain_context("coolant", "hot"),
            tools.explain_context("battery", "hot"),
        ],
        "search_knowledge": [
            tools.search_knowledge("valve ticking", locale="en"),
            tools.search_knowledge("coolant fan", locale="en"),
            tools.search_knowledge("battery keeps going flat", locale="en"),
        ],
    }
    assert set(outputs) == registered, f"tool coverage mismatch: {set(outputs) ^ registered}"
    return outputs


def test_no_internal_identifiers_in_rider_facing_tool_output(tmp_path: Path) -> None:
    violations: list[str] = []
    for tool_name, results in _all_tool_outputs(tmp_path).items():
        for idx, res in enumerate(results):
            for field_path, text in _collect_rider_texts(res):
                matches = INTERNAL_ID_RE.findall(text)
                if matches:
                    violations.append(f"{tool_name}[{idx}].{field_path}: {matches} in {text!r}")
    assert not violations, "Internal identifiers leaked in rider-facing fields:\n" + "\n".join(
        violations
    )
