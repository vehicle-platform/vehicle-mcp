"""Small-model tool specs: short descriptions, short argument docs, minimal JSON schemas.

A 4k-context local model reads every tool spec in its router prompt, so each description
is at most 160 characters and each argument description at most 80 (tests enforce both,
and a total budget). The long developer documentation stays in the method docstrings.
The published schemas are trimmed (no titles, optional arguments are plain types);
validation still happens in the tool function's own argument model.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from .matching import TOPIC_KEYS

MAX_DESCRIPTION_CHARS = 160
MAX_ARGUMENT_CHARS = 80

DESCRIPTIONS: dict[str, str] = {
    "list_sessions": "List recorded rides, newest first. Use first to find a ride id.",
    "get_session_report": "Read one ride's stored report (data quality, notes) as JSON or text.",
    "list_series": "List the signal names and units recorded in one ride.",
    "get_timeseries_slice": (
        "Raw samples of up to 8 signals. Context only; use compute_signal_stats for numbers."
    ),
    "compute_signal_stats": (
        "Min, mean, max, median and spread of one signal in one ride. For average, highest, lowest."
    ),
    "compare_sessions": "Compare one signal between two rides (stats and b minus a difference).",
    "query_ride_history": (
        "List rides in a date range with their stored totals. For last week or this month."
    ),
    "get_ride_summary": (
        "One ride's stored totals: ride time, distance, fuel. For how long or how far."
    ),
    "list_fault_catalog": "List possible faults from the catalog. NOT detected faults.",
    "get_fault": "Read one catalog fault in full: cause and advice. Not a diagnosis.",
    "get_data_card": (
        "The ride's numbers for a topic (coolant, battery, idle, "
        "speed, rpm, fuel...). Latest ride by default."
    ),
    "check_symptom": (
        "Match a rider's complaint (noise, smell, light, hot, weak) "
        "to catalog causes and ride data. Not a diagnosis."
    ),
    "search_knowledge": (
        "Search catalog, docs and manuals for text with citations. References, not measurements."
    ),
    "explain_context": (
        "Context to explain a reading: ride numbers, conditions, past rides, catalog "
        "causes. For why or is it normal. Not a diagnosis."
    ),
}

# argument name -> description, per tool. Self-explanatory arguments (ride ids, windows) have none.
ARGUMENTS: dict[str, dict[str, str]] = {
    "get_timeseries_slice": {"signals": "Names from list_series, at most 8."},
    "compute_signal_stats": {"signal": "Name from list_series."},
    "compare_sessions": {"signal": "Name present in both rides."},
    "query_ride_history": {"start_date": "YYYY-MM-DD.", "end_date": "YYYY-MM-DD."},
    "list_fault_catalog": {"subsystem": "Optional, e.g. cooling."},
    "get_fault": {"fault_id": "Id from list_fault_catalog."},
    "get_data_card": {"session_id": "Omit for the latest ride."},
    "check_symptom": {
        "complaint": "The rider's own words.",
        "session_id": "Omit for the latest ride.",
        "locale": "Language code: en, tr.",
    },
    "search_knowledge": {
        "query": "Question or words, any language.",
        "locale": "Language code: en, tr, or any.",
        "k": "Number of hits, 1 to 10.",
    },
    "explain_context": {
        "topic_or_signal": "Topic word or signal name, e.g. coolant.",
        "session_id": "Omit for the latest ride.",
    },
}

# argument name -> allowed values, published as a JSON-schema enum.
ENUMS: dict[str, dict[str, list[str]]] = {
    "get_data_card": {"topic": list(TOPIC_KEYS)},
}


def _strip(node: Any) -> Any:
    """Drop titles and collapse `anyOf [T, null]` and null defaults (optional = not required)."""
    if isinstance(node, list):
        return [_strip(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {k: _strip(v) for k, v in node.items() if k not in {"title"}}
    any_of = out.get("anyOf")
    if isinstance(any_of, list):
        rest = [x for x in any_of if x != {"type": "null"}]
        if len(rest) == 1 and isinstance(rest[0], dict):
            del out["anyOf"]
            out.update(rest[0])
    if out.get("default") is None:
        out.pop("default", None)
    return out


def apply_specs(server: MCPServer) -> None:
    """Replace the registered tools' descriptions and schemas with the small-model specs."""
    for tool in server._tool_manager.list_tools():  # noqa: SLF001 (no public mutation API)
        tool.description = DESCRIPTIONS[tool.name]
        schema = _strip(tool.parameters)
        for arg, text in ARGUMENTS.get(tool.name, {}).items():
            schema["properties"][arg]["description"] = text
        for arg, values in ENUMS.get(tool.name, {}).items():
            schema["properties"][arg]["enum"] = values
        tool.parameters = schema
        # cached_property: replace the published output schema without its title
        if tool.output_schema is not None:
            tool.__dict__["output_schema"] = _strip(tool.output_schema)
