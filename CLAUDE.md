# CLAUDE.md — vehicle-mcp

@.claude/PLATFORM-RULES.md

## What this repo is

An **independent open-source** Model Context Protocol server (Python). Exposes vehicle telemetry to LLMs as **read-only** tools. Runs on the Raspi 5, on the same machine as `vehicle-linux-node`, but is a separate repo. Purpose: so other DIY vehicle/motorcycle projects can use it too, and it serves as an org showcase.

## Tools — adding a tool requires user approval

- **History (v0.1, D-092)**, over a `HistorySource` (`src/vehicle_mcp/source.py`; vehicle-server's HTTP API is the default, `client.py`): `list_sessions`, `get_session_report`, `list_series`, `get_timeseries_slice` (at most 2000 points per series, D-076), `compute_signal_stats` (exact, from the server's `/stats`; with an older server it falls back to `/series`: min/max exact, mean/std/percentiles only when the window is not downsampled), `compare_sessions`, `query_ride_history(start_date, end_date)` (D-092 item 2).
- **Ride summaries (D-102, built in mcp#17; 8 tools in all; 10 with the fault tools below):** `get_ride_summary(session_id)` over the server's `GET /sessions/{id}/summary` (stored class-B values as served, never recomputed, `defs_mismatch` passed through); `query_ride_history` adds each listed session's class-B scalars from `GET /rides` (same bounds, joined by `session_id`; without `/rides` it answers as before and says so). No new cross-ride aggregate without approval.
- **Fault catalog lookup (D-084):** `list_fault_catalog(subsystem=None)` and `get_fault(fault_id)` over the server's `GET /faults` and `GET /faults/{id}`; displayable entries only (`include_unconfirmed` is never sent); lookup of the sourced catalog, never a diagnosis; `advise_stop` means tell the rider to stop safely and have it inspected; a 404 is answered as `available: false` with the reason.
- **Small-model tools (T43, D-127 item 2, D-128 item 5):** `get_data_card(topic, session_id?)` (ride numbers for a topic word, signals picked by substring from server-reported names) and `check_symptom(complaint, session_id?, locale?)` (catalog complaint phrases to entries plus the ride's values of their `needs`; never a diagnosis; D-127 addendum: the catalog is fetched with `include_unconfirmed=true` ONLY here, and unconfirmed matches are reduced to one `data_only` item with the measured values, never id, text or severity; refused -> displayable-only with a `reason`); `explain_context(topic_or_signal, session_id?)` (D-131 item 2: reading, ride conditions from a topic table in `explain.py`, history on up to 10 previous rides, `references` (empty until the server exposes defs reference values; never hard-coded here) and direction-matched catalog entries, unconfirmed ones only as `data_only`; never a diagnosis); 13 tools in all. Tool descriptions are at most 160 characters, argument descriptions at most 80, specs total under the budget in `tests/test_small_model.py`; every measured number is a labelled quantity `{name, unit, value, label}` (README). Edit specs in `src/vehicle_mcp/specs.py`.
- **Knowledge search (T60, D-132 item 2c):** `search_knowledge(query, locale?, k?)` over a local SQLite index (FTS5 BM25 + sqlite-vec, multilingual-e5-small on ONNX Runtime, RRF k=60) built by `vehicle-mcp-index build` (catalog, Markdown dirs, `--local-pdf` for the rider's own manual, flagged `user-local`, snippets at most 300 characters). Optional `rag` extra; missing extra, model or index -> `available: false` with a reason. Unconfirmed catalog entries are indexed only as complaint phrases + `needs` and shown as one `data_only` hit (D-127 addendum). Never commit a PDF, manual, index, model weights or embeddings (D-033; `tests/test_no_manuals_tracked.py`). 14 tools in all.
- **Planned (approved list, not built):** live tools once linux-node/Kuksa exists: `get_live_snapshot`, `get_recent_stats(window_s)`, `get_event_log(n)`, `get_anomaly_status`, `get_maintenance_status`.

## Rules

1. **Read-only.** No write or actuator tools; every source call is a read (the HTTP source issues GET only, tested). Phase 2 actuators are Q-027. NEVER extended to engine/ECU control.
2. **The LLM does not do math (invariant 7):** tools return computed results with units. A raw slice (`get_timeseries_slice`) is context and display only (D-076 item 2 (c)); every statistic, distance or comparison comes from a tool.
3. **Position (D-076, provisional, Q-035):** position may reach the LLM only through vehicle-mcp, from the user's own servers. vehicle-server does not serve it yet (D-076 item 3), so v0.1 hides position-looking series names. A position tool needs the user's approval. The one position rule is vendored from vehicle-defs v0.30.0 (`src/vehicle_mcp/_position_vendored.py`, T22); never edit it, re-vendor from a release and update the sha256 in `tests/test_position.py`.
4. **Vehicle-independent core:** tools see only source interfaces (`HistorySource`; a live Kuksa/VSS source later), never URLs, CAN IDs or vehicle-platform internals. Series names, kinds and units come from the source, never hand-written (invariant 2).
5. **Transport:** stdio only in v0.1. An externally exposed mode (streamable HTTP, an open D-077 item) needs TLS + authentication and listens on localhost by default. The bearer token never appears in a tool result, an error or a log.
6. README, docstrings and code in English (open source). Licence: MIT (D-036). Runtime dependencies: `mcp` (MIT) and `httpx` (BSD-3) only; a new one needs the user's approval and a licence check (D-091). The optional `rag` extra (T60: `sqlite-vec`, `onnxruntime`, `tokenizers`, `numpy`, `pypdf`) is the only exception and stays out of the core install.

## Dependencies

At runtime the vehicle-server read API (`GET /sessions`, `/sessions/{id}/report`, `/sessions/{id}/series`, `/sessions/{id}/stats`, and with D-102 `/sessions/{id}/summary`, `/rides`) with `VEHICLE_SERVER_URL` and `VEHICLE_API_TOKEN` from the environment; later Kuksa (gRPC) for the live tools. No vehicle-defs submodule in v0.1: names and units come from the server, which takes them from defs; revisit with the D-084 matcher.

## Build

`uv` + `ruff` + `pytest`, the official MCP Python SDK (`mcp` 2.x, `MCPServer`). Tests run against a fake source (`httpx.MockTransport`, an in-memory `HistorySource`), synthetic data only.

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
VEHICLE_API_TOKEN=... uv run vehicle-mcp   # stdio
```

## Context

ARCHITECTURE §7 · `../vehicle-defs/docs/hardware-architecture.md` §5b.7.
