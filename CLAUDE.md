# CLAUDE.md — vehicle-mcp

@.claude/PLATFORM-RULES.md

## What this repo is

An **independent open-source** Model Context Protocol server (Python). Exposes vehicle telemetry to LLMs as **read-only** tools. Runs on the Raspi 5, on the same machine as `vehicle-linux-node`, but is a separate repo. Purpose: so other DIY vehicle/motorcycle projects can use it too, and it serves as an org showcase.

## Tools — adding a tool requires user approval

- **History (v0.1, D-092)**, over a `HistorySource` (`src/vehicle_mcp/source.py`; vehicle-server's HTTP API is the default, `client.py`): `list_sessions`, `get_session_report`, `list_series`, `get_timeseries_slice` (at most 2000 points per series, D-076), `compute_signal_stats` (min/max exact; mean/std/percentiles only when the window is not downsampled), `compare_sessions`, `query_ride_history(start_date, end_date)` (D-092 item 2).
- **Planned (approved list, not built):** live tools once linux-node/Kuksa exists: `get_live_snapshot`, `get_recent_stats(window_s)`, `get_event_log(n)`, `get_anomaly_status`, `get_maintenance_status`; the D-084 fault-catalog matcher.

## Rules

1. **Read-only.** No write or actuator tools; every source call is a read (the HTTP source issues GET only, tested). Phase 2 actuators are Q-027. NEVER extended to engine/ECU control.
2. **The LLM does not do math (invariant 7):** tools return computed results with units. A raw slice (`get_timeseries_slice`) is context and display only (D-076 item 2 (c)); every statistic, distance or comparison comes from a tool.
3. **Position (D-076, provisional, Q-035):** position may reach the LLM only through vehicle-mcp, from the user's own servers. vehicle-server does not serve it yet (D-076 item 3), so v0.1 hides position-looking series names. A position tool needs the user's approval.
4. **Vehicle-independent core:** tools see only source interfaces (`HistorySource`; a live Kuksa/VSS source later), never URLs, CAN IDs or vehicle-platform internals. Series names, kinds and units come from the source, never hand-written (invariant 2).
5. **Transport:** stdio only in v0.1. An externally exposed mode (streamable HTTP, an open D-077 item) needs TLS + authentication and listens on localhost by default. The bearer token never appears in a tool result, an error or a log.
6. README, docstrings and code in English (open source). Licence: MIT (D-036). Runtime dependencies: `mcp` (MIT) and `httpx` (BSD-3) only; a new one needs the user's approval and a licence check (D-091).

## Dependencies

At runtime the vehicle-server read API (`GET /sessions`, `/sessions/{id}/report`, `/sessions/{id}/series`) with `VEHICLE_SERVER_URL` and `VEHICLE_API_TOKEN` from the environment; later Kuksa (gRPC) for the live tools. No vehicle-defs submodule in v0.1: names and units come from the server, which takes them from defs; revisit with the D-084 matcher.

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
