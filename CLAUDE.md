# CLAUDE.md — vehicle-mcp

@.claude/PLATFORM-RULES.md

## What this repo is

An **independent open-source** Model Context Protocol server (Python). Exposes vehicle telemetry to LLMs as **read-only** tools. Runs on the Raspi 5, on the same machine as `vehicle-linux-node`, but is a separate repo. Purpose: so other DIY vehicle/motorcycle projects can use it too, and it serves as an org showcase.

## Tools — fixed list, adding a new tool requires user approval

`get_live_snapshot`, `get_recent_stats(window_s)`, `get_event_log(n)`, `get_anomaly_status`, `get_maintenance_status`, `query_ride_history(date_range)`

## Rules

1. **Read-only.** No write or actuator tools. In Phase 2, only its own newly added peripheral actuators (lights, heating) may be added, behind an operator approval gate. NEVER extended to engine/ECU control.
2. **The LLM does not do math:** tools return already-computed results, with units and explanations. Never return a raw series and let the LLM interpret it.
3. **No raw GPS returned.** Location is only returned as a summary ("rural", "12 km from home"). Any special tool requiring coordinates is kept explicitly separate.
4. **Vehicle-independent core:** the data source is an adapter interface (the Kuksa/VSS adapter is the default). Nothing specific to vehicle-platform leaks into the core. VSS paths are used, not CAN IDs.
5. TLS + authentication are mandatory in the externally exposed "home demo" mode. The default listen address is localhost.
6. README, docstrings, and code are in English (open source). The licensing decision is asked of the user before publishing.

## Dependencies

At runtime, only VSS (Kuksa gRPC) and the vehicle-server API (`query_ride_history`). Depends on `vehicle-defs` only optionally, for VSS overlay/schema reference.

## Build

`uv` + `ruff` + `pytest`, the official MCP Python SDK. Tests run against a fake data adapter.

## Context

ARCHITECTURE §7 · `../vehicle-defs/docs/hardware-architecture.md` §5b.7.
