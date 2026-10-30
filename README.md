# vehicle-mcp

Part of [vehicle-platform](https://github.com/vehicle-platform), an SDV-style diagnostics, telemetry and rider-assistance platform for motorcycles (first vehicle: Honda CL250).

An **independent open-source** Model Context Protocol server (Python). Exposes vehicle telemetry to LLMs as **read-only** tools. Runs on the Raspi 5, on the same machine as `vehicle-linux-node`, but is a separate repo. Purpose: so other DIY vehicle/motorcycle projects can use it too, and it serves as an org showcase.

**Status:** v0.1.0 — a stdio MCP server over the read-only HTTP API of `vehicle-server`.

- Architecture and decisions: [vehicle-defs/docs](https://github.com/vehicle-platform/vehicle-defs/tree/main/docs) (`ARCHITECTURE.md`, `DECISIONS.md`)
- Series names, CAN IDs and DIDs come only from `vehicle-server` / [vehicle-defs](https://github.com/vehicle-platform/vehicle-defs); this repo hard-codes none
- Scope rules for contributors and Claude Code: [`CLAUDE.md`](CLAUDE.md)

## What it is

`vehicle-mcp` speaks MCP over stdio and wraps the read-only HTTP API of
[`vehicle-server`](https://github.com/vehicle-platform/vehicle-server): recorded sessions,
reports, time-series slices, computed statistics and session comparisons. It never talks to
the motorcycle bus and never writes to the server.

## Install

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/vehicle-platform/vehicle-mcp.git
cd vehicle-mcp
uv sync
```

## Configuration

Configuration is environment-only. The token is never logged and never returned by a tool.

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `VEHICLE_API_TOKEN` | yes | — | Bearer token of `vehicle-server`; startup fails without it |
| `VEHICLE_SERVER_URL` | no | `http://127.0.0.1:8000` | Base URL of `vehicle-server` |

## Run

```sh
export VEHICLE_API_TOKEN=...
uv run vehicle-mcp
```

The server uses stdio transport. Example MCP client configuration (Claude Desktop):

```json
{
  "mcpServers": {
    "vehicle": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/vehicle-mcp", "vehicle-mcp"],
      "env": {
        "VEHICLE_SERVER_URL": "http://127.0.0.1:8000",
        "VEHICLE_API_TOKEN": "<your-token>"
      }
    }
  }
}
```

## Tools

Every tool is read-only and issues GET requests only.

| Tool | Arguments | Returns |
|---|---|---|
| `list_sessions` | — | Session index metadata (`session_id`, timestamps, packet count, loss, status, feature flags, defs version) |
| `get_session_report` | `session_id`, `format` (`json` or `text`) | The session report as structured JSON or pre-rendered text |
| `list_series` | `session_id` | Series names, kinds, units and point counts; no samples |
| `get_timeseries_slice` | `session_id`, `signals`, `start_ms`, `end_ms`, `max_points` (default 500) | Samples (`t_ms`, `v`), `points_total` and `downsampled`; capped at 2000 points per series |
| `compute_signal_stats` | `session_id`, `signal`, `start_ms`, `end_ms` | `count`, `min`, `max`, `mean`, `std`, `p05`, `p50`, `p95`, `duration_ms`, `unit` |
| `compare_sessions` | `session_a`, `session_b`, `signal` | Stats for both sessions plus the `b - a` delta for each non-null field |

If `vehicle-server` downsamples a window, `min` and `max` are still exact (min/max bucketing)
while `mean`, `std` and the percentiles are `null` with an `approximate_reason`; the server
does not guess them from the reduced series.

## Guarantees

- **Read-only:** only GET requests; no writes, no UDS, no CAN, no vehicle-bus access.
- **No invented signals:** names, kinds and units come from the server; an unknown name is a
  clear error, and nothing is hard-coded here.
- **No model math:** statistics are computed by this server. `std` is the population standard
  deviation and the percentiles use linear interpolation between the closest ranks.
- **No position:** `vehicle-server` does not expose lat/lon/height yet (D-076 item 3), so there
  is no position tool and position-looking series are dropped or rejected.
- **Token safety:** the token comes from the environment, never appears in tool results or
  errors, and failures are explanatory messages without stack traces.
- **Privacy:** no real ride data in this repo; the tests use synthetic sessions only.

## Development

```sh
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## License

MIT, see [LICENSE](LICENSE) (D-036).
