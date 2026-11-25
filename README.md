# vehicle-mcp

Part of [vehicle-platform](https://github.com/vehicle-platform), an SDV-style diagnostics, telemetry and rider-assistance platform for motorcycles (first vehicle: Honda CL250).

An **independent open-source** Model Context Protocol server (Python). Exposes vehicle telemetry to LLMs as **read-only** tools. Runs on the Raspi 5, on the same machine as `vehicle-linux-node`, but is a separate repo. Purpose: so other DIY vehicle/motorcycle projects can use it too, and it serves as an org showcase.

**Status:** v0.2.0 — a stdio MCP server over the read-only HTTP API of `vehicle-server`.

- Architecture and decisions: [vehicle-defs/docs](https://github.com/vehicle-platform/vehicle-defs/tree/main/docs) (`ARCHITECTURE.md`, `DECISIONS.md`)
- Series names, CAN IDs and DIDs come only from `vehicle-server` / [vehicle-defs](https://github.com/vehicle-platform/vehicle-defs); this repo hard-codes none
- Scope rules for contributors and Claude Code: [`CLAUDE.md`](CLAUDE.md)

## What it is

`vehicle-mcp` speaks MCP over stdio and wraps the read-only HTTP API of
[`vehicle-server`](https://github.com/vehicle-platform/vehicle-server): recorded sessions,
reports, time-series slices, computed statistics, session comparisons and stored class-B ride
summaries. It never talks to the motorcycle bus and never writes to the server.

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
| `VEHICLE_SERVER_URL` | no | `http://127.0.0.1:8000` | Base URL of `vehicle-server`. Plain `http://` is accepted only for localhost and private/loopback IP addresses; anything else needs `https://` (or `VEHICLE_MCP_ALLOW_INSECURE_HTTP=1` on a trusted network path) |

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
| `compute_signal_stats` | `session_id`, `signal`, `start_ms`, `end_ms` | Labelled quantities: `values` (min, mean, max, p05, p50, p95, std), `samples`, `duration` (s) |
| `compare_sessions` | `session_a`, `session_b`, `signal` | Labelled min, mean, max and median for both sessions plus `delta_b_minus_a` |
| `query_ride_history` | `start_date`, `end_date` (ISO 8601 date or timestamp) | Sessions inside the inclusive range sorted by `created_utc`, with `count`, `total_packet_count`, `mean_loss_percent`, a status histogram and each session's stored class-B `ride` scalars; sessions with a missing or unparseable `created_utc` are returned under `undated` |
| `get_ride_summary` | `session_id` | The stored class-B ride summary of one session: scalars (name, value, unit, status, flag) and binned signals (axes, cells), exactly as `vehicle-server` computed and stored them |
| `list_fault_catalog` | `subsystem` (optional) | The displayable fault catalog entries in compact form: `id`, `subsystem`, `root_cause`, `severity`, `advise_stop`, `detectable_now`, `active: false`, symptom summary; the result is `kind: catalog_reference`, `active: false`. Lookup only, never a diagnosis; unconfirmed entries are never requested |
| `get_fault` | `fault_id` | One displayable catalog entry in full, including `rider_text`, `action` and `sources`. A 404 (older server without `/faults`, unknown or unconfirmed entry) is answered as `available: false` with the reason |
| `get_data_card` | `topic`, `session_id` (optional) | The ride's numbers for a topic (`coolant`, `battery`, `idle`, `speed`, `rpm`, `throttle`, `fuel`, `stops`, `gears`, `ride`): per matching signal the labelled min, mean, max and last value, plus matching stored ride scalars (ride time, distance, ...), the ride id and its date. Latest ride when `session_id` is omitted. Signals are picked by substring from the names the server reports for that ride |
| `check_symptom` | `complaint`, `session_id` (optional), `locale` (default `en`) | Up to three catalog entries whose complaint phrases overlap the rider's words (English phrases, plus the locale's phrases for another `locale`): `id`, `subsystem`, `root_cause`, `severity`, `advise_stop`, `detectable_now`, `needs`, `rider_text`, `action`, `match_score`; for entries with `needs`, the labelled values of those signals in the chosen or latest ride and `moved_as_catalog` (did the ride's trend match the catalog symptom direction); for entries nothing can measure (`engine_mechanical`), `measurable: false` with a note. `diagnosis` is always `false`. Complaints also match unconfirmed catalog entries (D-127 addendum), but those return only one merged `data_only` item: the measured values and trend of their `needs` signals, with no id, cause, rider text, action, severity or phrase. To keep it quiet, `data_only` appears only when its best score is at least 0.8 x the best confirmed match (or there is none), uses the best unconfirmed entry plus one within 0.05 of it, orders signals by how many of those entries need them and shows at most four. If the server refuses `include_unconfirmed` for the token, only confirmed entries are matched and `reason` says so |

A date-only bound covers the whole UTC day; a full timestamp is an exact instant, and one without an
offset is taken as UTC. The result echoes the bounds it used as `start_utc` and `end_utc`.
`mean_loss_percent` is the unweighted mean of the per-session `loss_percent` values (`null` when
none is numeric). Sessions whose `created_utc` is missing or unparseable are listed under `undated`
whatever the range, and are excluded from the summary.

Every dated session also gets a `ride` object with the server's `values`, `flags` and `columns`
units from `GET /rides`; a session the server skipped gets `ride: null` with a reason, and the
server's `skipped` count is added. Undated sessions get no ride scalars (`/rides` selects by
`created_utc`), stated once as `undated_ride_reason`. Position-looking ride names are dropped and
listed once under the top-level `dropped_position_names`. An older server without `/rides`, or any
`/rides` error, keeps today's session output and adds `ride_scalars: "unavailable"` with the
reason instead of failing the query.

### Result shape for small models

The tool descriptions are at most 160 characters and the argument descriptions at most 80, so the
12 tool specs (name, description, input schema) stay at about 3.2k characters for a 4k-context
model, down from 6.3k for 10 tools; a test fails above 3500. Optional arguments are plain types
and not required, and `get_data_card`'s `topic` is an enum.

Every number a tool measures or computes is one **labelled quantity**, a flat object with the same
four keys everywhere (`compute_signal_stats`, `compare_sessions`, `get_data_card`, `check_symptom`):

```json
{"name": "ride_duration", "unit": "s", "value": 1115.83, "label": "ride duration"}
```

`name` is a stable snake_case id (`engine_speed_mean`, `engine_speed_mean_delta`), `label` a plain
phrase (`engine speed average`), `unit` the server-reported unit, and `value` is rounded at the
source (2 decimals, or 3 significant digits below 1). The model should quote a number only with
its label and unit. Stored ride scalars from `get_ride_summary` and `/rides` keep the server's own
`name`/`unit`/`value` entries unchanged. `structuredContent` is the same JSON.

Catalog results are references, not findings: `list_fault_catalog` returns `kind:
"catalog_reference"`, `active: false` (also on each entry) and a note that the faults are possible,
not detected; `check_symptom` returns `kind: "symptom_check"` and `diagnosis: false`. When the
catalog has no complaint phrases (a server before vehicle-defs v0.32.0) or nothing matches,
`check_symptom` returns `matched: []` with a `reason`.

**How `check_symptom` matches (deterministic, pure Python, no model).** The complaint and every
catalog phrase of the active locales (English plus the requested `locale`) are folded (case,
diacritics, Turkish dotted/dotless i), split into words and stripped of filler words from
`src/vehicle_mcp/locales/<locale>.json` (English is always applied; an unknown locale has no
extra list; add a language by adding a file). `sorun`/`sıkıntı`-style words ("problem") are
treated as filler because riders put them in the question, not in the symptom. Two words are
equivalent when they are equal, share a stem (common prefix of at least 3 characters and 60 % of
the shorter word, so suffixes do not matter) or are a one-character typo (both at least 5
characters, the shorter at most 7, same first character, at most one edit in the first 5
characters). Each phrase word weighs `log(1 + N/df)` over all phrases in the fetched catalog, so
rare words count more than "engine" or "noise". The score is the weighted share of a phrase's
words that the complaint covers (extra words in the complaint cost nothing), and a match needs at
least one covered word at or above the median weight of the phrase's own words. Entries scoring
at least 0.55 are returned, best three first. Before matching, a complaint word is also expanded with the one-way `synonyms` table of `locales/<locale>.json` (English always; e.g. "dying" also counts as "flat", "dead", "drained", "hot" as "overheating"), so riders' words reach the catalog's wording. The threshold was tuned on a synthetic table in
`tests/test_small_model.py` and checked against a live catalog.

`get_ride_summary` treats one session as one ride. The values are computed once by
`vehicle-server` from vehicle-defs definitions; the tool never recomputes, rebins, converts,
rounds or aggregates them. A `null` value carries the server's status or flag reason in the same
entry. Binned signals keep the server's axes and cells, and a stored `defs_mismatch` adds a
warning that the stored bin edges differ from the current defs. Position-looking names are
dropped and listed under `dropped_position_names`.

`compute_signal_stats` and `compare_sessions` ask `vehicle-server` for exact statistics
(`GET /sessions/{id}/stats`) and do not need the full series. With an older server that lacks
that route, they fall back to `GET /sessions/{id}/series`; if that window was downsampled,
`min` and `max` are still exact (min/max bucketing) while `mean`, `std` and the percentiles are
`null` with an `approximate_reason`; the server does not guess them from the reduced series.
Statistics always come from the server or this tool, never from the model.

The ride-summary tools require `vehicle-server` with server#46 (merged 2026-11-02). No release
tag contains that merge yet: `v0.6.0` and older have no `/sessions/{id}/summary` or `/rides`
route. A server without `/rides` is handled with the fallback described above; a server without
`/sessions/{id}/summary` makes `get_ride_summary` fail with the server's 404 error.

## Data sources

The tools read through `HistorySource` (`src/vehicle_mcp/source.py`): `sessions()`,
`report()`, `series()`, `stats()`, `ride_summary()` and `rides()` in the vehicle-server
response shapes. `stats()` returns the exact full-resolution count, mean, standard deviation,
min/max, p05/p50/p95 and first/last timestamps; the tools fall back to `series()` only when the
server has no `/stats` route. `ride_summary()` and `rides()` return the stored class-B summary
and ride table without changing the values. `VehicleServerClient` is the default
implementation; another project can plug in its own source without touching the tools.

## Guarantees

- **Untrusted text:** strings that originate from the vehicle or the phone (free-text meta fields, report reasons) are returned as `{"untrusted_text": "..."}` (truncated, control characters removed) or between `<<<UNTRUSTED ...>>>` fences in text reports; the server instructions tell the model they are data, never instructions. Result sizes are capped (sessions per call, at most 8 series and 8000 points per slice, 8 MiB per server answer).
- **Read-only:** only GET requests; no writes, no UDS, no CAN, no vehicle-bus access.
- **No invented signals:** names, kinds and units come from the server; an unknown name is a
  clear error, and nothing is hard-coded here.
- **No model math:** statistics are computed by this server. `std` is the population standard
  deviation and the percentiles use linear interpolation between the closest ranks.
- **No position:** `vehicle-server` does not expose lat/lon/height yet (D-076 item 3), so there
  is no position tool and position-looking series are dropped or rejected (latitude/longitude
  anywhere in the name; height/altitude only as the last word, so `derived.altitude_change`, a
  barometric difference since key-on, stays).
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
