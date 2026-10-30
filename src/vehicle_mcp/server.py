"""stdio entry point: expose the read-only tools over MCP."""

from __future__ import annotations

import sys

from mcp.server.mcpserver import MCPServer

from . import __version__
from .client import VehicleServerClient
from .config import ConfigError, load_settings
from .source import HistorySource
from .tools import VehicleTools

INSTRUCTIONS = (
    "Read-only access to recorded motorcycle sessions through vehicle-server. "
    "Every tool issues GET requests only; nothing is written, cleared or sent to the "
    "vehicle bus. Series names, kinds and units come from the server, and position "
    "series are not available (D-076 item 3 is not implemented). The server never "
    "returns the API token."
)


def build_server(source: HistorySource) -> MCPServer:
    server = MCPServer(
        name="vehicle-mcp",
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    tools = VehicleTools(source)
    for tool in (
        tools.list_sessions,
        tools.get_session_report,
        tools.list_series,
        tools.get_timeseries_slice,
        tools.compute_signal_stats,
        tools.compare_sessions,
    ):
        server.tool()(tool)
    return server


def main() -> None:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"vehicle-mcp: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    client = VehicleServerClient(settings)
    server = build_server(client)
    try:
        server.run(transport="stdio")
    finally:
        client.close()
