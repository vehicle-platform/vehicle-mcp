"""The position-name rule of vehicle-mcp (D-076, invariant 7).

The rule is vehicle-defs' ``vehicle_defs.position`` (T22, generated from
``privacy/position_rule.yaml``), vendored unchanged as ``_position_vendored.py`` because this
repo has no defs submodule. ``tests/test_position.py`` pins the vendored file to the v0.30.0
release by sha256 and runs the golden vectors; to update, vendor the new release file, its
vectors and the expected hash together.
"""

from __future__ import annotations

from vehicle_mcp._position_vendored import is_position_name

__all__ = ["is_position_name"]
