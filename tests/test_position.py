"""Position-name rule: the vendored vehicle-defs rule, its release pin and golden vectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vehicle_mcp import _position_vendored
from vehicle_mcp.position import is_position_name

VENDORED_TAG = "v0.30.0"
# sha256 of gen/python/vehicle_defs/position.py and privacy/position_vectors.json at the tag.
VENDORED_SHA256 = "0f637ee73381cac609b11a854ecf54b3812d879a4a88aba493ccde77e6df5ffe"
VECTORS_SHA256 = "317ccc9e08a76263c0ab8c1fbc6e73b2e15a9a94c9156eba95537b0588f7ee7d"
VECTORS = Path(__file__).parent / "data" / "position_vectors.json"

POSITION = [
    "sensor.gps_lat_deg",
    "gps_latitude_e7",
    "sensor.gps_longitude",
    "gps_lon",
    "gps_lng",
    "sensor.gps_altitude_m",
    "gps_height_above_ellipsoid",
    "gps_ecef_x",
    "gps_hMSL",
    "gps_alt_m",
    "gps_height_mm",
    "gpsLat",
    "GpsLongitude",
    "sensor.gps_altitude",
]
NOT_POSITION = [
    "altitude_change",
    "derived.altitude_change",
    "throttle_pos_pct",
    "did.ENGINE_SPEED",
    "derived.LEAN_ANGLE",
    "sensor.gps_ground_speed",
    "sensor.gps_heading",
    "position_sensor_ok",
]


@pytest.mark.parametrize("name", POSITION)
def test_position_names_are_refused(name: str) -> None:
    assert is_position_name(name)


@pytest.mark.parametrize("name", NOT_POSITION)
def test_non_position_names_pass(name: str) -> None:
    assert not is_position_name(name)


def test_vendored_rule_is_byte_identical_to_the_release() -> None:
    data = Path(_position_vendored.__file__).read_bytes()
    assert hashlib.sha256(data).hexdigest() == VENDORED_SHA256, (
        f"_position_vendored.py differs from vehicle-defs {VENDORED_TAG} "
        "gen/python/vehicle_defs/position.py; re-vendor it, never edit it"
    )
    assert hashlib.sha256(VECTORS.read_bytes()).hexdigest() == VECTORS_SHA256


def test_the_reexport_is_the_vendored_rule() -> None:
    assert is_position_name is _position_vendored.is_position_name


@pytest.mark.position_vectors
def test_defs_position_vectors() -> None:
    vectors = json.loads(VECTORS.read_text())
    assert vectors["is_position_key"] and vectors["drop_position_columns"]
    for case in vectors["is_position_key"]:
        assert is_position_name(case["name"]) is case["expected"], case["name"]
    for case in vectors["drop_position_columns"]:
        kept = [c for c in case["columns"] if not is_position_name(c)]
        assert kept == case["expected"]
    for case in vectors["redact"]:
        assert _position_vendored.redact(case["input"]) == case["expected"]
