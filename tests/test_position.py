"""Position-name rule table. Mirror of vehicle-server tests (position.py); keep both in sync."""

from __future__ import annotations

import pytest

from vehicle_mcp.position import is_position_name

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
