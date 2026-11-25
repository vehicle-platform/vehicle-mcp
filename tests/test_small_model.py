"""Small-model tool specs, labelled results, get_data_card and check_symptom (T43)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from tests.support import HOT_SESSIONS, TOKEN, FakeVehicleServer
from vehicle_mcp import matching
from vehicle_mcp.client import VehicleServerClient
from vehicle_mcp.config import Settings
from vehicle_mcp.server import build_server
from vehicle_mcp.specs import MAX_ARGUMENT_CHARS, MAX_DESCRIPTION_CHARS
from vehicle_mcp.tools import VehicleTools

# Measured name + description + input schema of all 12 tools: 3188 characters. The budget is
# that plus 10 %. The 10 tools before this change measured 6254 characters.
SPEC_BUDGET_CHARS = 3500


def _client(server: FakeVehicleServer) -> VehicleServerClient:
    return VehicleServerClient(
        Settings(server_url="http://server.test", api_token=TOKEN),
        transport=httpx.MockTransport(server.handler),
    )


@pytest.fixture
def hot() -> VehicleTools:
    server = FakeVehicleServer(sessions=HOT_SESSIONS)
    client = _client(server)
    yield VehicleTools(client)
    client.close()


def _specs() -> list[Any]:
    client = _client(FakeVehicleServer())
    try:
        return asyncio.run(build_server(client).list_tools())
    finally:
        client.close()


def _spec_chars(tool: Any) -> int:
    schema = json.dumps(tool.input_schema, separators=(",", ":"))
    return len(tool.name) + len(tool.description or "") + len(schema)


def test_descriptions_and_arguments_are_short() -> None:
    for tool in _specs():
        assert 0 < len(tool.description) <= MAX_DESCRIPTION_CHARS, tool.name
        for arg, prop in tool.input_schema["properties"].items():
            assert len(prop.get("description", "")) <= MAX_ARGUMENT_CHARS, (tool.name, arg)
            assert "title" not in prop


def test_total_spec_size_stays_in_budget() -> None:
    total = sum(_spec_chars(tool) for tool in _specs())
    assert total <= SPEC_BUDGET_CHARS, f"tool specs are {total} characters"


def test_optional_arguments_are_plain_types_and_not_required() -> None:
    by_name = {tool.name: tool for tool in _specs()}
    schema = by_name["check_symptom"].input_schema
    assert schema["required"] == ["complaint"]
    assert schema["properties"]["session_id"]["type"] == "string"
    assert "anyOf" not in json.dumps(schema)
    topics = by_name["get_data_card"].input_schema["properties"]["topic"]["enum"]
    assert {"coolant", "battery", "idle", "speed", "rpm", "fuel"} <= set(topics)


def test_trimmed_schema_still_validates_through_the_server() -> None:
    client = _client(FakeVehicleServer(sessions=HOT_SESSIONS))
    try:
        server = build_server(client)
        result = asyncio.run(server.call_tool("get_data_card", {"topic": "coolant"}))
        omitted = asyncio.run(server.call_tool("check_symptom", {"complaint": "valves noise"}))
    finally:
        client.close()
    assert result.is_error is False
    assert result.structured_content["topic"] == "coolant"
    assert omitted.is_error is False


def test_catalog_results_say_they_are_not_active_faults(tools: VehicleTools) -> None:
    result = tools.list_fault_catalog()
    assert result["kind"] == "catalog_reference"
    assert result["active"] is False
    assert result["note"].startswith("These are possible faults")


def test_normalise_folds_turkish_i_and_punctuation() -> None:
    # dotted capital I (U+0130), dotless i (U+0131), s-cedilla (U+015F), g-breve (U+011F)
    assert matching.normalise("DIM, \u0130 \u0131 \u015f \u011f!") == "dim i i s g"
    assert matching.normalise("I\u0307ce") == "ice"


def test_check_symptom_matches_a_measurable_entry_with_ride_data(hot: VehicleTools) -> None:
    result = hot.check_symptom("my engine is overheating!")
    assert result["diagnosis"] is False
    assert result["kind"] == "symptom_check"
    assert result["session_id"] == "hot"
    assert result["ride_date"] == "2026-10-05T10:00:00Z"
    first = result["matched"][0]
    assert first["id"] == "F-TEST-01"
    assert first["active"] is False
    assert first["advise_stop"] is True
    assert first["detectable_now"] is True
    assert first["needs"] == ["did.COOLANT_TEMP"]
    assert first["measurable"] is True
    assert first["rider_text"].startswith("Stop safely")
    assert first["action"] == "inspect cooling system"
    card = first["signals"][0]
    values = {q["name"]: q for q in card["values"]}
    assert values["coolant_temp_max"]["value"] == 108.12
    assert values["coolant_temp_max"]["unit"] == "degC"
    assert values["coolant_temp_last"]["value"] == 108.12
    assert card["trend_over_ride"] == "rising"
    assert card["catalog_direction"] == "up"
    assert card["moved_as_catalog"] is True
    assert "not a diagnosis" in result["note"].lower() or "not confirmed" in result["note"]


def test_check_symptom_reports_when_the_ride_did_not_move_as_the_catalog(
    hot: VehicleTools,
) -> None:
    server = FakeVehicleServer(sessions=HOT_SESSIONS)
    server.fault_entries[0]["needs"] = ["did.BATTERY_VOLTAGE"]
    server.fault_entries[0]["symptoms"] = [{"signal": "did.BATTERY_VOLTAGE", "direction": "down"}]
    client = _client(server)
    try:
        card = VehicleTools(client).check_symptom("engine overheating")["matched"][0]["signals"][0]
    finally:
        client.close()
    assert card["trend_over_ride"] == "flat"
    assert card["moved_as_catalog"] is False


def test_check_symptom_mechanical_entry_is_not_measurable(hot: VehicleTools) -> None:
    result = hot.check_symptom("my valves make a noise, is something wrong?")
    first = result["matched"][0]
    assert first["id"] == "F-TEST-ENG"
    assert first["subsystem"] == "engine_mechanical"
    assert first["measurable"] is False
    assert first["detectable_now"] is False
    assert "No sensor" in first["measurement_note"]
    assert first["action"].startswith("check the oil level")
    assert "signals" not in first
    assert result["session_id"] is None  # nothing needed a ride, so none was fetched


def test_check_symptom_uses_locale_phrases_only_for_that_locale(hot: VehicleTools) -> None:
    other = hot.check_symptom("Valvu zonk!", locale="zz")
    assert other["matched"][0]["id"] == "F-TEST-ENG"
    english_only = hot.check_symptom("Valvu zonk!", locale="en")
    assert english_only["matched"] == []
    assert "no catalog entry matches" in english_only["reason"]
    assert hot.check_symptom("MOTU HOTTA", locale="zz-ZZ")["matched"][0]["id"] == "F-TEST-01"


def test_check_symptom_returns_at_most_three_ordered_matches() -> None:
    server = FakeVehicleServer(sessions=HOT_SESSIONS)
    for i in range(5):
        extra = {**server.fault_entries[1], "id": f"F-X{i}", "complaints": [f"noise number {i}"]}
        server.fault_entries.append(extra)
    client = _client(server)
    try:
        matched = VehicleTools(client).check_symptom("noise number 0 1 2 3 4")["matched"]
    finally:
        client.close()
    assert len(matched) == 3
    scores = [m["match_score"]["value"] for m in matched]
    assert scores == sorted(scores, reverse=True)


def test_check_symptom_without_complaint_phrases_says_why(hot: VehicleTools) -> None:
    server = FakeVehicleServer(sessions=HOT_SESSIONS)
    for entry in server.fault_entries:
        entry.pop("complaints")
        entry.pop("complaints_i18n")
    client = _client(server)
    try:
        result = VehicleTools(client).check_symptom("engine overheating")
    finally:
        client.close()
    assert result["matched"] == []
    assert "no complaint phrases" in result["reason"]
    assert "v0.32.0" in result["reason"]


def test_check_symptom_old_server_and_empty_complaint(hot: VehicleTools) -> None:
    client = _client(FakeVehicleServer(supports_faults=False))
    try:
        result = VehicleTools(client).check_symptom("noise")
    finally:
        client.close()
    assert result["available"] is False
    assert result["matched"] == []
    assert "older server" in result["reason"]
    with pytest.raises(ToolError, match="must not be empty"):
        hot.check_symptom("   ")


def test_check_symptom_never_asks_for_unconfirmed_and_only_reads() -> None:
    server = FakeVehicleServer(sessions=HOT_SESSIONS)
    client = _client(server)
    try:
        VehicleTools(client).check_symptom("engine overheating")
    finally:
        client.close()
    assert all(r.method == "GET" for r in server.requests)
    faults = [r for r in server.requests if r.url.path == "/faults"]
    assert faults
    assert all("include_unconfirmed" not in str(r.url) for r in faults)


def test_data_card_for_coolant_uses_the_latest_ride(hot: VehicleTools) -> None:
    card = hot.get_data_card("overheating")
    assert card["available"] is True
    assert card["topic"] == "coolant"
    assert card["session_id"] == "hot"
    assert card["ride_date"] == "2026-10-05T10:00:00Z"
    assert [s["signal"] for s in card["signals"]] == ["did.COOLANT_TEMP"]
    values = {q["name"]: q["value"] for q in card["signals"][0]["values"]}
    assert values == {
        "coolant_temp_min": 80,
        "coolant_temp_mean": 93.02,
        "coolant_temp_max": 108.12,
        "coolant_temp_last": 108.12,
    }
    assert all(q["unit"] == "degC" for q in card["signals"][0]["values"])


def test_data_card_topics_pick_signals_by_server_names(hot: VehicleTools) -> None:
    assert [s["signal"] for s in hot.get_data_card("battery")["signals"]] == ["did.BATTERY_VOLTAGE"]
    assert [s["signal"] for s in hot.get_data_card("rpm")["signals"]] == ["did.ENGINE_SPEED"]
    assert [s["signal"] for s in hot.get_data_card("engine speed")["signals"]] == [
        "did.ENGINE_SPEED"
    ]
    # engine speed is not "speed"; the ride has no road-speed signal
    speed = hot.get_data_card("speed")
    assert speed["signals"] == []
    assert "no recorded signal for speed" in speed["reason"]
    assert "coolant temp" in speed["available_signals"]


def test_data_card_ride_topic_uses_stored_summary_scalars(hot: VehicleTools) -> None:
    card = hot.get_data_card("distance", session_id="hot")
    assert card["topic"] == "ride"
    values = {q["name"]: q for q in card["ride_values"]}
    assert values["ride_distance"] == {
        "name": "ride_distance",
        "unit": "km",
        "value": 42.5,
        "label": "ride distance",
    }
    assert values["ride_time"]["value"] == 3601.46
    assert hot.get_data_card("stops", session_id="hot")["ride_values"][0]["value"] == 3


def test_data_card_unknown_topic_and_no_rides(hot: VehicleTools) -> None:
    unknown = hot.get_data_card("astrology")
    assert unknown["available"] is False
    assert "coolant" in unknown["topics"]
    client = _client(FakeVehicleServer(sessions=[]))
    try:
        empty = VehicleTools(client).get_data_card("coolant")
    finally:
        client.close()
    assert empty["available"] is False


def test_data_card_hides_position_and_stays_read_only() -> None:
    server = FakeVehicleServer(sessions=HOT_SESSIONS)
    client = _client(server)
    try:
        card = VehicleTools(client).get_data_card("coolant", session_id="hot")
    finally:
        client.close()
    assert "gps" not in json.dumps(card).lower()
    assert server.requests
    assert all(r.method == "GET" for r in server.requests)
    assert TOKEN not in json.dumps(card)


def test_trend_and_direction_helpers() -> None:
    assert matching.trend([1, 2, 3]) == "flat"  # too few points to call a trend
    assert matching.trend([5.0] * 8) == "flat"
    assert matching.trend([1, 2, 3, 4, 5, 6, 7, 8]) == "rising"
    assert matching.trend([8, 7, 6, 5, 4, 3, 2, 1]) == "falling"
    assert matching.trend([1, 9, 1, 9, 1, 9, 1, 9]) == "unstable"
    assert matching.moved_as("up", "rising") is True
    assert matching.moved_as("down", "rising") is False
    assert matching.moved_as("unstable", "unstable") is True
    assert matching.moved_as("absent", None) is True
    assert matching.moved_as("absent", "flat") is False
    assert matching.moved_as("sideways", "flat") is None
    assert matching.moved_as("up", None) is None


def test_topic_words_resolve_to_one_canonical_topic() -> None:
    assert matching.canonical_topic("Coolant") == "coolant"
    assert matching.canonical_topic("how hot does it get") == "coolant"
    assert matching.canonical_topic("engine speed") == "rpm"
    assert matching.canonical_topic("how long was the trip") == "ride"
    assert matching.canonical_topic("???") is None
    assert matching.canonical_topic("") is None


def test_data_card_without_a_ride_summary_still_returns_the_signals() -> None:
    server = FakeVehicleServer(sessions=HOT_SESSIONS)
    server.summaries.pop("hot")
    client = _client(server)
    try:
        card = VehicleTools(client).get_data_card("coolant", session_id="hot")
    finally:
        client.close()
    assert card["ride_values"] == []
    assert card["signals"][0]["signal"] == "did.COOLANT_TEMP"


def test_data_card_for_a_named_ride_not_in_the_index_has_no_date(hot: VehicleTools) -> None:
    card = hot.get_data_card("battery", session_id="s1")
    assert card["session_id"] == "s1"
    assert card["signals"] == []  # s1 has no battery series
    assert "no recorded signal" in card["reason"]
    assert hot.get_data_card("coolant", session_id="hot")["ride_date"] == "2026-10-05T10:00:00Z"
    assert hot.get_data_card("rpm", session_id="s2")["ride_date"] is None


# --- multilingual matcher (stems, typos, stopwords, IDF weighting) -------------------------
# Turkish-like morphology is simulated with English stems plus synthetic suffixes, so the
# tests stay English only (D-125).

SYNTHETIC_PHRASES = {
    "valve": ["valves xyz noise", "valvesxyz tick"],
    "cool": ["engine overheating", "coolant boiling over"],
    "chain": ["chain xyz noise", "chain is loose"],
    "bat": ["battery goes flat", "battery will not hold charge"],
}


def _index(locale: str = "xx") -> matching.PhraseIndex:
    return matching.PhraseIndex(
        [p for ps in SYNTHETIC_PHRASES.values() for p in ps], ("en", locale)
    )


def _best(complaint: str) -> tuple[str, float]:
    index = _index()
    scores = {
        key: matching.best_phrase(complaint, phrases, index)[0]
        for key, phrases in SYNTHETIC_PHRASES.items()
    }
    key = max(scores, key=lambda k: scores[k])
    return key, scores[key]


def test_same_stems_typos_and_negatives() -> None:
    assert matching.same("noise", "noisexyz")  # stem plus suffix
    assert matching.same("sesi", "ses")  # 3 shared characters, 100 % of the shorter token
    assert matching.same("valve", "valvesxyzdan")
    assert matching.same("subap", "supap")  # one substitution
    assert matching.same("supap", "subaplarimdan")  # typo on the first five characters
    assert matching.same("valve", "valv")  # deletion
    assert not matching.same("noise", "number")
    assert not matching.same("light", "right")  # different first character
    assert not matching.same("overheating", "overfilled")  # long words sharing a start
    assert not matching.same("ab", "abc")  # prefix below 3 characters
    assert not matching.same("valves", "valxyz")  # prefix below 60 % of the shorter


def test_stopwords_come_from_locale_files() -> None:
    assert "the" in matching.stopwords("en")
    assert matching.stopwords("en-GB") == matching.stopwords("engb")  # letters only
    assert matching.stopwords("../../etc") == frozenset()
    assert matching.stopwords("zz") == frozenset()  # unknown locale: no filler words
    assert matching.stopwords("") == frozenset()
    turkish = matching.stopwords("tr")
    assert {"bir", "mi", "cok", "sikinti", "sorun"} <= turkish  # folded like the text
    assert matching.tokens("Is there a problem?", ("tr",)) == []
    assert matching.tokens("bir sorun var mi valve", ("tr",)) == ["valve"]
    assert matching.tokens("bir sorun var mi valve", ()) == ["bir", "sorun", "var", "mi", "valve"]


def test_idf_matcher_on_synthetic_morphology() -> None:
    assert _best("valvesxyzdan make noise, is something wrong?")[0] == "valve"
    key, score = _best("my engine is overheating a lot, what should I do?")
    assert key == "cool"
    assert score >= 0.55
    assert _best("chains xyz noise")[0] == "chain"


def test_long_complaints_are_not_punished_and_common_words_are_not_enough() -> None:
    long_one = "valves xyz noise " + "while riding on the highway every morning " * 3
    assert _best(long_one)[1] == 1.0
    assert _best("noise")[1] < 0.55  # a word shared by several phrases
    assert _best("engine")[1] < 0.55
    assert _best("the weather forecast")[1] == 0.0
    assert matching.phrase_score([], ["valve"], _index()) == 0.0
    assert matching.phrase_score(["valve"], [], _index()) == 0.0


def test_rare_words_weigh_more_than_frequent_ones() -> None:
    index = _index()
    assert index.weight("tick") > index.weight("noise")
    assert index.weight("unseen") > 0
    assert matching.PhraseIndex([]).weights == {}


def test_check_symptom_matches_inflected_locale_words(hot: VehicleTools) -> None:
    result = hot.check_symptom("valvuxyzdan zonk geliyor, is there a problem?", locale="zz")
    assert result["matched"][0]["id"] == "F-TEST-ENG"
