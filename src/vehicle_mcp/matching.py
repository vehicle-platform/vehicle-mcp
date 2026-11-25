"""Pure helpers for `get_data_card` and `check_symptom`: text matching and trends.

Nothing here knows a signal name, scale or CAN id (invariant 2). Topic matching uses
substrings of names the server reports; complaint phrases come from the fault catalog.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections.abc import Iterable, Sequence

# Canonical topic -> (words that select it, substrings of server-reported names to include,
# substrings to exclude). Topic words are English; locale keyword tables are the caller's job.
TOPICS: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "coolant": (
        ("coolant", "temperature", "temp", "overheating", "overheat", "hot", "heat"),
        ("coolant", "engine_temp", "water_temp"),
        (),
    ),
    "battery": (
        ("battery", "charging", "charge", "voltage", "alternator"),
        ("battery", "charg", "voltage"),
        (),
    ),
    "idle": (("idle", "idling"), ("idle", "engine_speed", "rpm"), ()),
    "speed": (
        ("speed", "velocity", "fast"),
        ("vehicle_speed", "wheel_speed", "speed"),
        ("engine_speed",),
    ),
    "rpm": (("rpm", "revs", "engine speed", "engine_speed"), ("engine_speed", "rpm"), ()),
    "throttle": (("throttle", "acceleration", "accelerator"), ("throttle", "accel"), ()),
    "fuel": (("fuel", "consumption", "economy", "mileage"), ("fuel",), ()),
    "stops": (("stops", "stop", "stopped", "standstill"), ("stop", "standstill"), ()),
    "gears": (("gears", "gear", "shifting", "shift"), ("gear",), ()),
    "ride": (
        ("ride", "time", "duration", "distance", "trip", "how long", "how far"),
        ("ride_time", "ride_distance", "distance", "duration"),
        (),
    ),
}
TOPIC_KEYS = tuple(TOPICS)


def canonical_topic(topic: str) -> str | None:
    """Map a topic word or phrase to a canonical topic key, or None."""
    text = normalise(topic)
    if not text:
        return None
    for key, (words, _, _) in TOPICS.items():
        if text == key or text in {normalise(w) for w in words}:
            return key
    tokens = set(text.split())
    for key, (words, _, _) in TOPICS.items():
        if any(normalise(w) in tokens for w in words if " " not in w):
            return key
    return None


def name_matches(topic: str, name: str) -> bool:
    """True when a server-reported name belongs to the canonical topic."""
    _, include, exclude = TOPICS[topic]
    low = name.lower()
    return any(part in low for part in include) and not any(part in low for part in exclude)


_STOPWORDS = frozenset(
    "a an the is are was were be it its my me i we you your of on in at to for from with "
    "and or but that this there something anything wrong problem normal bike motorcycle "
    "motor does do did has have had can could when while sometimes".split()
)
_TURKISH_FOLD = str.maketrans({"\u0131": "i", "\u0130": "i", "\u00df": "ss"})


def normalise(text: str) -> str:
    """Casefold, fold Turkish dotted/dotless i and other diacritics, drop punctuation."""
    folded = unicodedata.normalize("NFKD", text.translate(_TURKISH_FOLD).casefold())
    stripped = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z]+", " ", stripped).strip()


def tokens(text: str) -> list[str]:
    return [t for t in normalise(text).split() if t not in _STOPWORDS]


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 4 and long_.startswith(short)


def phrase_score(complaint: Sequence[str], phrase: Sequence[str]) -> float:
    """Dice overlap of two token lists (prefix-tolerant); 0.0 when nothing is shared."""
    if not complaint or not phrase:
        return 0.0
    shared = sum(1 for p in set(phrase) if any(_same(p, c) for c in set(complaint)))
    return 2 * shared / (len(set(phrase)) + len(set(complaint)))


def best_phrase(complaint: str, phrases: Iterable[str]) -> tuple[float, str | None]:
    """The best Dice score of a complaint against catalog phrases, with the phrase."""
    c_tokens = tokens(complaint)
    best, best_text = 0.0, None
    for phrase in phrases:
        score = phrase_score(c_tokens, tokens(phrase))
        if score > best:
            best, best_text = score, phrase
    return best, best_text


def trend(values: Sequence[float], threshold: float = 0.1) -> str:
    """rising / falling / flat / unstable over a ride, from first vs last quarter means.

    `unstable` means the sample-to-sample jumps are large against the value range.
    """
    clean = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if len(clean) < 4:
        return "flat"
    span = max(clean) - min(clean)
    if span == 0:
        return "flat"
    jumps = statistics.fmean(abs(b - a) for a, b in zip(clean, clean[1:], strict=False))
    if jumps / span > 0.25:
        return "unstable"
    q = max(1, len(clean) // 4)
    change = (statistics.fmean(clean[-q:]) - statistics.fmean(clean[:q])) / span
    if change > threshold:
        return "rising"
    if change < -threshold:
        return "falling"
    return "flat"


def moved_as(direction: object, observed: str | None) -> bool | None:
    """Whether an observed trend matches a catalog symptom direction (up/down/unstable/absent)."""
    if observed is None:
        return direction == "absent" or None
    table = {"up": "rising", "down": "falling", "unstable": "unstable"}
    if direction in table:
        return observed == table[str(direction)]
    if direction == "absent":
        return False
    return None
