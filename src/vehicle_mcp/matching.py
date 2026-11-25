"""Pure helpers for `get_data_card` and `check_symptom`: text matching and trends.

Nothing here knows a signal name, scale or CAN id (invariant 2). Topic matching uses
substrings of names the server reports; complaint phrases come from the fault catalog.
"""

from __future__ import annotations

import functools
import json
import math
import re
import statistics
import unicodedata
from collections.abc import Iterable, Sequence
from importlib import resources

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


_TURKISH_FOLD = str.maketrans({"\u0131": "i", "\u0130": "i", "\u00df": "ss"})

STEM_MIN_PREFIX = 3  # common prefix of two equivalent tokens: at least this many characters ...
STEM_MIN_RATIO = 0.6  # ... and at least this share of the shorter token
TYPO_MIN_LENGTH = 5  # typo tolerance only for tokens of at least this length ...
TYPO_MAX_SHORT = 7  # ... and when the shorter token is at most this long (a stem, not a long word)
TYPO_WINDOW = 5  # edit distance is measured on this many leading characters


def normalise(text: str) -> str:
    """Casefold, fold Turkish dotted/dotless i and other diacritics, drop punctuation."""
    folded = unicodedata.normalize("NFKD", text.translate(_TURKISH_FOLD).casefold())
    stripped = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z]+", " ", stripped).strip()


@functools.cache
def stopwords(locale: str) -> frozenset[str]:
    """Filler words of one locale from `locales/<locale>.json`, folded like the text.

    An unknown or malformed locale has no stopwords. The locale code is reduced to letters
    first, so it can never address another file.
    """
    code = re.sub(r"[^a-z]", "", locale.casefold())
    if not code:
        return frozenset()
    try:
        raw = (resources.files("vehicle_mcp") / "locales" / f"{code}.json").read_text("utf-8")
        words = json.loads(raw).get("stopwords", [])
    except (OSError, ValueError, AttributeError):
        return frozenset()
    return frozenset(normalise(w) for w in words if isinstance(w, str))


@functools.cache
def synonyms(locale: str) -> dict[str, tuple[str, ...]]:
    """Token expansions of one locale from the `synonyms` table in `locales/<locale>.json`.

    Keys and values are folded like the text. A complaint word listed here also counts as
    each of its synonyms ("dying" as "flat", "dead", "drained"), so riders' words reach the
    catalog's wording. The table is one-way and exact (no stemming on the key).
    """
    code = re.sub(r"[^a-z]", "", locale.casefold())
    if not code:
        return {}
    try:
        raw = (resources.files("vehicle_mcp") / "locales" / f"{code}.json").read_text("utf-8")
        table = json.loads(raw).get("synonyms", {})
    except (OSError, ValueError, AttributeError):
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for key, values in table.items():
        if isinstance(key, str) and isinstance(values, list):
            result[normalise(key)] = tuple(normalise(v) for v in values if isinstance(v, str))
    return result


def expand(words: Sequence[str], locales: Sequence[str] = ()) -> list[str]:
    """The words plus the synonyms of every word (English table always, then each locale)."""
    extra: list[str] = []
    for locale in ("en", *locales):
        table = synonyms(locale)
        for word in words:
            extra.extend(table.get(word, ()))
    return list(words) + [w for w in dict.fromkeys(extra) if w not in words]


def active_stopwords(locales: Sequence[str]) -> frozenset[str]:
    """English is always applied, together with every requested locale."""
    result = set(stopwords("en"))
    for locale in locales:
        result |= stopwords(locale)
    return frozenset(result)


def tokens(text: str, locales: Sequence[str] = ()) -> list[str]:
    """Content tokens of a text: folded, split, filler words of the active locales dropped."""
    drop = active_stopwords(locales)
    return [t for t in normalise(text).split() if t not in drop]


def _within_one_edit(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    short, long_ = sorted((a, b), key=len)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return short[i:] == long_[i + 1 :]


def same(a: str, b: str) -> bool:
    """Language-agnostic token equivalence: equal, shared stem, or a one-character typo.

    Stem: the common prefix is at least 3 characters and 60 % of the shorter token (so
    "noise"/"noises" and a stem plus any suffix match). Typo: both tokens have at least
    5 characters (the shorter at most 7, so two long words that merely share a start do not
    match), the same first character, and their first 5 characters differ by at most one
    edit (substitution, insertion or deletion).
    """
    if a == b:
        return True
    short = min(len(a), len(b))
    prefix = 0
    while prefix < short and a[prefix] == b[prefix]:
        prefix += 1
    if prefix >= STEM_MIN_PREFIX and prefix >= STEM_MIN_RATIO * short:
        return True
    if TYPO_MIN_LENGTH <= short <= TYPO_MAX_SHORT and a[0] == b[0]:
        return _within_one_edit(a[:TYPO_WINDOW], b[:TYPO_WINDOW])
    return False


class PhraseIndex:
    """Token weights over the complaint phrases of the active locale(s) in one catalog.

    A token's weight is `log(1 + N / df)` (N phrases, df of them containing the token), so
    words every phrase uses ("noise", "engine") count less than rare, telling ones.
    """

    def __init__(self, phrases: Iterable[str], locales: Sequence[str] = ()) -> None:
        self.locales = tuple(locales)
        token_sets = [set(tokens(p, self.locales)) for p in phrases]
        token_sets = [t for t in token_sets if t]
        count = len(token_sets)
        df: dict[str, int] = {}
        for token_set in token_sets:
            for token in token_set:
                df[token] = df.get(token, 0) + 1
        self.weights = {t: math.log(1 + count / n) for t, n in df.items()}

    def weight(self, token: str) -> float:
        # A token the index never saw (cannot happen for catalog phrases) counts as rare.
        return self.weights.get(token, math.log(2))


def phrase_score(complaint: Sequence[str], phrase: Sequence[str], index: PhraseIndex) -> float:
    """Weighted share of the phrase's content tokens that the complaint covers, 0..1.

    The complaint's extra words are free (a long question is not punished). At least one
    covered token must weigh at least the median weight of the phrase's own tokens, so a
    match made only of the phrase's commonest words ("engine", "starts") scores 0.0.
    """
    wanted = set(phrase)
    if not complaint or not wanted:
        return 0.0
    covered = [p for p in wanted if any(same(p, c) for c in set(complaint))]
    median = statistics.median(index.weight(p) for p in wanted)
    if not covered or max(index.weight(p) for p in covered) < median:
        return 0.0
    return sum(index.weight(p) for p in covered) / sum(index.weight(p) for p in wanted)


def best_phrase(
    complaint: str, phrases: Iterable[str], index: PhraseIndex
) -> tuple[float, str | None]:
    """The best coverage score of a complaint against catalog phrases, with the phrase."""
    c_tokens = expand(tokens(complaint, index.locales), index.locales)
    best, best_text = 0.0, None
    for phrase in phrases:
        score = phrase_score(c_tokens, tokens(phrase, index.locales), index)
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
