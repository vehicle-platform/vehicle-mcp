"""Synthetic corpus, a deterministic fake embedder and a tiny PDF writer for the T60 tests."""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Words of one concept share a dimension, so the fake embedder "understands" synonyms and
# English/Turkish pairs the way a multilingual model would (coarsely).
CONCEPTS: dict[str, tuple[str, ...]] = {
    "tick": ("tick", "ticking", "tap", "clatter", "tıkırtı", "tikirti", "supap", "valve"),
    "battery": ("battery", "akü", "flat", "charging", "şarj", "stator"),
    "heat": ("coolant", "hot", "overheat", "radiator", "fan", "soğutma", "sıcak"),
    "traffic": ("traffic", "idle", "trafik", "standstill"),
    "oil": ("oil", "yağ", "pressure"),
}
DIM = 16
_WORD = re.compile(r"\w+", re.UNICODE)


class FakeEmbedder:
    """Concept-bucket + hashed bag of words, L2-normalised. Deterministic, no model."""

    model_id = "fake-embedder@1"
    dim = DIM
    min_similarity = 0.0

    def __init__(self) -> None:
        self.prefixed: list[str] = []

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in _WORD.findall(text.lower()):
            for index, words in enumerate(CONCEPTS.values()):
                if any(word.startswith(w) for w in words):
                    vec[index] += 2.0
                    break
            else:
                vec[len(CONCEPTS) + sum(map(ord, word)) % (DIM - len(CONCEPTS))] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.prefixed.extend("query: " + t for t in texts)
        return [self._vector(t) for t in texts]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.prefixed.extend("passage: " + t for t in texts)
        return [self._vector(t) for t in texts]


def catalog_payload() -> dict[str, Any]:
    return {
        "defs_version": "v0.0.0",
        "vehicle": "test",
        "entries": [
            {
                "id": "ENG-001",
                "subsystem": "engine",
                "root_cause": "Valve clearance too large",
                "symptoms": [
                    {"signal": "did.ENGINE_SPEED", "direction": "up", "condition": "idle"}
                ],
                "needs": ["did.ENGINE_SPEED"],
                "advise_stop": False,
                "rider_text": "A ticking noise from the top of the engine.",
                "action": "Have the valve clearance checked.",
                "displayable": True,
                "complaints": ["ticking noise from the engine", "tapping sound at idle"],
                "complaints_i18n": {"tr": ["supaplardan tıkırtı sesi", "motor tık tık ediyor"]},
            },
            {
                "id": "COO-001",
                "subsystem": "cooling",
                "root_cause": "Radiator fan relay stuck open",
                "symptoms": [],
                "needs": ["did.COOLANT_TEMP"],
                "rider_text": "Coolant rises in traffic because the fan does not start.",
                "action": "Check the fan relay.",
                "displayable": True,
                "complaints": ["coolant gets hot in traffic"],
                "complaints_i18n": {"tr": ["trafikte motor ısınıyor"]},
            },
            {
                "id": "CHG-009",
                "subsystem": "charging",
                "root_cause": "SECRET-WORDING stator winding open",
                "needs": ["did.BATTERY_VOLTAGE", "derived.charging_ripple"],
                "rider_text": "SECRET-WORDING stop now",
                "action": "SECRET-WORDING test the stator",
                "displayable": False,
                "complaints": ["battery keeps going flat"],
                "complaints_i18n": {"tr": ["akü sürekli bitiyor"]},
            },
        ],
    }


DOC_TEXT = """# Cooling guide

Intro paragraph about the cooling system.

## Fan

The radiator fan switches on at a coolant temperature threshold. In slow traffic the airflow
is low, so the coolant temperature rises until the fan starts.

```
# not a heading
```

## Oil

The oil pressure warning means low oil pressure.
"""


def write_docs(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cooling.md").write_text(DOC_TEXT, encoding="utf-8")
    (directory / "archive").mkdir(exist_ok=True)
    (directory / "archive" / "old.md").write_text("# Old\n\narchived text", encoding="utf-8")
    return directory


def make_pdf(path: Path, pages: Sequence[str]) -> Path:
    """A minimal valid PDF with one text page per entry (generated at test time, never kept)."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    font_id = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        safe = text.replace("(", "[").replace(")", "]").replace("\\", "/")
        stream = zlib.compress(f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode())
        objects.append(
            f"<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        )
        assert page_id == 3 + 2 * i
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(bytes(out))
    return path
