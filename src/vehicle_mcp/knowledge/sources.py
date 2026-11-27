"""Turn the three source kinds into chunks: the fault catalog, Markdown docs, a rider's PDF."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vehicle_mcp.matching import human_label, humanise_identifiers

from . import KnowledgeUnavailable
from .chunking import chunk_markdown, chunk_plain, first_heading
from .store import Chunk

CATALOG_SOURCE = "fault-catalog"
DOCS_SOURCE = "docs"
USER_LOCAL = "user-local"
DOCS_LICENSE = "project"
SKIP_DIRS = frozenset({"archive", "node_modules", ".git", ".claude"})  # archive: D-038


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def _symptom_line(symptom: Any, *, human: bool = False) -> str | None:
    if not isinstance(symptom, dict):
        return None
    parts: list[str] = []
    for key in ("signal", "direction", "condition"):
        val = symptom.get(key)
        if not val:
            continue
        text = str(val)
        if human:
            text = human_label(text) if key == "signal" else humanise_identifiers(text)
        parts.append(text)
    return " ".join(parts) or None


def _fault_text(entry: dict[str, Any], complaints: list[str], *, human: bool = False) -> str:
    raw_subsystem = str(entry.get("subsystem") or "").strip()
    if human:
        subsystem = human_label(raw_subsystem) if raw_subsystem else "general"
    else:
        subsystem = raw_subsystem or "unknown"
    lines = [f"Fault {entry['id']} ({subsystem})."]
    if entry.get("root_cause"):
        cause = str(entry["root_cause"])
        lines.append(f"Root cause: {humanise_identifiers(cause) if human else cause}.")
    raw_symptoms = entry.get("symptoms") or []
    symptoms = [s for x in raw_symptoms if (s := _symptom_line(x, human=human))]
    if symptoms:
        lines.append("Symptoms: " + "; ".join(symptoms) + ".")
    for label, key in (("Rider text", "rider_text"), ("Action", "action")):
        if entry.get(key):
            val = str(entry[key])
            lines.append(f"{label}: {humanise_identifiers(val) if human else val}")
    if complaints:
        lines.append("Complaints: " + "; ".join(complaints) + ".")
    return "\n".join(lines)


def catalog_chunks(payload: Any, *, all_displayable: bool = False) -> list[Chunk]:
    """Chunks from a `/faults` payload (`{"entries": [...]}` or a bare list).

    Displayable entries: one full-text chunk (root cause, symptoms, rider text, action and the
    English complaints) plus one chunk per complaint phrase (English and every i18n locale), all
    grouped by the fault id. Unconfirmed entries (D-084 item 5, D-127 addendum): only their
    complaint phrases and `needs`; no root cause or wording is indexed, and the result shows only
    the measurable signals, never the id. `all_displayable` is for a server that refused the
    unconfirmed fetch (every served entry is displayable by definition).
    """
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    chunks: list[Chunk] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        phrases: list[tuple[str, str]] = [("en", p) for p in _str_list(entry.get("complaints"))]
        i18n = entry.get("complaints_i18n")
        if isinstance(i18n, dict):
            for locale in sorted(i18n):
                phrases.extend((str(locale), p) for p in _str_list(i18n[locale]))
        if all_displayable or entry.get("displayable") is True:
            chunks.extend(_displayable(entry, phrases))
        else:
            chunks.extend(_unconfirmed(entry, phrases))
    return chunks


def _displayable(entry: dict[str, Any], phrases: list[tuple[str, str]]) -> list[Chunk]:
    fid = str(entry["id"])
    raw_subsystem = str(entry.get("subsystem") or "").strip()
    subsystem_label = human_label(raw_subsystem) if raw_subsystem else ""
    fallback_label = (
        humanise_identifiers(str(entry["root_cause"]).strip())
        if isinstance(entry.get("root_cause"), str) and entry["root_cause"].strip()
        else "catalog entry"
    )
    title = f"{fid} {subsystem_label or fallback_label}".strip()
    path = f"faults/{fid}"
    common = {
        "source": CATALOG_SOURCE,
        "title": title,
        "url_or_path": path,
        "license": DOCS_LICENSE,
        "ref": fid,
    }
    english = [p for loc, p in phrases if loc == "en"]
    out = [
        Chunk(
            text=_fault_text(entry, english),
            display=_fault_text(entry, english, human=True),
            locale="en",
            section="fault entry",
            kind="fault",
            **common,
        )
    ]
    for locale, phrase in phrases:
        out.append(
            Chunk(
                text=phrase,
                locale=locale,
                section=f"complaint phrase ({locale})",
                kind="complaint",
                **common,
            )
        )
    return out


def _unconfirmed(entry: dict[str, Any], phrases: list[tuple[str, str]]) -> list[Chunk]:
    needs = _str_list(entry.get("needs"))
    if not needs or not phrases:
        return []
    labels = [human_label(n) for n in needs]
    shown = "Measurable signals: " + ", ".join(labels) + ". No fault is named or confirmed."
    return [
        Chunk(
            source=CATALOG_SOURCE,
            title="unconfirmed catalog entry (data only)",
            text=f"{phrase}\nneeds: {' '.join(needs)}",
            locale=locale,
            section="data only",
            url_or_path="faults/unconfirmed",
            license=DOCS_LICENSE,
            kind="data_only",
            ref="unconfirmed",  # one data-only hit per search, never an id
            display=shown,
        )
        for locale, phrase in phrases
    ]


def docs_chunks(directory: Path, *, locale: str = "en") -> Iterator[Chunk]:
    """Markdown files below `directory` (skipping archive/ and hidden tool folders)."""
    if not directory.is_dir():
        raise KnowledgeUnavailable(f"docs directory {directory.name!r} does not exist")
    root = directory.resolve()
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        title = first_heading(text) or human_label(path.stem)
        shown_path = f"{root.name}/{rel.as_posix()}"
        for piece in chunk_markdown(text):
            yield Chunk(
                source=DOCS_SOURCE,
                title=title,
                text=piece.text,
                locale=locale,
                section=piece.section,
                url_or_path=shown_path,
                license=DOCS_LICENSE,
                kind="doc",
                ref="",
            )


def pdf_chunks(path: Path, *, locale: str = "en") -> Iterator[Chunk]:
    """A rider's own manual: text per page, flagged `user-local`, never shown beyond a snippet."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise KnowledgeUnavailable(
            "the optional 'rag' extra is not installed (pip install 'vehicle-mcp[rag]')"
        ) from None
    if not path.is_file():
        raise KnowledgeUnavailable(f"PDF {path.name!r} does not exist")
    reader = PdfReader(str(path))
    meta_title = getattr(reader.metadata, "title", None) if reader.metadata else None
    title = (meta_title or "").strip() or path.stem
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        for window in chunk_plain(text):
            yield Chunk(
                source=f"manual:{path.stem}",
                title=title,
                text=window,
                locale=locale,
                section=f"p. {number}",
                url_or_path=f"local:{path.name}",
                license=USER_LOCAL,
                kind="manual",
                ref="",
            )
