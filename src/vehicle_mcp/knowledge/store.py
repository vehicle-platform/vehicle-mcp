"""The knowledge index: one SQLite file with chunks, an FTS5 table and a sqlite-vec table.

Search is deterministic for a given index and query vector: BM25 (FTS5) and cosine (vec0)
rankings are fused with Reciprocal Rank Fusion (k = 60) and ties break on the chunk id.
Turkish FTS indexing and querying apply a light suffix-stripping stemmer driven by
`locales/tr.json` (G2-2); FTS queries also expand complaint synonyms from `locales/*.json`.
"""

from __future__ import annotations

import functools
import json
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from vehicle_mcp import matching

from . import KnowledgeUnavailable

SCHEMA_VERSION = "2"
RRF_K = 60
MAX_QUERY_TOKENS = 32
VECTOR_POOL_MIN = 30
VECTOR_POOL_PER_K = 5
VECTOR_FETCH_MAX = 500
FTS_POOL = 50
DEFAULT_RELATIVE_CUTOFF = 0.55
DEFAULT_MIN_SIMILARITY = 0.0

_WORD = re.compile(r"\w+", re.UNICODE)
_TR_LONG_SUFFIX_MIN_STEM = 3
_TR_SHORT_SUFFIX_MIN_STEM = 4
_TR_MAX_STEM_PASSES = 2


@dataclass(frozen=True)
class Chunk:
    """One indexed passage. `display`, when set, replaces `text` in search results."""

    source: str
    title: str
    text: str
    locale: str = "en"
    section: str = ""
    url_or_path: str = ""
    license: str = "project"
    kind: str = "doc"
    ref: str = ""  # group key: a fault id, or an opaque key (results keep one hit per ref)
    display: str | None = None


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    chunk_id: int
    score: float
    bm25_rank: int | None
    vector_rank: int | None
    similarity: float | None


@functools.cache
def turkish_suffixes() -> tuple[str, ...]:
    """Longest-first folded Turkish suffix list from `locales/tr.json`."""
    try:
        raw = (resources.files("vehicle_mcp") / "locales" / "tr.json").read_text("utf-8")
        items = json.loads(raw).get("suffixes", [])
    except (OSError, ValueError, AttributeError):
        return ()
    folded = [matching.normalise(s) for s in items if isinstance(s, str)]
    unique = dict.fromkeys(s for s in folded if s)
    return tuple(sorted(unique, key=len, reverse=True))


def stem_turkish(token: str) -> str:
    """Light suffix-stripping Turkish stemmer driven by `locales/tr.json` (no dependency)."""
    stem = matching.normalise(token)
    if len(stem) < _TR_SHORT_SUFFIX_MIN_STEM:
        return stem
    suffixes = turkish_suffixes()
    for _ in range(_TR_MAX_STEM_PASSES):
        stripped = False
        for suffix in suffixes:
            if not stem.endswith(suffix):
                continue
            candidate = stem[: -len(suffix)]
            min_len = _TR_LONG_SUFFIX_MIN_STEM if len(suffix) >= 3 else _TR_SHORT_SUFFIX_MIN_STEM
            if len(candidate) >= min_len:
                stem = candidate
                stripped = True
                break
        if not stripped:
            break
    return stem


def fts_index_text(text: str, locale: str) -> str:
    """Text inserted into `chunks_fts`: appends unique Turkish stems for `locale == 'tr'`."""
    if locale != "tr":
        return text
    extras: list[str] = []
    seen: set[str] = set()
    for raw in _WORD.findall(text):
        folded = matching.normalise(raw)
        if not folded:
            continue
        seen.add(folded)
        stemmed = stem_turkish(folded)
        if stemmed and stemmed != folded and stemmed not in seen:
            seen.add(stemmed)
            extras.append(stemmed)
    return f"{text} {' '.join(extras)}" if extras else text


def query_tokens(query: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(query)][:MAX_QUERY_TOKENS]


def fts_query(
    query: str,
    *,
    locales: Sequence[str] | None = None,
    stem_tr: bool = False,
    expand_synonyms: bool = False,
    drop_stopwords: bool = False,
) -> str:
    """An OR query of quoted word tokens (no FTS operator in user text can take effect)."""
    raw_tokens = query_tokens(query)
    if not raw_tokens:
        return ""
    if not (stem_tr or expand_synonyms or drop_stopwords):
        return " OR ".join(f'"{t}"' for t in raw_tokens)
    active_locales = tuple(locales) if locales is not None else ("en", "tr")
    stop = matching.active_stopwords(active_locales) if drop_stopwords else frozenset()
    kept = [t for t in raw_tokens if matching.normalise(t) not in stop]
    base = kept or raw_tokens
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        if term and term not in seen and len(terms) < MAX_QUERY_TOKENS:
            seen.add(term)
            terms.append(term)

    tr_active = stem_tr and ("tr" in active_locales)
    for tok in base:
        add(tok)
        folded = matching.normalise(tok)
        if folded:
            add(folded)
        if tr_active and folded:
            add(stem_turkish(folded))
    if expand_synonyms:
        folded_base = [matching.normalise(t) for t in base if matching.normalise(t)]
        for expanded in matching.expand(folded_base, active_locales):
            add(expanded)
    return " OR ".join(f'"{t}"' for t in terms)


def pack_vector(vector: Sequence[float]) -> bytes:
    import sqlite_vec

    return bytes(sqlite_vec.serialize_float32(list(vector)))


class KnowledgeIndex:
    """Open (or create) an index file. Use `open_index` / `create_index`."""

    def __init__(self, conn: sqlite3.Connection, dim: int) -> None:
        self._conn = conn
        self.dim = dim

    def close(self) -> None:
        self._conn.close()

    # --- writing ---------------------------------------------------------------------------
    def delete_source(self, source: str) -> int:
        ids = [
            r[0] for r in self._conn.execute("SELECT id FROM chunks WHERE source = ?", (source,))
        ]
        self._delete_ids(ids)
        return len(ids)

    def delete_where_source_like(self, prefix: str) -> int:
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        ids = [
            r[0]
            for r in self._conn.execute(
                "SELECT id FROM chunks WHERE source LIKE ? ESCAPE '\\'", (escaped + "%",)
            )
        ]
        self._delete_ids(ids)
        return len(ids)

    def _delete_ids(self, ids: list[int]) -> None:
        for chunk_id in ids:
            self._conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            self._conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
            self._conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (chunk_id,))
        self._conn.commit()

    def add_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.dim:
                raise ValueError(f"vector has {len(vector)} dimensions, index has {self.dim}")
            cursor = self._conn.execute(
                "INSERT INTO chunks(source, title, section, locale, text, url_or_path, license,"
                " kind, ref, display) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    chunk.source,
                    chunk.title,
                    chunk.section,
                    chunk.locale,
                    chunk.text,
                    chunk.url_or_path,
                    chunk.license,
                    chunk.kind,
                    chunk.ref,
                    chunk.display,
                ),
            )
            rowid = cursor.lastrowid
            fts_text = fts_index_text(chunk.text, chunk.locale)
            fts_title = fts_index_text(f"{chunk.title} {chunk.section}".strip(), chunk.locale)
            self._conn.execute(
                "INSERT INTO chunks_fts(rowid, text, title) VALUES (?,?,?)",
                (rowid, fts_text, fts_title),
            )
            self._conn.execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES (?,?)",
                (rowid, pack_vector(vector)),
            )
        self._conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value))
        self._conn.commit()

    # --- reading ---------------------------------------------------------------------------
    def meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def counts_by_source(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source"
        )
        return {str(s): int(n) for s, n in rows}

    def _chunk(self, chunk_id: int) -> Chunk:
        row = self._conn.execute(
            "SELECT source, title, section, locale, text, url_or_path, license, kind, ref, display"
            " FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        names = "source title section locale text url_or_path license kind ref display".split()
        return Chunk(**dict(zip(names, row, strict=True)))

    def _locale_and_ref_of(self, chunk_id: int) -> tuple[str, str]:
        row = self._conn.execute(
            "SELECT locale, ref FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return str(row[0]), str(row[1])

    def _bm25(
        self,
        query: str,
        locales: Sequence[str] | None,
        *,
        stem_tr: bool = True,
        expand_synonyms: bool = True,
        drop_stopwords: bool = True,
    ) -> list[int]:
        match = fts_query(
            query,
            locales=locales,
            stem_tr=stem_tr,
            expand_synonyms=expand_synonyms,
            drop_stopwords=drop_stopwords,
        )
        if not match:
            return []
        sql = (
            "SELECT chunks_fts.rowid FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ?"
        )
        args: list[Any] = [match]
        if locales:
            sql += f" AND c.locale IN ({','.join('?' * len(locales))})"
            args.extend(locales)
        sql += " ORDER BY bm25(chunks_fts), chunks_fts.rowid LIMIT ?"
        args.append(FTS_POOL)
        return [int(r[0]) for r in self._conn.execute(sql, args)]

    def _nearest(
        self,
        vector: Sequence[float],
        k: int,
        locales: Sequence[str] | None,
        *,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> list[tuple[int, float]]:
        total = self.count()
        if total == 0:
            return []
        fetch = min(total, VECTOR_FETCH_MAX)
        rows = self._conn.execute(
            "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (pack_vector(vector), fetch),
        ).fetchall()
        rows.sort(key=lambda row: (row[1], row[0]))  # a distance tie breaks on the chunk id
        keep = max(VECTOR_POOL_MIN, VECTOR_POOL_PER_K * k)
        out: list[tuple[int, float]] = []
        for rowid, distance in rows:
            sim = 1.0 - float(distance)
            if sim < min_similarity:
                break
            cid = int(rowid)
            locale, _ = self._locale_and_ref_of(cid)
            if locales and locale not in locales:
                continue
            out.append((cid, sim))
            if len(out) >= keep:
                break
        return out

    def search(
        self,
        query: str,
        vector: Sequence[float],
        *,
        k: int = 5,
        locales: Sequence[str] | None = None,
        relative_cutoff: float = DEFAULT_RELATIVE_CUTOFF,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        stem_tr: bool = True,
        expand_synonyms: bool = True,
        drop_stopwords: bool = True,
    ) -> list[Hit]:
        """Hybrid search: top `k` groups (one hit per `ref`) by fused score, best first.

        BM25 and vector rankings are fused at the group (`ref`) level so a fault entry whose
        BM25 hit and vector hit land on different complaint phrases still combines both ranks.
        """
        bm25_ids = self._bm25(
            query,
            locales,
            stem_tr=stem_tr,
            expand_synonyms=expand_synonyms,
            drop_stopwords=drop_stopwords,
        )
        near = self._nearest(vector, k, locales, min_similarity=min_similarity)
        bm25_group: dict[str, tuple[int, int]] = {}
        for rank, cid in enumerate(bm25_ids, start=1):
            _, ref = self._locale_and_ref_of(cid)
            group = ref or f"#{cid}"
            if group not in bm25_group:
                bm25_group[group] = (rank, cid)
        vec_group: dict[str, tuple[int, int, float]] = {}
        for rank, (cid, sim) in enumerate(near, start=1):
            _, ref = self._locale_and_ref_of(cid)
            group = ref or f"#{cid}"
            if group not in vec_group:
                vec_group[group] = (rank, cid, sim)
        scored: list[tuple[float, int, int | None, int | None, float | None]] = []
        for group in set(bm25_group) | set(vec_group):
            bm = bm25_group.get(group)
            vc = vec_group.get(group)
            score = (1.0 / (RRF_K + bm[0]) if bm else 0.0) + (1.0 / (RRF_K + vc[0]) if vc else 0.0)
            cid = bm[1] if (bm is not None and (vc is None or bm[0] <= vc[0])) else vc[1]  # type: ignore[index]
            scored.append(
                (score, cid, bm[0] if bm else None, vc[0] if vc else None, vc[2] if vc else None)
            )
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score = scored[0][0] if scored else 0.0
        min_score = best_score * relative_cutoff if relative_cutoff > 0 else 0.0
        hits: list[Hit] = []
        for score, cid, br, vr, hit_sim in scored:
            if hits and score < min_score:
                break
            hits.append(Hit(self._chunk(cid), cid, score, br, vr, hit_sim))
            if len(hits) >= k:
                break
        return hits


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    section TEXT NOT NULL DEFAULT '',
    locale TEXT NOT NULL,
    text TEXT NOT NULL,
    url_or_path TEXT NOT NULL DEFAULT '',
    license TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'doc',
    ref TEXT NOT NULL DEFAULT '',
    display TEXT
);
CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, title, tokenize = 'unicode61 remove_diacritics 2'
);
"""


def _connect(path: Path | str) -> sqlite3.Connection:
    try:
        import sqlite_vec
    except ImportError:
        raise KnowledgeUnavailable(
            "the optional 'rag' extra is not installed (pip install 'vehicle-mcp[rag]')"
        ) from None
    conn = sqlite3.connect(str(path))
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.Error) as exc:
        conn.close()
        raise KnowledgeUnavailable(
            f"this Python's sqlite3 cannot load the sqlite-vec extension ({exc})"
        ) from None
    return conn


def create_index(path: Path, *, dim: int, model_id: str, rebuild: bool = False) -> KnowledgeIndex:
    """Open the index for writing, creating the file and tables (and wiping with `rebuild`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild:
        path.unlink(missing_ok=True)
    conn = _connect(path)
    conn.executescript(_SCHEMA)
    existing = conn.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
    if existing is not None and (int(existing[0]) != dim or _model(conn) != model_id):
        conn.close()
        raise KnowledgeUnavailable(
            "the index was built with another embedding model; rebuild it with --rebuild"
        )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
        f"embedding float[{dim}] distance_metric=cosine)"
    )
    index = KnowledgeIndex(conn, dim)
    index.set_meta("schema_version", SCHEMA_VERSION)
    index.set_meta("dim", str(dim))
    index.set_meta("model_id", model_id)
    return index


def _model(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'model_id'").fetchone()
    return None if row is None else str(row[0])


def open_index(path: Path, *, model_id: str) -> KnowledgeIndex:
    """Open an existing index for searching; raises KnowledgeUnavailable with a reason."""
    if not path.is_file():
        raise KnowledgeUnavailable(
            "no knowledge index exists yet; build one with `vehicle-mcp-index build`"
        )
    conn = _connect(path)
    try:
        has_meta = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'meta' AND type = 'table'"
        ).fetchone()
        dim_row = (
            conn.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
            if has_meta
            else None
        )
    except sqlite3.DatabaseError:
        conn.close()
        raise KnowledgeUnavailable("the knowledge index file is not a valid index") from None
    if dim_row is None:
        conn.close()
        raise KnowledgeUnavailable("the knowledge index file is not a valid index")
    if _model(conn) != model_id:
        conn.close()
        raise KnowledgeUnavailable(
            "the index was built with another embedding model; rebuild it with --rebuild"
        )
    return KnowledgeIndex(conn, int(dim_row[0]))
