"""The knowledge index: one SQLite file with chunks, an FTS5 table and a sqlite-vec table.

Search is deterministic for a given index and query vector: BM25 (FTS5) and cosine (vec0)
rankings are fused with Reciprocal Rank Fusion (k = 60) and ties break on the chunk id.
RRF scores are rank-based, so relevance is NOT decided on them (a hit found in one list at rank 1
scores exactly half of one found in both). A group is kept when it passes the vector cosine floor
(it is in the vector pool) or is a lexical match within the top `lexical_top_n` BM25 groups.
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
from .settings import (
    DEFAULT_FTS_POOL,
    DEFAULT_LEXICAL_TOP_N,
    DEFAULT_RRF_K,
    DEFAULT_VECTOR_POOL_MIN,
    RetrievalSettings,
)

SCHEMA_VERSION = "2"
RRF_K = DEFAULT_RRF_K
MAX_QUERY_TOKENS = 32
VECTOR_POOL_MIN = DEFAULT_VECTOR_POOL_MIN
VECTOR_POOL_PER_K = 5
VECTOR_FETCH_MAX = 500
FTS_POOL = DEFAULT_FTS_POOL
DEFAULT_MIN_SIMILARITY = 0.0

_WORD = re.compile(r"\w+", re.UNICODE)
_TR_MIN_SUFFIX_LEN = 2  # one-letter suffixes over-strip ("lamba" -> "lamb")
_TR_MIN_STEM = 5  # a stem shorter than this is never produced (akü, lamba stay intact)


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
    bm25_rank: int | None  # 1-based rank of the group's best chunk among BM25 groups
    vector_rank: int | None  # 1-based rank of the group's best chunk among vector groups
    similarity: float | None  # cosine of that vector chunk (None: not in the vector pool)


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
    """Light, single-pass Turkish suffix stripper driven by `locales/tr.json` (no dependency).

    Only suffixes of at least two letters are used, one is stripped at most once and the stem
    never gets shorter than five letters, so short roots and words that merely end in a vowel
    ("akü", "lamba", "motor") stay intact.
    """
    stem = matching.normalise(token)
    if len(stem) <= _TR_MIN_STEM:
        return stem
    for suffix in turkish_suffixes():
        if len(suffix) >= _TR_MIN_SUFFIX_LEN and stem.endswith(suffix):
            candidate = stem[: -len(suffix)]
            if len(candidate) >= _TR_MIN_STEM:
                return candidate
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
    """An OR query of quoted word tokens (no FTS operator in user text can take effect).

    `stem_tr` is set by the caller only when the active locale is exactly `tr`.
    """
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

    tr_active = stem_tr
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

    def __init__(
        self,
        conn: sqlite3.Connection,
        dim: int,
        *,
        retrieval: RetrievalSettings | None = None,
    ) -> None:
        self._conn = conn
        self.dim = dim
        self._retrieval = retrieval or RetrievalSettings()

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

    def _locale_and_ref_of(
        self, chunk_id: int, cache: dict[int, tuple[str, str]] | None = None
    ) -> tuple[str, str]:
        if cache is not None and chunk_id in cache:
            return cache[chunk_id]
        row = self._conn.execute(
            "SELECT locale, ref FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        value = (str(row[0]), str(row[1]))
        if cache is not None:
            cache[chunk_id] = value
        return value

    def _bm25(
        self,
        query: str,
        locales: Sequence[str] | None,
        *,
        stem_tr: bool = False,
        expand_synonyms: bool = True,
        drop_stopwords: bool = True,
        fts_pool: int = FTS_POOL,
    ) -> list[int]:
        if fts_pool <= 0:
            return []
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
        args.append(fts_pool)
        return [int(r[0]) for r in self._conn.execute(sql, args)]

    def _nearest(
        self,
        vector: Sequence[float],
        k: int,
        locales: Sequence[str] | None,
        *,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        vector_pool_min: int = VECTOR_POOL_MIN,
        cache: dict[int, tuple[str, str]] | None = None,
    ) -> list[tuple[int, float]]:
        total = self.count()
        if total == 0:
            return []
        keep = max(vector_pool_min, VECTOR_POOL_PER_K * k)
        if keep <= 0:
            return []
        fetch = min(total, VECTOR_FETCH_MAX)
        rows = self._conn.execute(
            "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (pack_vector(vector), fetch),
        ).fetchall()
        rows.sort(key=lambda row: (row[1], row[0]))  # a distance tie breaks on the chunk id
        out: list[tuple[int, float]] = []
        for rowid, distance in rows:
            sim = 1.0 - float(distance)
            if sim < min_similarity:
                break
            cid = int(rowid)
            locale, _ = self._locale_and_ref_of(cid, cache)
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
        lexical_top_n: int = DEFAULT_LEXICAL_TOP_N,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        stem_tr: bool = False,
        expand_synonyms: bool = True,
        drop_stopwords: bool = True,
    ) -> list[Hit]:
        """Hybrid search: top `k` groups (one hit per `ref`) by fused score, best first.

        BM25 and vector rankings are fused at the group (`ref`) level so a fault entry whose
        BM25 hit and vector hit land on different complaint phrases still combines both ranks
        (`Hit.bm25_rank` / `Hit.vector_rank` are the ranks of the group's best chunk in each
        list). Relevance is decided on the underlying signals, never on the rank-based fused
        score: a group survives when its best chunk has cosine similarity >= `min_similarity`
        (it is in the vector pool) or when it is among the first `lexical_top_n` BM25 groups.
        With `lexical_top_n = 0` or `fts_weight = 0` only vector hits survive; with
        `vector_weight = 0` only FTS hits survive.
        """
        cfg = self._retrieval
        cache: dict[int, tuple[str, str]] = {}
        bm25_ids = (
            self._bm25(
                query,
                locales,
                stem_tr=stem_tr,
                expand_synonyms=expand_synonyms,
                drop_stopwords=drop_stopwords,
                fts_pool=cfg.fts_pool,
            )
            if cfg.fts_weight > 0.0 and lexical_top_n > 0
            else []
        )
        near = (
            self._nearest(
                vector,
                k,
                locales,
                min_similarity=min_similarity,
                vector_pool_min=cfg.vector_pool_min,
                cache=cache,
            )
            if cfg.vector_weight > 0.0
            else []
        )
        bm25_group: dict[str, tuple[int, int]] = {}
        for cid in bm25_ids:
            group = self._locale_and_ref_of(cid, cache)[1] or f"#{cid}"
            if group not in bm25_group:
                bm25_group[group] = (len(bm25_group) + 1, cid)
        vec_group: dict[str, tuple[int, int, float]] = {}
        for cid, sim in near:
            group = self._locale_and_ref_of(cid, cache)[1] or f"#{cid}"
            if group not in vec_group:
                vec_group[group] = (len(vec_group) + 1, cid, sim)
        return self._fuse(bm25_group, vec_group, k=k, lexical_top_n=lexical_top_n)

    def _fuse(
        self,
        bm25_group: dict[str, tuple[int, int]],
        vec_group: dict[str, tuple[int, int, float]],
        *,
        k: int,
        lexical_top_n: int,
    ) -> list[Hit]:
        cfg = self._retrieval
        scored: list[tuple[float, int, int | None, int | None, float | None]] = []
        for group in set(bm25_group) | set(vec_group):
            bm = bm25_group.get(group)
            vc = vec_group.get(group)
            if bm is not None and bm[0] > lexical_top_n:
                bm = None  # a weak lexical match does not count as evidence on its own
            if bm is None and vc is None:
                continue
            bm_score = (cfg.fts_weight / (cfg.rrf_k + bm[0])) if bm else 0.0
            vc_score = (cfg.vector_weight / (cfg.rrf_k + vc[0])) if vc else 0.0
            score = bm_score + vc_score
            if bm is not None and (vc is None or bm[1] == vc[1] or bm_score >= vc_score):
                cid = bm[1]
            elif vc is not None:
                cid = vc[1]
            else:
                continue
            scored.append(
                (score, cid, bm[0] if bm else None, vc[0] if vc else None, vc[2] if vc else None)
            )
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            Hit(self._chunk(cid), cid, score, br, vr, hit_sim)
            for score, cid, br, vr, hit_sim in scored[:k]
        ]


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


def open_index(
    path: Path, *, model_id: str, retrieval: RetrievalSettings | None = None
) -> KnowledgeIndex:
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
    version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if version is None or str(version[0]) != SCHEMA_VERSION:
        conn.close()
        raise KnowledgeUnavailable(
            "the index was built by another version of vehicle-mcp (schema "
            f"{version[0] if version else 'unknown'}, expected {SCHEMA_VERSION}); "
            "rebuild it with `vehicle-mcp-index build --rebuild`"
        )
    return KnowledgeIndex(conn, int(dim_row[0]), retrieval=retrieval)
