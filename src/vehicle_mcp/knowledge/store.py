"""The knowledge index: one SQLite file with chunks, an FTS5 table and a sqlite-vec table.

Search is deterministic for a given index and query vector: BM25 (FTS5) and cosine (vec0)
rankings are fused with Reciprocal Rank Fusion (k = 60) and ties break on the chunk id.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import KnowledgeUnavailable

SCHEMA_VERSION = "1"
RRF_K = 60
MAX_QUERY_TOKENS = 32
VECTOR_POOL_MIN = 30
VECTOR_POOL_PER_K = 5
VECTOR_FETCH_MAX = 500
FTS_POOL = 50

_WORD = re.compile(r"\w+", re.UNICODE)


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


def query_tokens(query: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(query)][:MAX_QUERY_TOKENS]


def fts_query(query: str) -> str:
    """An OR query of quoted word tokens (no FTS operator in user text can take effect)."""
    return " OR ".join(f'"{t}"' for t in query_tokens(query))


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
            self._conn.execute(
                "INSERT INTO chunks_fts(rowid, text, title) VALUES (?,?,?)",
                (rowid, chunk.text, f"{chunk.title} {chunk.section}".strip()),
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

    def _locale_of(self, chunk_id: int) -> str:
        return str(
            self._conn.execute("SELECT locale FROM chunks WHERE id = ?", (chunk_id,)).fetchone()[0]
        )

    def _bm25(self, query: str, locales: Sequence[str] | None) -> list[int]:
        match = fts_query(query)
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
        self, vector: Sequence[float], k: int, locales: Sequence[str] | None
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
            if locales and self._locale_of(int(rowid)) not in locales:
                continue
            out.append((int(rowid), 1.0 - float(distance)))
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
    ) -> list[Hit]:
        """Hybrid search: top `k` groups (one hit per `ref`) by fused score, best first."""
        bm25 = {cid: rank for rank, cid in enumerate(self._bm25(query, locales), start=1)}
        near = self._nearest(vector, k, locales)
        vec_rank = {cid: rank for rank, (cid, _) in enumerate(near, start=1)}
        similarity = dict(near)
        fused: dict[int, float] = {}
        for ranks in (bm25, vec_rank):
            for cid, rank in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        ordered = sorted(fused, key=lambda cid: (-fused[cid], cid))
        hits: list[Hit] = []
        seen: set[str] = set()
        for cid in ordered:
            chunk = self._chunk(cid)
            group = chunk.ref or f"#{cid}"
            if group in seen:
                continue
            seen.add(group)
            hits.append(
                Hit(chunk, cid, fused[cid], bm25.get(cid), vec_rank.get(cid), similarity.get(cid))
            )
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
