"""`vehicle-mcp-index`: build the local knowledge index (T60).

    vehicle-mcp-index fetch-model                 # one-time download of the pinned model
    vehicle-mcp-index build --docs ../vehicle-defs/docs --local-pdf ~/manuals/cl250.pdf

Sources: (a) the fault catalog from vehicle-server (`/faults?include_unconfirmed=true`),
(b) Markdown directories, (c) a rider's own PDF (flagged `user-local`, kept only in this local
index). The index file must stay out of git; a PDF import is refused when the target file would
sit inside an unignored git work tree.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from vehicle_mcp.config import ConfigError, load_settings
from vehicle_mcp.source import HistorySource, SourceError, UnconfirmedRefused

from . import KnowledgeUnavailable
from .embedder import Embedder, OnnxE5Embedder, fetch_model, model_present, verify_model
from .eval_retrieval import (
    DEFAULT_K,
    DEFAULT_QUERIES_DIR,
    evaluate_all,
    format_table,
    report_to_json,
    without_holdout,
)
from .settings import (
    DEFAULT_FTS_POOL,
    DEFAULT_FTS_WEIGHT,
    DEFAULT_LEXICAL_TOP_N,
    DEFAULT_RRF_K,
    DEFAULT_VECTOR_POOL_MIN,
    DEFAULT_VECTOR_WEIGHT,
    RetrievalSettings,
    default_db_path,
    default_model_dir,
)
from .sources import CATALOG_SOURCE, DOCS_SOURCE, catalog_chunks, docs_chunks, pdf_chunks
from .store import Chunk, KnowledgeIndex, create_index

ClientFactory = Callable[[], HistorySource]


def passage_text(chunk: Chunk) -> str:
    """What the embedder sees: the document and section context, then the passage."""
    head = " | ".join(p for p in (chunk.title, chunk.section) if p)
    return f"{head}\n{chunk.text}" if head and chunk.kind in {"doc", "manual"} else chunk.text


def _embed_all(embedder: Embedder, chunks: Sequence[Chunk]) -> list[list[float]]:
    """Embed in length order (less padding), return vectors in the original order."""
    order = sorted(range(len(chunks)), key=lambda i: (len(chunks[i].text), i))
    vectors = embedder.embed_passages([passage_text(chunks[i]) for i in order])
    out: list[list[float]] = [[] for _ in chunks]
    for position, original in enumerate(order):
        out[original] = vectors[position]
    return out


def _tracked_by_git_risk(db: Path) -> bool:
    """True when `db` would sit inside a git work tree that does not ignore it."""
    parent = db.resolve().parent
    for candidate in (parent, *parent.parents):
        if (candidate / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(candidate), "check-ignore", "-q", str(db.resolve())],
                capture_output=True,
                check=False,
            )
            return result.returncode != 0
    return False


def _fetch_catalog(client_factory: ClientFactory) -> tuple[Any, bool]:
    client = client_factory()
    try:
        try:
            return client.fault_catalog(subsystem=None, include_unconfirmed=True), False
        except UnconfirmedRefused:
            return client.fault_catalog(subsystem=None), True
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _replace(
    index: KnowledgeIndex, embedder: Embedder, source: str, chunks: list[Chunk], log: Any
) -> None:
    index.delete_source(source)
    index.add_chunks(chunks, _embed_all(embedder, chunks))
    log(f"  {source}: {len(chunks)} chunks")


def build(
    args: argparse.Namespace,
    *,
    embedder: Embedder,
    client_factory: ClientFactory | None,
    log: Callable[[str], None],
) -> int:
    db = Path(args.db) if args.db else default_db_path()
    pdfs = [Path(p) for p in args.local_pdf]
    if pdfs and _tracked_by_git_risk(db):
        log(
            "refusing to import a local PDF: the index file would sit inside a git work tree "
            "that does not ignore it (manuals must never reach a public repo)"
        )
        return 2
    index = create_index(db, dim=embedder.dim, model_id=embedder.model_id, rebuild=args.rebuild)
    try:
        if not args.no_catalog and client_factory is not None:
            payload, refused = _fetch_catalog(client_factory)
            chunks = catalog_chunks(payload, all_displayable=refused)
            if args.exclude_holdout:
                chunks = without_holdout(chunks, DEFAULT_QUERIES_DIR, ("en", "tr"))
            _replace(index, embedder, CATALOG_SOURCE, chunks, log)
        docs: list[Chunk] = []
        for directory in args.docs:
            docs.extend(docs_chunks(Path(directory), locale=args.docs_locale))
        if args.docs:
            _replace(index, embedder, DOCS_SOURCE, docs, log)
        for pdf in pdfs:
            pages = list(pdf_chunks(pdf, locale=args.pdf_locale))
            _replace(index, embedder, f"manual:{pdf.stem}", pages, log)
        log(f"index: {index.count()} chunks, {db.stat().st_size // 1024} KiB")
    finally:
        index.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vehicle-mcp-index", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-model", help="download the pinned embedding model")
    fetch.add_argument("--verify", action="store_true", help="re-hash the cached files")
    fetch.add_argument("--model-dir", help="override the model cache directory")
    b = sub.add_parser("build", help="build or update the index")
    b.add_argument("--db", help="index file (default: user data dir, or $VEHICLE_MCP_KNOWLEDGE_DB)")
    b.add_argument("--rebuild", action="store_true", help="delete the index first")
    b.add_argument("--no-catalog", action="store_true", help="skip the fault catalog")
    b.add_argument("--docs", action="append", default=[], metavar="DIR", help="Markdown directory")
    b.add_argument("--docs-locale", default="en")
    b.add_argument(
        "--local-pdf", action="append", default=[], metavar="PDF", help="your own manual"
    )
    b.add_argument("--pdf-locale", default="en")
    b.add_argument(
        "--exclude-holdout",
        action="store_true",
        help="eval index only: leave the held-out complaint phrases (tests/retrieval_eval) out",
    )
    b.add_argument("--model-dir", help="override the model cache directory")
    b.add_argument(
        "--no-prefixes",
        action="store_true",
        help="disable e5 `query: `/`passage: ` prefixes when embedding passages",
    )
    ev = sub.add_parser("eval", help="evaluate retrieval recall@1/@3/@5 and MRR per locale")
    ev.add_argument(
        "--db", help="index file (default: user data dir, or $VEHICLE_MCP_KNOWLEDGE_DB)"
    )
    ev.add_argument(
        "--queries-dir",
        default=str(DEFAULT_QUERIES_DIR),
        help="directory holding queries.json and locales/<locale>.json "
        "(default: tests/retrieval_eval of the source checkout)",
    )
    ev.add_argument(
        "--locales",
        nargs="+",
        default=["en", "tr"],
        help="locales to evaluate (default: en tr)",
    )
    ev.add_argument(
        "--split",
        choices=["tune", "test", "holdout", "both", "all"],
        default="all",
        help="query split to report (default: all; tune parameters on `tune` only)",
    )
    ev.add_argument("--k", type=int, default=DEFAULT_K, help="top-k cutoff (default: 5)")
    ev.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF constant k (default: 60)")
    ev.add_argument(
        "--fts-weight",
        type=float,
        default=DEFAULT_FTS_WEIGHT,
        help="RRF weight of the FTS arm (default: 1.0; 0 = vector-only)",
    )
    ev.add_argument(
        "--vector-weight",
        type=float,
        default=DEFAULT_VECTOR_WEIGHT,
        help="RRF weight of the vector arm (default: 1.0; 0 = FTS-only)",
    )
    ev.add_argument(
        "--fts-pool", type=int, default=DEFAULT_FTS_POOL, help="BM25 candidate pool size"
    )
    ev.add_argument(
        "--vector-pool-min",
        type=int,
        default=DEFAULT_VECTOR_POOL_MIN,
        help="minimum vector candidate pool size",
    )
    ev.add_argument(
        "--lexical-top-n",
        type=int,
        default=DEFAULT_LEXICAL_TOP_N,
        help="top BM25 groups eligible on lexical evidence alone (default: 2)",
    )
    ev.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="override cosine similarity floor (default: embedder default 0.83)",
    )
    ev.add_argument(
        "--no-prefixes",
        action="store_true",
        help="disable e5 `query: `/`passage: ` prefixes when embedding queries",
    )
    ev.add_argument("--model-dir", help="override the model cache directory")
    ev.add_argument("--json", action="store_true", dest="as_json", help="emit JSON report")
    ev.add_argument("--show-misses", action="store_true", help="print non-rank-1 queries")
    return parser


def _default_client() -> HistorySource:
    from vehicle_mcp.client import VehicleServerClient

    return VehicleServerClient(load_settings())


def main(
    argv: Sequence[str] | None = None,
    *,
    embedder: Embedder | None = None,
    client_factory: ClientFactory | None = None,
) -> int:
    args = _parser().parse_args(argv)

    def log(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        if args.command == "fetch-model":
            model_dir = Path(args.model_dir) if args.model_dir else default_model_dir()
            fetch_model(model_dir, log=log)
            bad = verify_model(model_dir) if args.verify else []
            if bad:
                log("verification failed: " + ", ".join(bad))
                return 1
            log(f"model ready{' and verified' if args.verify else ''}")
            return 0
        model_dir = Path(args.model_dir) if args.model_dir else default_model_dir()
        use_prefixes = not getattr(args, "no_prefixes", False)
        if embedder is None:
            if not model_present(model_dir):
                log("the embedding model is missing; running fetch-model first")
                fetch_model(model_dir, log=log)
            embedder = OnnxE5Embedder(model_dir, use_prefixes=use_prefixes)
        if args.command == "eval":
            db = Path(args.db) if args.db else default_db_path()
            queries_dir = Path(args.queries_dir)
            if not queries_dir.is_dir():
                log(f"vehicle-mcp-index: queries directory {queries_dir!s} does not exist")
                return 2
            split_map: dict[str, tuple[str, ...]] = {
                "all": ("tune", "test", "holdout"),
                "both": ("tune", "holdout"),
            }
            splits = split_map.get(args.split, (args.split,))
            retrieval = RetrievalSettings(
                rrf_k=args.rrf_k,
                fts_weight=args.fts_weight,
                vector_weight=args.vector_weight,
                fts_pool=args.fts_pool,
                vector_pool_min=args.vector_pool_min,
                lexical_top_n=args.lexical_top_n,
                min_similarity=args.min_similarity,
                use_prefixes=use_prefixes,
            )
            reports = evaluate_all(
                db,
                queries_dir,
                args.locales,
                embedder=embedder,
                k=args.k,
                splits=splits,
                retrieval=retrieval,
            )
            if args.as_json:
                print(report_to_json(reports))
            else:
                print(format_table(reports))
                if args.show_misses:
                    for rep in reports:
                        if rep.misses:
                            print(f"\n[{rep.locale}/{rep.split} non-rank-1: {len(rep.misses)}]")
                            for m in rep.misses:
                                print(
                                    f"  {m.id} (rank={m.rank}): {m.query!r} "
                                    f"expected={list(m.expected)} got={list(m.top_ids)}"
                                )
            return 0
        factory = client_factory
        if factory is None and not args.no_catalog:
            factory = _default_client
        return build(args, embedder=embedder, client_factory=factory, log=log)
    except (KnowledgeUnavailable, ConfigError, SourceError, ValueError) as exc:
        log(f"vehicle-mcp-index: {exc}")
        return 2


def cli() -> None:
    raise SystemExit(main())
