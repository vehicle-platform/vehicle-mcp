"""Retrieval quality evaluation harness for `search_knowledge` (G2-2, T60 follow-up).

Loads per-locale query files (`queries.json` for English, `locales/<locale>.json` for other
locales) and computes recall@1, recall@3, recall@5 and Mean Reciprocal Rank (MRR) against a
built knowledge index, plus weak-query noise metrics (average hits returned for out-of-domain
queries). Every query carries a `split`: `tune` queries may be used to choose parameters,
`holdout` queries never are (they come from the fault catalog's complaint phrases, not from the
author of the tuned parameters) and are only reported.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .embedder import Embedder
from .search import KnowledgeSearch
from .store import Chunk

DEFAULT_K = 5
SPLITS = ("tune", "test", "holdout")
DEFAULT_QUERIES_DIR = Path(__file__).resolve().parents[3] / "tests" / "retrieval_eval"


@dataclass(frozen=True)
class EvalQuery:
    id: str
    query: str
    expected: tuple[str, ...]
    split: str = "tune"


@dataclass(frozen=True)
class QueryOutcome:
    id: str
    query: str
    expected: tuple[str, ...]
    rank: int | None
    top_ids: tuple[str, ...]
    hit_count: int


@dataclass(frozen=True)
class LocaleReport:
    locale: str
    split: str
    queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    weak_queries: int
    weak_avg_hits: float
    weak_empty_rate: float
    misses: tuple[QueryOutcome, ...]


def _split_of(item: dict[str, Any]) -> str:
    split = str(item.get("split", "tune"))
    if split not in SPLITS:
        raise ValueError(f"query {item.get('id')!r}: split must be one of {SPLITS}, got {split!r}")
    return split


def load_query_file(queries_dir: Path, locale: str) -> tuple[list[EvalQuery], list[EvalQuery]]:
    """Returns `(answerable_queries, weak_queries)` for `locale`."""
    path = (
        queries_dir / "queries.json"
        if locale == "en"
        else queries_dir / "locales" / f"{locale}.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    answerable = [
        EvalQuery(
            id=str(item["id"]),
            query=str(item["query"]),
            expected=tuple(str(e) for e in item.get("expected", [])),
            split=_split_of(item),
        )
        for item in data.get("queries", [])
    ]
    weak = [
        EvalQuery(id=str(item["id"]), query=str(item["query"]), expected=(), split=_split_of(item))
        for item in data.get("weak_queries", [])
    ]
    return answerable, weak


def without_holdout(
    chunks: Sequence[Chunk], queries_dir: Path, locales: Sequence[str]
) -> list[Chunk]:
    """Chunks minus the catalog complaint phrases that are holdout queries.

    A holdout query is itself a catalog complaint phrase, so against an index that contains it
    the query is an exact match and every metric saturates. Building the eval index with this
    filter (`vehicle-mcp-index build --exclude-holdout`) removes each held-out phrase from its
    own complaint chunk and from the fault chunk's `Complaints:` line, so the holdout measures
    retrieval of a fault from a phrase the index has never seen.
    """
    held: set[str] = set()
    for locale in locales:
        answerable, _ = load_query_file(queries_dir, locale)
        held.update(q.query.strip().lower() for q in answerable if q.split in ("holdout", "test"))
    kept: list[Chunk] = []
    for chunk in chunks:
        if chunk.kind == "complaint" and chunk.text.strip().lower() in held:
            continue
        if chunk.kind == "fault" and "Complaints: " in chunk.text:
            head, _, tail = chunk.text.rpartition("Complaints: ")
            phrases = [p for p in tail.rstrip(".").split("; ") if p.strip().lower() not in held]
            text = head + ("Complaints: " + "; ".join(phrases) + "." if phrases else "")
            chunk = replace(chunk, text=text.rstrip())
        kept.append(chunk)
    return kept


def hit_source_id(hit: dict[str, Any]) -> str:
    """Canonical source id of a search hit: fault id or `url_or_path#section`."""
    fault_id = hit.get("fault_id")
    if isinstance(fault_id, str) and fault_id:
        return fault_id
    citation = hit.get("citation") or {}
    path = str(citation.get("url_or_path", ""))
    section = str(citation.get("section", ""))
    return f"{path}#{section}" if section else path


def matches_expected(hit: dict[str, Any], expected: Sequence[str]) -> bool:
    """True when `hit` matches any expected fault id or `url_or_path#section` target."""
    fault_id = hit.get("fault_id")
    citation = hit.get("citation") or {}
    path = str(citation.get("url_or_path", ""))
    section = str(citation.get("section", ""))
    for target in expected:
        if "#" in target:
            exp_path, exp_section = target.split("#", 1)
            if path == exp_path and (section == exp_section or section.endswith(exp_section)):
                return True
        elif fault_id == target or path == f"faults/{target}":
            return True
    return False


def evaluate_locale(
    searcher: KnowledgeSearch,
    locale: str,
    answerable: Sequence[EvalQuery],
    weak: Sequence[EvalQuery] = (),
    *,
    k: int = DEFAULT_K,
    split: str = "tune",
) -> LocaleReport:
    """Runs all queries of one locale and computes recall@1/@3/@5, MRR and weak-query noise."""
    r1 = r3 = r5 = 0
    rr_sum = 0.0
    misses: list[QueryOutcome] = []
    for item in answerable:
        result = searcher.search(item.query, locale=locale, k=k)
        hits = result.get("hits", []) if result.get("available") else []
        found_rank: int | None = None
        for idx, hit in enumerate(hits, start=1):
            if matches_expected(hit, item.expected):
                found_rank = idx
                break
        if found_rank is not None:
            if found_rank <= 1:
                r1 += 1
            if found_rank <= 3:
                r3 += 1
            if found_rank <= 5:
                r5 += 1
            rr_sum += 1.0 / found_rank
        if found_rank is None or found_rank > 1:
            misses.append(
                QueryOutcome(
                    id=item.id,
                    query=item.query,
                    expected=item.expected,
                    rank=found_rank,
                    top_ids=tuple(hit_source_id(h) for h in hits[:3]),
                    hit_count=len(hits),
                )
            )
    weak_hits_total = 0
    weak_empty = 0
    for item in weak:
        result = searcher.search(item.query, locale=locale, k=k)
        hits = result.get("hits", []) if result.get("available") else []
        weak_hits_total += len(hits)
        if not hits:
            weak_empty += 1
    n = len(answerable) or 1
    nw = len(weak) or 1
    return LocaleReport(
        locale=locale,
        split=split,
        queries=len(answerable),
        recall_at_1=r1 / n,
        recall_at_3=r3 / n,
        recall_at_5=r5 / n,
        mrr=rr_sum / n,
        weak_queries=len(weak),
        weak_avg_hits=(weak_hits_total / nw) if weak else 0.0,
        weak_empty_rate=(weak_empty / nw) if weak else 0.0,
        misses=tuple(misses),
    )


def evaluate_all(
    db_path: Path,
    queries_dir: Path,
    locales: Sequence[str],
    *,
    embedder: Embedder,
    k: int = DEFAULT_K,
    splits: Sequence[str] = SPLITS,
    lexical_top_n: int | None = None,
    min_similarity: float | None = None,
) -> list[LocaleReport]:
    """Evaluates each requested locale and split against `db_path` using `embedder`."""
    options: dict[str, Any] = {}
    if lexical_top_n is not None:
        options["lexical_top_n"] = lexical_top_n
    searcher = KnowledgeSearch(
        db_path, embedder_factory=lambda: embedder, min_similarity=min_similarity, **options
    )
    reports: list[LocaleReport] = []
    for locale in locales:
        answerable, weak = load_query_file(queries_dir, locale)
        for split in splits:
            part = [q for q in answerable if q.split == split]
            if not part:
                continue
            noise = [q for q in weak if q.split == split]
            reports.append(evaluate_locale(searcher, locale, part, noise, k=k, split=split))
    return reports


def format_table(reports: Sequence[LocaleReport]) -> str:
    """Formats a compact plain-text summary table for CLI output."""
    header = (
        f"{'locale':<6}  {'split':<7}  {'n':>3}  {'R@1':>6}  {'R@3':>6}  {'R@5':>6}  "
        f"{'MRR':>6}  {'weak_hits':>9}  {'weak_empty':>10}"
    )
    lines = [header, "-" * len(header)]
    for rep in reports:
        lines.append(
            f"{rep.locale:<6}  {rep.split:<7}  {rep.queries:>3}  {rep.recall_at_1:>6.3f}  "
            f"{rep.recall_at_3:>6.3f}  {rep.recall_at_5:>6.3f}  {rep.mrr:>6.3f}  "
            f"{rep.weak_avg_hits:>9.2f}  {rep.weak_empty_rate:>10.3f}"
        )
    return "\n".join(lines)


def report_to_json(reports: Sequence[LocaleReport]) -> str:
    return json.dumps([asdict(r) for r in reports], indent=2, ensure_ascii=False)
