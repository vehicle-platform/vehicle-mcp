"""T60: hybrid knowledge index, chunking, sources, indexer CLI and the search_knowledge tool."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from tests.knowledge_support import (
    DIM,
    FakeEmbedder,
    catalog_payload,
    make_pdf,
    write_docs,
)
from tests.support import FakeVehicleServer
from vehicle_mcp.knowledge import KnowledgeUnavailable, indexer
from vehicle_mcp.knowledge import embedder as emb
from vehicle_mcp.knowledge.chunking import (
    chunk_markdown,
    chunk_plain,
    first_heading,
    window_blocks,
)
from vehicle_mcp.knowledge.search import (
    MAX_SNIPPET_CHARS,
    KnowledgeSearch,
    locales_for,
    snippet,
)
from vehicle_mcp.knowledge.settings import (
    MODEL_FILES,
    MODEL_REVISION,
    ModelFile,
    default_db_path,
    default_model_dir,
)
from vehicle_mcp.knowledge.sources import (
    catalog_chunks,
    docs_chunks,
    pdf_chunks,
)
from vehicle_mcp.knowledge.store import (
    Chunk,
    create_index,
    fts_index_text,
    fts_query,
    open_index,
    query_tokens,
    stem_turkish,
)
from vehicle_mcp.matching import normalise as matching_fold
from vehicle_mcp.source import UnconfirmedRefused
from vehicle_mcp.tools import VehicleTools

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --- chunking ----------------------------------------------------------------------------------
def test_markdown_chunks_follow_headings_and_ignore_fenced_hashes() -> None:
    pieces = chunk_markdown(
        "# A\n\nintro\n\n## B\n\nbody\n\n```\n# not a heading\n```\n\n# C\n\nend"
    )
    sections = [p.section for p in pieces]
    assert sections == ["A", "A > B", "C"]
    assert "# not a heading" in pieces[1].text


def test_windows_respect_size_and_overlap() -> None:
    blocks = [" ".join(f"w{i}x{j}" for j in range(40)) for i in range(10)]
    windows = window_blocks(blocks, max_words=100, overlap=45)
    assert len(windows) > 3
    assert all(len(w.split()) <= 100 for w in windows)
    # Consecutive windows share a block (the overlap).
    assert windows[0].split("\n\n")[-1] == windows[1].split("\n\n")[0]


def test_a_single_long_block_is_split_by_words() -> None:
    windows = window_blocks([" ".join(str(i) for i in range(250))], max_words=100, overlap=20)
    assert [len(w.split()) for w in windows] == [100, 100, 90]


def test_plain_chunks_and_first_heading() -> None:
    assert chunk_plain("para one\n\npara two") == ["para one\n\npara two"]
    assert chunk_plain("") == []
    assert first_heading("text\n## sub\n# Title\n") == "Title"
    assert first_heading("```\n# no\n```\n") is None


# --- sources -----------------------------------------------------------------------------------
def test_displayable_entries_become_full_text_and_phrase_chunks() -> None:
    chunks = [c for c in catalog_chunks(catalog_payload()) if c.ref == "ENG-001"]
    kinds = sorted(c.kind for c in chunks)
    assert kinds == ["complaint"] * 4 + ["fault"]
    full = next(c for c in chunks if c.kind == "fault")
    assert "Valve clearance too large" in full.text and "tapping sound" in full.text
    assert {c.locale for c in chunks} == {"en", "tr"}


def test_unconfirmed_entries_index_only_phrases_and_needs() -> None:
    chunks = [c for c in catalog_chunks(catalog_payload()) if c.kind == "data_only"]
    assert len(chunks) == 2
    blob = json.dumps([c.__dict__ for c in chunks])
    assert "SECRET-WORDING" not in blob
    assert "CHG-009" not in blob
    assert all(c.display and "battery voltage" in c.display for c in chunks)
    assert all(c.display and "did.BATTERY_VOLTAGE" not in c.display for c in chunks)
    assert all(c.title == "unconfirmed catalog entry (data only)" for c in chunks)


def test_refused_unconfirmed_fetch_treats_every_entry_as_displayable() -> None:
    payload = {"entries": [{"id": "X-1", "subsystem": "s", "complaints": ["a b"]}]}
    assert [c.kind for c in catalog_chunks(payload, all_displayable=True)] == ["fault", "complaint"]
    assert catalog_chunks(payload) == []  # no `needs`: nothing to point at
    assert catalog_chunks("junk") == [] and catalog_chunks({"entries": [1, {"id": 2}]}) == []


def test_docs_chunks_skip_archive_and_keep_relative_paths(tmp_path: Path) -> None:
    root = write_docs(tmp_path / "docs")
    chunks = list(docs_chunks(root))
    assert chunks and all(c.url_or_path == "docs/cooling.md" for c in chunks)
    assert {c.title for c in chunks} == {"Cooling guide"}
    assert any(c.section == "Cooling guide > Fan" for c in chunks)
    with pytest.raises(KnowledgeUnavailable):
        list(docs_chunks(tmp_path / "missing"))


def test_pdf_chunks_are_user_local_with_page_sections(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "manual.pdf", ["tighten the drain bolt", "chain slack twenty mm"])
    chunks = list(pdf_chunks(pdf))
    assert [c.section for c in chunks] == ["p. 1", "p. 2"]
    assert {c.license for c in chunks} == {"user-local"}
    assert chunks[0].url_or_path == "local:manual.pdf" and "drain bolt" in chunks[0].text
    with pytest.raises(KnowledgeUnavailable):
        list(pdf_chunks(tmp_path / "nope.pdf"))


# --- store and hybrid search -------------------------------------------------------------------
def _built(tmp_path: Path) -> tuple[Path, FakeEmbedder]:
    db = tmp_path / "k.sqlite"
    fake = FakeEmbedder()
    index = create_index(db, dim=fake.dim, model_id=fake.model_id)
    chunks = [
        *catalog_chunks(catalog_payload()),
        *docs_chunks(write_docs(tmp_path / "docs")),
    ]
    index.add_chunks(chunks, indexer._embed_all(fake, chunks))
    index.close()
    return db, fake


def test_query_helpers_never_let_user_text_into_the_fts_syntax() -> None:
    assert fts_query('coolant" OR NOT (x*) NEAR') == '"coolant" OR "or" OR "not" OR "x" OR "near"'
    assert query_tokens("") == [] and fts_query("!!!") == ""


def test_hybrid_search_ranks_the_matching_fault_first(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    index = open_index(db, model_id=fake.model_id)
    hits = index.search("valve ticking", fake.embed_queries(["valve ticking"])[0], k=3)
    assert hits[0].chunk.ref == "ENG-001"
    assert hits[0].bm25_rank is not None and hits[0].vector_rank is not None
    refs = [h.chunk.ref or h.chunk_id for h in hits]
    assert len(set(refs)) == len(refs)  # one hit per group
    again = index.search("valve ticking", fake.embed_queries(["valve ticking"])[0], k=3)
    assert [(h.chunk_id, h.score) for h in again] == [(h.chunk_id, h.score) for h in hits]
    index.close()


def test_vector_only_match_crosses_languages(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    index = open_index(db, model_id=fake.model_id)
    query = "tıkırtı"  # no word of this appears in the English fault text
    hits = index.search(query, fake.embed_queries([query])[0], k=3, locales=["en", "und"])
    assert hits and hits[0].chunk.ref == "ENG-001" and hits[0].bm25_rank is None
    index.close()


def test_locale_filter_and_group_dedupe(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    index = open_index(db, model_id=fake.model_id)
    vec = fake.embed_queries(["akü"])[0]
    only_en = index.search("akü", vec, k=5, locales=["en"])
    assert all(h.chunk.locale == "en" for h in only_en)
    tr = index.search("akü sürekli", vec, k=5, locales=["tr"])
    assert tr and all(h.chunk.locale == "tr" for h in tr)
    assert index.search("zzzz", vec, k=5, locales=["fr"]) == []
    assert index.count() > 5 and "fault-catalog" in index.counts_by_source()
    index.close()


def test_delete_source_removes_text_and_vectors(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    index = open_index(db, model_id=fake.model_id)
    before = index.count()
    removed = index.delete_source("docs")
    assert removed and index.count() == before - removed
    assert index.delete_where_source_like("fault") > 0 and index.count() == 0
    assert index.search("fan", fake.embed_queries(["fan"])[0]) == []
    index.close()


def test_index_rejects_wrong_dimension_and_other_models(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    with pytest.raises(KnowledgeUnavailable, match="another embedding model"):
        open_index(db, model_id="other@1")
    with pytest.raises(KnowledgeUnavailable, match="another embedding model"):
        create_index(db, dim=DIM, model_id="other@1")
    index = create_index(db, dim=fake.dim, model_id=fake.model_id)
    with pytest.raises(ValueError, match="dimensions"):
        index.add_chunks([Chunk("s", "t", "x")], [[0.0] * 3])
    with pytest.raises(ValueError, match="differ"):
        index.add_chunks([Chunk("s", "t", "x")], [])
    index.close()


def test_open_index_rejects_missing_and_foreign_files(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeUnavailable, match="no knowledge index"):
        open_index(tmp_path / "none.sqlite", model_id="x")
    junk = tmp_path / "junk.sqlite"
    junk.write_bytes(b"not a database at all" * 20)
    with pytest.raises(KnowledgeUnavailable, match="not a valid index"):
        open_index(junk, model_id="x")
    import sqlite3

    empty = tmp_path / "empty.sqlite"
    sqlite3.connect(empty).execute("CREATE TABLE meta(key, value)").connection.commit()
    with pytest.raises(KnowledgeUnavailable, match="not a valid index"):
        open_index(empty, model_id="x")


# --- the search backend ------------------------------------------------------------------------
def _searcher(db: Path, fake: FakeEmbedder) -> KnowledgeSearch:
    return KnowledgeSearch(db, embedder_factory=lambda: fake)


def test_search_result_shape_citations_and_note(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    result = _searcher(db, fake).search("why does coolant rise in traffic", "en", 3)
    assert result["available"] is True and result["kind"] == "knowledge_search"
    assert "not measurements" in result["note"]
    first = result["hits"][0]
    assert first["fault_id"] == "COO-001" and first["rank"] == 1
    assert set(first["citation"]) == {"source", "title", "section", "url_or_path"}
    assert first["citation"]["url_or_path"] == "faults/COO-001"
    doc = next(h for h in result["hits"] if h["kind"] == "doc")
    assert set(doc["snippet"]) == {"untrusted_text"}
    assert doc["citation"]["url_or_path"] == "docs/cooling.md"
    assert fake.prefixed[-1] == "query: why does coolant rise in traffic"


def test_turkish_query_finds_turkish_phrases_and_english_text(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    result = _searcher(db, fake).search("supaplardan tıkırtı sesi", "tr", 3)
    assert result["hits"][0]["fault_id"] == "ENG-001"
    assert {h["locale"] for h in result["hits"]} <= {"tr", "en", "und"}
    assert locales_for("tr") == ["tr", "und", "en"] and locales_for("EN") == ["en", "und"]
    assert locales_for("any") is None and locales_for("") is None


def test_unconfirmed_hits_never_carry_id_or_wording(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    result = _searcher(db, fake).search("akü sürekli bitiyor", "tr", 5)
    data_only = [h for h in result["hits"] if h["kind"] == "data_only"]
    assert data_only
    blob = json.dumps(result, ensure_ascii=False)
    assert "CHG-009" not in blob and "SECRET-WORDING" not in blob
    assert "fault_id" not in data_only[0]
    assert "battery voltage" in data_only[0]["snippet"]
    assert "did.BATTERY_VOLTAGE" not in data_only[0]["snippet"]


def test_snippets_are_at_most_300_characters(tmp_path: Path) -> None:
    long_text = "word " * 400 + "needle " + "tail " * 200
    assert len(snippet(long_text, "needle")) <= MAX_SNIPPET_CHARS
    assert snippet(long_text, "needle").startswith("...") and "needle" in snippet(
        long_text, "needle"
    )
    assert len(snippet(long_text, "absent")) <= MAX_SNIPPET_CHARS
    assert snippet("short\x00 text", "x") == "short text"
    tail = "a " * 20 + "needle " + "b " * 5
    assert snippet(tail * 8, "needle").endswith("b") or len(snippet(tail * 8, "needle")) <= 300

    db = tmp_path / "m.sqlite"
    fake = FakeEmbedder()
    index = create_index(db, dim=fake.dim, model_id=fake.model_id)
    chunk = Chunk("manual:x", "Manual", long_text, license="user-local", kind="manual")
    index.add_chunks([chunk], fake.embed_passages([chunk.text]))
    index.close()
    hit = _searcher(db, fake).search("needle", "en", 1)["hits"][0]
    assert hit["license"] == "user-local"
    assert len(hit["snippet"]["untrusted_text"]) <= MAX_SNIPPET_CHARS


def test_search_without_index_extra_or_model_is_unavailable_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeEmbedder()
    missing = _searcher(tmp_path / "none.sqlite", fake).search("fan")
    assert missing["available"] is False and "no knowledge index" in missing["reason"]

    db, _ = _built(tmp_path)
    monkeypatch.setitem(sys.modules, "sqlite_vec", None)
    no_extra = _searcher(db, fake).search("fan")
    assert no_extra["available"] is False and "'rag' extra" in no_extra["reason"]
    monkeypatch.undo()

    no_model = KnowledgeSearch(db, env={"VEHICLE_MCP_MODEL_DIR": str(tmp_path / "empty")})
    reason = no_model.search("fan")["reason"]
    assert "fetch-model" in reason

    wrong = KnowledgeSearch(db, embedder_factory=lambda: _Other())
    assert "another embedding model" in wrong.search("fan")["reason"]


class _Other(FakeEmbedder):
    model_id = "other@2"


def test_empty_result_has_a_reason(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    result = _searcher(db, fake).search("zzz", "fr", 3)
    assert result["available"] is True and result["hits"] != [] or "reason" in result
    index = create_index(tmp_path / "e.sqlite", dim=fake.dim, model_id=fake.model_id)
    index.close()
    empty = _searcher(tmp_path / "e.sqlite", fake).search("zzz", "en", 3)
    assert empty["hits"] == [] and "no passage" in empty["reason"]


def test_default_paths_follow_the_environment(tmp_path: Path) -> None:
    assert default_db_path({"VEHICLE_MCP_KNOWLEDGE_DB": "~/k.sqlite"}).name == "k.sqlite"
    assert default_db_path({"XDG_DATA_HOME": str(tmp_path)}) == (
        tmp_path / "vehicle-mcp" / "knowledge.sqlite"
    )
    assert default_db_path({}).name == "knowledge.sqlite"
    path = default_model_dir({"VEHICLE_MCP_MODEL_DIR": str(tmp_path)})
    assert path == tmp_path / "multilingual-e5-small" / MODEL_REVISION
    assert default_model_dir({"XDG_CACHE_HOME": str(tmp_path)}).is_relative_to(tmp_path)


# --- the tool ----------------------------------------------------------------------------------
def test_tool_clamps_k_validates_query_and_reports_unconfigured(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    tools = VehicleTools(FakeVehicleServer(), _searcher(db, fake))  # type: ignore[arg-type]
    assert len(tools.search_knowledge("fan coolant traffic", "en", 99)["hits"]) <= 10
    assert len(tools.search_knowledge("fan coolant traffic", "en", 0)["hits"]) == 1
    with pytest.raises(ToolError):
        tools.search_knowledge("   ")
    bare = VehicleTools(FakeVehicleServer())  # type: ignore[arg-type]
    result = bare.search_knowledge("fan")
    assert result["available"] is False and "not configured" in result["reason"]


# --- the indexer CLI ---------------------------------------------------------------------------
class _CatalogSource:
    def __init__(self, refuse: bool = False) -> None:
        self.refuse = refuse
        self.calls: list[bool] = []
        self.closed = False

    def fault_catalog(self, *, subsystem: str | None, include_unconfirmed: bool = False) -> Any:
        self.calls.append(include_unconfirmed)
        if include_unconfirmed and self.refuse:
            raise UnconfirmedRefused("refused")
        payload = catalog_payload()
        if not include_unconfirmed:
            payload["entries"] = [e for e in payload["entries"] if e.get("displayable")]
        return payload

    def close(self) -> None:
        self.closed = True


def _run(argv: list[str], source: _CatalogSource | None = None) -> int:
    return indexer.main(
        argv,
        embedder=FakeEmbedder(),
        client_factory=(lambda: source) if source is not None else None,  # type: ignore[arg-type, return-value]
    )


def test_build_indexes_catalog_docs_and_a_local_pdf(tmp_path: Path) -> None:
    db = tmp_path / "idx" / "k.sqlite"
    docs = write_docs(tmp_path / "docs")
    pdf = make_pdf(tmp_path / "my-manual.pdf", ["torque the axle nut to the stated value"])
    source = _CatalogSource()
    code = _run(["build", "--db", str(db), "--docs", str(docs), "--local-pdf", str(pdf)], source)
    assert code == 0 and source.calls == [True] and source.closed
    index = open_index(db, model_id=FakeEmbedder.model_id)
    counts = index.counts_by_source()
    assert set(counts) == {"fault-catalog", "docs", "manual:my-manual"}
    index.close()
    hit = _searcher(db, FakeEmbedder()).search("axle nut torque", "en", 1)["hits"][0]
    assert hit["license"] == "user-local" and hit["citation"]["section"] == "p. 1"
    # A rebuild with the same sources is idempotent.
    assert _run(["build", "--db", str(db), "--docs", str(docs)], _CatalogSource()) == 0
    index = open_index(db, model_id=FakeEmbedder.model_id)
    assert index.counts_by_source() == counts
    index.close()
    assert _run(["build", "--db", str(db), "--rebuild", "--no-catalog", "--docs", str(docs)]) == 0
    index = open_index(db, model_id=FakeEmbedder.model_id)
    assert set(index.counts_by_source()) == {"docs"}
    index.close()


def test_build_falls_back_when_the_server_refuses_unconfirmed(tmp_path: Path) -> None:
    db = tmp_path / "k.sqlite"
    source = _CatalogSource(refuse=True)
    assert _run(["build", "--db", str(db)], source) == 0
    assert source.calls == [True, False]
    index = open_index(db, model_id=FakeEmbedder.model_id)
    assert not any(k for k in index.counts_by_source() if k.startswith("manual"))
    index.close()


def test_build_refuses_a_pdf_import_into_an_unignored_git_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    pdf = make_pdf(tmp_path / "m.pdf", ["secret manual text"])
    db = repo / "knowledge.sqlite"
    assert _run(["build", "--db", str(db), "--no-catalog", "--local-pdf", str(pdf)]) == 2
    assert not db.exists()
    (repo / ".gitignore").write_text("*.sqlite\n")
    assert _run(["build", "--db", str(db), "--no-catalog", "--local-pdf", str(pdf)]) == 0


def test_cli_reports_errors_as_exit_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(["build", "--db", str(tmp_path / "k.sqlite"), "--no-catalog", "--docs", "nope"])
    assert code == 2 and "does not exist" in capsys.readouterr().err
    assert (
        _run(["build", "--db", str(tmp_path / "k.sqlite"), "--no-catalog", "--local-pdf", "no.pdf"])
        == 2
    )


def test_build_without_a_token_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("VEHICLE_API_TOKEN", raising=False)
    code = indexer.main(["build", "--db", str(tmp_path / "k.sqlite")], embedder=FakeEmbedder())
    assert code == 2 and "VEHICLE_API_TOKEN" in capsys.readouterr().err


# --- model download and the ONNX embedder ------------------------------------------------------
def _fake_model_files(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    blobs = {"model.onnx": b"onnx-bytes" * 10, "tokenizer.json": b"{}"}
    files = tuple(
        ModelFile(f"onnx/{name}", name, len(data), hashlib.sha256(data).hexdigest())
        for name, data in blobs.items()
    )
    monkeypatch.setattr(emb, "MODEL_FILES", files)
    return blobs


def test_fetch_model_downloads_verifies_and_skips_present_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blobs = _fake_model_files(monkeypatch)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=blobs[request.url.path.rsplit("/", 1)[1]])

    target = tmp_path / "m"
    emb.fetch_model(target, transport=httpx.MockTransport(handler))
    assert emb.model_present(target) and emb.verify_model(target) == []
    assert all(MODEL_REVISION in url and "intfloat/multilingual-e5-small" in url for url in seen)
    emb.fetch_model(target, transport=httpx.MockTransport(handler))
    assert len(seen) == 2  # nothing downloaded twice
    (target / "model.onnx").write_bytes(b"x" * len(blobs["model.onnx"]))
    assert emb.verify_model(target) == ["model.onnx"]


def test_fetch_model_rejects_a_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_model_files(monkeypatch)
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"tampered"))
    with pytest.raises(KnowledgeUnavailable, match="sha256"):
        emb.fetch_model(tmp_path / "m", transport=transport)
    assert not list((tmp_path / "m").glob("*"))


def test_cli_fetch_model_with_verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blobs = _fake_model_files(monkeypatch)
    real_client = httpx.Client
    monkeypatch.setattr(
        emb.httpx,
        "Client",
        lambda **kw: real_client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, content=blobs[r.url.path.rsplit("/", 1)[1]])
            )
        ),
    )
    assert indexer.main(["fetch-model", "--verify", "--model-dir", str(tmp_path / "m")]) == 0
    (tmp_path / "m" / "model.onnx").write_bytes(b"y" * len(blobs["model.onnx"]))
    monkeypatch.setattr(emb, "fetch_model", lambda *a, **k: tmp_path / "m")
    monkeypatch.setattr(indexer, "fetch_model", lambda *a, **k: tmp_path / "m")
    assert indexer.main(["fetch-model", "--verify", "--model-dir", str(tmp_path / "m")]) == 1


def test_onnx_embedder_needs_the_model_files(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeUnavailable, match="fetch-model"):
        emb.OnnxE5Embedder(tmp_path)


class _Enc:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        self.attention_mask = [1] * len(ids)


class _Tokenizer:
    def __init__(self) -> None:
        self.max_tokens = 0
        self.padded = False

    def enable_truncation(self, n: int) -> None:
        self.max_tokens = n

    def enable_padding(self) -> None:
        self.padded = True

    def encode_batch(self, texts: list[str]) -> list[_Enc]:
        width = max(len(t.split()) for t in texts)
        return [
            _Enc([5 + i for i in range(len(t.split()))] + [0] * (width - len(t.split())))
            for t in texts
        ]


class _Session:
    def get_inputs(self) -> list[Any]:
        return [
            type("I", (), {"name": n})() for n in ("input_ids", "attention_mask", "token_type_ids")
        ]

    def run(self, _outputs: Any, feeds: dict[str, Any]) -> list[Any]:
        ids = feeds["input_ids"]
        assert feeds["token_type_ids"].shape == ids.shape
        hidden = np.stack([ids, ids + 1.0, np.ones_like(ids)], axis=-1).astype(np.float32)
        return [hidden]


def test_onnx_embedder_prefixes_pools_and_normalises() -> None:
    e = object.__new__(emb.OnnxE5Embedder)
    e._np = np
    e._session = _Session()
    e._input_names = {"input_ids", "attention_mask", "token_type_ids"}
    e._tokenizer = _Tokenizer()
    vectors = e.embed_queries(["a b", "a b c d"])
    assert e._tokenizer.max_tokens == emb.QUERY_MAX_TOKENS and e._tokenizer.padded
    assert len(vectors) == 2 and all(abs(np.linalg.norm(v) - 1.0) < 1e-6 for v in vectors)
    e.embed_passages(["a"] * 20)  # more than one batch
    assert e._tokenizer.max_tokens == emb.PASSAGE_MAX_TOKENS
    assert abs(sum(x * x for x in emb.normalise([3.0, 4.0])) - 1.0) < 1e-9
    assert emb.normalise([0.0, 0.0]) == [0.0, 0.0]


# --- the real model (optional) -----------------------------------------------------------------
@pytest.mark.slow
def test_real_model_retrieves_across_languages(tmp_path: Path) -> None:
    """Needs the downloaded model (`vehicle-mcp-index fetch-model`); skipped by default."""
    directory = default_model_dir()
    if not emb.model_present(directory):
        pytest.skip("model not downloaded")
    real = emb.OnnxE5Embedder(directory)
    db = tmp_path / "real.sqlite"
    index = create_index(db, dim=real.dim, model_id=real.model_id)
    chunks = catalog_chunks(catalog_payload())
    index.add_chunks(chunks, indexer._embed_all(real, chunks))
    index.close()
    result = KnowledgeSearch(db, embedder_factory=lambda: real).search(
        "supaplardan tıkırtı sesi", "tr", 3
    )
    assert result["hits"][0]["fault_id"] == "ENG-001"
    assert MODEL_FILES  # pinned


# --- G2-2 retrieval quality: Turkish stemming, synonyms, cutoffs, eval CLI --------------------
def test_turkish_stemming_and_fts_expansion() -> None:
    assert stem_turkish("supaplardan") == "supap"
    assert stem_turkish("amortisörden") == "amortisor"
    assert stem_turkish("akü") == "aku"  # short root preserved
    # words that merely end like a suffix must stay intact (one-letter suffixes, short stems)
    for word in ("lamba", "motor", "yağ", "fren", "zincir", "kapak", "sele", "aku", "ısı"):
        assert stem_turkish(word) == matching_fold(word), word
    assert stem_turkish("lambalar") == "lamba"  # one pass, stem stays long enough
    assert stem_turkish("supaplarından") == "supaplari"  # no repeated over-stripping
    indexed = fts_index_text("supaplar ses yapıyor", "tr")
    assert "supap" in indexed
    assert fts_index_text("valves ticking", "en") == "valves ticking"
    q_tr = fts_query(
        "supaplardan tıkırtı sesi ile",
        locales=["tr", "und", "en"],
        stem_tr=True,
        expand_synonyms=True,
        drop_stopwords=True,
    )
    assert '"supap"' in q_tr and '"ile"' not in q_tr
    q_en = fts_query(
        "battery is dying",
        locales=["en", "und"],
        stem_tr=False,
        expand_synonyms=True,
        drop_stopwords=True,
    )
    assert '"flat"' in q_en and '"drained"' in q_en and '"is"' not in q_en


def test_vector_floor_suppresses_weak_vector_hits(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    idx = open_index(db, model_id=fake.model_id)
    args = ("zzzzunrelated", fake.embed_queries(["zzzzunrelated"])[0])
    # no lexical match and a high cosine floor: nothing survives
    assert idx.search(*args, k=5, locales=["en", "und"], min_similarity=0.95) == []
    # without the floor the weakest vector neighbours are returned
    assert idx.search(*args, k=5, locales=["en", "und"], min_similarity=0.0)
    idx.close()


def test_single_list_hits_survive_without_a_relative_rrf_cutoff(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    idx = open_index(db, model_id=fake.model_id)
    # vector-only (cross-lingual) hit at rank 1: its fused score is half of a both-lists hit
    vec_only = idx.search("tıkırtı", fake.embed_queries(["tıkırtı"])[0], k=3, locales=["en"])
    assert vec_only and vec_only[0].chunk.ref == "ENG-001"
    assert vec_only[0].bm25_rank is None and vec_only[0].vector_rank == 1
    # lexical-only hit (the vector floor rejects everything): kept inside the top-N lexical
    query = "radiator"
    lex = idx.search(query, fake.embed_queries([query])[0], k=3, locales=["en"], min_similarity=1.1)
    assert lex and lex[0].vector_rank is None and lex[0].bm25_rank == 1
    # lexical_top_n = 0 turns the lexical path off
    none = idx.search(
        query,
        fake.embed_queries([query])[0],
        k=3,
        locales=["en"],
        min_similarity=1.1,
        lexical_top_n=0,
    )
    assert none == []
    idx.close()


def test_group_level_fusion_combines_ranks_across_chunks(tmp_path: Path) -> None:
    db = tmp_path / "g.sqlite"
    fake = FakeEmbedder()
    index = create_index(db, dim=fake.dim, model_id=fake.model_id)
    chunks = [
        Chunk(source="s", title="A", text="alpha ticking noise", ref="F-1", kind="complaint"),
        Chunk(source="s", title="A", text="engine tap clatter", ref="F-1", kind="complaint"),
        Chunk(source="s", title="B", text="alpha filler words", ref="F-2", kind="complaint"),
    ]
    index.add_chunks(chunks, indexer._embed_all(fake, chunks))
    # BM25 best chunk of F-1 is "alpha ticking noise"; the vector best chunk is the other one
    query = "alpha tap"
    hits = index.search(query, fake.embed_queries([query])[0], k=3)
    by_ref = {h.chunk.ref: h for h in hits}
    assert by_ref["F-1"].bm25_rank is not None and by_ref["F-1"].vector_rank is not None
    assert by_ref["F-1"].score > by_ref["F-2"].score
    assert len(by_ref) == len(hits)  # one hit per ref
    index.close()


def test_open_index_rejects_another_schema_version(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    index = open_index(db, model_id=fake.model_id)
    index.set_meta("schema_version", "1")
    index.close()
    with pytest.raises(KnowledgeUnavailable, match="rebuild it with"):
        open_index(db, model_id=fake.model_id)
    result = _searcher(db, fake).search("valve")
    assert result["available"] is False and "schema" in result["reason"]


class _StrictEmbedder(FakeEmbedder):
    min_similarity = 1.1  # nothing is similar enough


def test_embedder_similarity_floor_applies_through_the_normal_path(tmp_path: Path) -> None:
    db, fake = _built(tmp_path)
    assert emb.E5_MIN_COSINE == emb.OnnxE5Embedder.min_similarity
    strict = KnowledgeSearch(db, embedder_factory=lambda: _StrictEmbedder(), lexical_top_n=0)
    assert strict.search("zzzzunrelated")["hits"] == []
    open_floor = KnowledgeSearch(db, embedder_factory=lambda: fake, lexical_top_n=0)
    assert open_floor.search("tıkırtı", "tr")["hits"]


def test_query_stemming_only_for_the_tr_locale() -> None:
    kwargs = {"expand_synonyms": False, "drop_stopwords": False}
    assert '"supap"' in fts_query("supaplardan", stem_tr=True, **kwargs)
    assert '"supap"' not in fts_query("supaplardan", locales=["tr", "en"], stem_tr=False, **kwargs)


def test_exclude_holdout_removes_the_held_out_phrases(tmp_path: Path) -> None:
    from vehicle_mcp.knowledge import eval_retrieval

    qdir = tmp_path / "eval"
    (qdir / "locales").mkdir(parents=True)
    held = {"id": "h", "query": "tapping sound at idle", "expected": ["ENG-001"]}
    (qdir / "queries.json").write_text(
        json.dumps({"locale": "en", "queries": [{**held, "split": "holdout"}]}), encoding="utf-8"
    )
    (qdir / "locales" / "tr.json").write_text(
        json.dumps({"locale": "tr", "queries": []}), encoding="utf-8"
    )
    chunks = catalog_chunks(catalog_payload())
    kept = eval_retrieval.without_holdout(chunks, qdir, ("en", "tr"))
    assert not [c for c in kept if c.kind == "complaint" and c.text == held["query"]]
    fault = next(c for c in kept if c.kind == "fault" and c.ref == "ENG-001")
    assert "tapping sound" not in fault.text and "ticking noise from the engine" in fault.text
    assert len(kept) == len(chunks) - 1


def test_retrieval_eval_runner_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from vehicle_mcp.knowledge import eval_retrieval

    db, fake = _built(tmp_path)
    qdir = tmp_path / "eval"
    (qdir / "locales").mkdir(parents=True)
    (qdir / "queries.json").write_text(
        json.dumps(
            {
                "locale": "en",
                "queries": [
                    {"id": "q1", "query": "valve ticking", "expected": ["ENG-001"]},
                    {
                        "id": "q2",
                        "query": "radiator fan coolant traffic",
                        "expected": ["docs/cooling.md#Fan"],
                    },
                    {"id": "q3", "query": "zzz nonexistent", "expected": ["NO-SUCH"]},
                    {
                        "id": "h1",
                        "query": "ticking noise from the engine",
                        "expected": ["ENG-001"],
                        "split": "holdout",
                    },
                ],
                "weak_queries": [{"id": "w1", "query": "zzz"}],
            }
        ),
        encoding="utf-8",
    )
    (qdir / "locales" / "tr.json").write_text(
        json.dumps(
            {
                "locale": "tr",
                "queries": [{"id": "t1", "query": "supaplardan tıkırtı", "expected": ["ENG-001"]}],
                "weak_queries": [],
            }
        ),
        encoding="utf-8",
    )
    reports = eval_retrieval.evaluate_all(db, qdir, ["en", "tr"], embedder=fake, k=3)
    assert [(r.locale, r.split) for r in reports] == [
        ("en", "tune"),
        ("en", "holdout"),
        ("tr", "tune"),
    ]
    en_rep, en_hold, tr_rep = reports
    assert en_hold.queries == 1 and en_hold.recall_at_1 == 1.0
    assert en_rep.queries == 3 and en_rep.recall_at_1 > 0 and en_rep.misses
    assert tr_rep.queries == 1 and tr_rep.recall_at_1 == 1.0
    assert "R@1" in eval_retrieval.format_table(reports)
    parsed = json.loads(eval_retrieval.report_to_json(reports))
    assert [r["locale"] for r in parsed] == ["en", "en", "tr"]
    with pytest.raises(ValueError, match="split"):
        eval_retrieval._split_of({"id": "x", "split": "dev"})

    # CLI eval text + misses + JSON + missing dir error
    assert _run(["eval", "--db", str(db), "--queries-dir", str(qdir), "--show-misses"]) == 0
    out = capsys.readouterr().out
    assert "R@1" in out and "non-rank-1" in out and "holdout" in out
    assert _run(["eval", "--db", str(db), "--queries-dir", str(qdir), "--split", "holdout"]) == 0
    assert "tune" not in capsys.readouterr().out.split("\n", 2)[2]
    assert _run(["eval", "--db", str(db), "--queries-dir", str(qdir), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["locale"] == "en"
    assert _run(["eval", "--db", str(db), "--queries-dir", str(tmp_path / "missing")]) == 2


def test_exclude_holdout_also_strips_test_split_phrases(tmp_path: Path) -> None:
    from vehicle_mcp.knowledge import eval_retrieval

    qdir = tmp_path / "eval"
    (qdir / "locales").mkdir(parents=True)
    held = {"id": "t-1", "query": "tapping sound at idle", "expected": ["ENG-001"], "split": "test"}
    (qdir / "queries.json").write_text(
        json.dumps({"locale": "en", "queries": [held]}), encoding="utf-8"
    )
    (qdir / "locales" / "tr.json").write_text(
        json.dumps({"locale": "tr", "queries": []}), encoding="utf-8"
    )
    chunks = catalog_chunks(catalog_payload())
    kept = eval_retrieval.without_holdout(chunks, qdir, ("en", "tr"))
    assert not [c for c in kept if c.kind == "complaint" and c.text == held["query"]]


def test_shipped_retrieval_eval_dataset_integrity_and_split() -> None:
    from vehicle_mcp.knowledge import eval_retrieval

    en_q, en_w = eval_retrieval.load_query_file(eval_retrieval.DEFAULT_QUERIES_DIR, "en")
    tr_q, tr_w = eval_retrieval.load_query_file(eval_retrieval.DEFAULT_QUERIES_DIR, "tr")
    assert len(en_q) == len(tr_q) >= 164
    assert len(en_w) == len(tr_w) >= 12
    assert len({q.id for q in en_q}) == len(en_q)
    assert len({q.id for q in tr_q}) == len(tr_q)
    assert len({q.query.strip().lower() for q in en_q}) == len(en_q)
    assert len({q.query.strip().lower() for q in tr_q}) == len(tr_q)
    assert all(q.query.isascii() for q in en_q)
    assert [q.expected for q in en_q] == [q.expected for q in tr_q]
    assert [q.split for q in en_q] == [q.split for q in tr_q]
    assert [q.split for q in en_w] == [q.split for q in tr_w]
    assert {q.split for q in en_q} == {"tune", "test", "holdout"}
    assert {q.split for q in en_w} == {"tune", "test"}
    # `test` holds only new G4-7 items: nothing the shipped parameters were chosen on
    assert all("-g4-" in q.id for q in en_q + tr_q if q.split == "test")
    assert all("-g4-" in q.id for q in en_w + tr_w if q.split == "test")
    assert sum(1 for q in en_q if q.split == "test") >= 30
    assert sum(1 for q in en_q if q.split == "holdout") == 62
    assert all(q.expected for q in en_q)
