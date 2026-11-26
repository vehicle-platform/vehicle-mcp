"""Embedding runtime: multilingual-e5-small on ONNX Runtime, no torch.

e5 needs the `query: ` / `passage: ` prefixes (applied here, never by the caller). The model
and tokenizer are downloaded once into a cache directory outside the repo at a pinned revision
and verified against pinned sha256 sums; nothing is fetched at query time.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import httpx

from . import KnowledgeUnavailable
from .settings import (
    MODEL_DIM,
    MODEL_FILES,
    MODEL_ID,
    MODEL_NAME,
    MODEL_REVISION,
    MODEL_URL,
    ModelFile,
    default_model_dir,
)

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
QUERY_MAX_TOKENS = 128
PASSAGE_MAX_TOKENS = 512
BATCH_SIZE = 16
# Cosine floor for multilingual-e5-small. e5 similarities live in a narrow 0.80-0.90 band, so
# an absolute floor is only meaningful per model. Calibrated with `vehicle-mcp-index eval`
# (grid in tests/retrieval_eval/tuning.json): chosen on the `tune` split, then vetoed by the
# `holdout` split, which showed that floors above 0.83 cost more than 0.05 holdout recall@5 for
# little extra noise reduction. Re-calibrate when the model revision changes.
E5_MIN_COSINE = 0.83


class Embedder(Protocol):
    """What the index needs: an identity, a dimension and two embedding entry points."""

    model_id: str
    dim: int
    min_similarity: float  # cosine floor below which a vector hit is noise for this model

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_present(model_dir: Path) -> bool:
    """Cheap check (exact file sizes); `verify_model` re-hashes."""
    return all(
        (model_dir / f.local).is_file() and (model_dir / f.local).stat().st_size == f.size
        for f in MODEL_FILES
    )


def verify_model(model_dir: Path) -> list[str]:
    """Names of files that are missing or do not match the pinned sha256."""
    bad = []
    for f in MODEL_FILES:
        path = model_dir / f.local
        if not path.is_file() or sha256_of(path) != f.sha256:
            bad.append(f.local)
    return bad


def _download(file: ModelFile, target: Path, client: httpx.Client) -> None:
    url = MODEL_URL.format(name=MODEL_NAME, revision=MODEL_REVISION, path=file.remote)
    partial = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        with partial.open("wb") as out:
            for block in response.iter_bytes(1 << 20):
                digest.update(block)
                out.write(block)
    if digest.hexdigest() != file.sha256:
        partial.unlink(missing_ok=True)
        raise KnowledgeUnavailable(f"downloaded {file.local} does not match the pinned sha256")
    partial.replace(target)


def fetch_model(
    model_dir: Path | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    log: Callable[[str], None] = lambda _msg: None,
) -> Path:
    """Download the pinned model files that are missing; returns the model directory."""
    target_dir = model_dir or default_model_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(transport=transport, timeout=httpx.Timeout(60.0)) as client:
        for file in MODEL_FILES:
            target = target_dir / file.local
            if target.is_file() and target.stat().st_size == file.size:
                continue
            log(f"downloading {file.local} ({file.size // 1_000_000} MB)")
            _download(file, target, client)
    return target_dir


class OnnxE5Embedder:
    """multilingual-e5-small (384-dim, mean pooling, L2-normalised) on CPU."""

    model_id = MODEL_ID
    dim = MODEL_DIM
    min_similarity = E5_MIN_COSINE

    def __init__(self, model_dir: Path | None = None, threads: int | None = None) -> None:
        directory = model_dir or default_model_dir()
        if not model_present(directory):
            raise KnowledgeUnavailable(
                "the embedding model is not downloaded; run `vehicle-mcp-index fetch-model`"
            )
        try:
            import numpy
            import onnxruntime
            from tokenizers import Tokenizer
        except ImportError:
            raise KnowledgeUnavailable(
                "the optional 'rag' extra is not installed (pip install 'vehicle-mcp[rag]')"
            ) from None
        self._np: Any = numpy
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        self._session = onnxruntime.InferenceSession(
            str(directory / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([QUERY_PREFIX + t for t in texts], QUERY_MAX_TOKENS)

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([PASSAGE_PREFIX + t for t in texts], PASSAGE_MAX_TOKENS)

    def _embed(self, texts: Sequence[str], max_tokens: int) -> list[list[float]]:
        out: list[list[float]] = []
        self._tokenizer.enable_truncation(max_tokens)
        self._tokenizer.enable_padding()
        for start in range(0, len(texts), BATCH_SIZE):
            out.extend(self._embed_batch(list(texts[start : start + BATCH_SIZE])))
        return out

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        np = self._np
        encoded = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feeds)[0]
        pooled = (hidden * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
        pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return [[float(x) for x in row] for row in pooled]


def normalise(vector: Sequence[float]) -> list[float]:
    """L2-normalise (used by test embedders)."""
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]
