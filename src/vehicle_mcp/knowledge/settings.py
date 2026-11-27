"""Paths and the pinned embedding model for the knowledge layer."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DB_ENV = "VEHICLE_MCP_KNOWLEDGE_DB"
MODEL_DIR_ENV = "VEHICLE_MCP_MODEL_DIR"
RRF_K_ENV = "VEHICLE_MCP_RAG_RRF_K"
FTS_WEIGHT_ENV = "VEHICLE_MCP_RAG_FTS_WEIGHT"
VECTOR_WEIGHT_ENV = "VEHICLE_MCP_RAG_VECTOR_WEIGHT"
FTS_POOL_ENV = "VEHICLE_MCP_RAG_FTS_POOL"
VECTOR_POOL_MIN_ENV = "VEHICLE_MCP_RAG_VECTOR_POOL_MIN"
LEXICAL_TOP_N_ENV = "VEHICLE_MCP_RAG_LEXICAL_TOP_N"
MIN_SIMILARITY_ENV = "VEHICLE_MCP_RAG_MIN_SIMILARITY"
USE_PREFIXES_ENV = "VEHICLE_MCP_RAG_USE_PREFIXES"

DEFAULT_RRF_K = 60
DEFAULT_FTS_WEIGHT = 1.0
DEFAULT_VECTOR_WEIGHT = 1.0
DEFAULT_FTS_POOL = 50
DEFAULT_VECTOR_POOL_MIN = 30
DEFAULT_LEXICAL_TOP_N = 2
DEFAULT_USE_PREFIXES = True

MODEL_NAME = "intfloat/multilingual-e5-small"  # MIT
# Exact revision of the model owner's own repository (the ONNX export lives there).
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_ID = f"{MODEL_NAME}@{MODEL_REVISION[:12]}"
MODEL_DIM = 384
MODEL_URL = "https://huggingface.co/{name}/resolve/{revision}/{path}"


@dataclass(frozen=True)
class ModelFile:
    remote: str
    local: str
    size: int
    sha256: str


MODEL_FILES = (
    ModelFile(
        "onnx/model.onnx",
        "model.onnx",
        470_268_510,
        "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
    ),
    ModelFile(
        "onnx/tokenizer.json",
        "tokenizer.json",
        17_082_730,
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    ),
)


def _home_dir(env: Mapping[str, str], var: str, fallback: str) -> Path:
    base = env.get(var, "").strip()
    return Path(base) if base else Path.home() / fallback


def default_db_path(env: Mapping[str, str] | None = None) -> Path:
    """The index file: `$VEHICLE_MCP_KNOWLEDGE_DB`, else under the user's data directory."""
    source = os.environ if env is None else env
    explicit = source.get(DB_ENV, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return _home_dir(source, "XDG_DATA_HOME", ".local/share") / "vehicle-mcp" / "knowledge.sqlite"


@dataclass(frozen=True)
class RetrievalSettings:
    """Tuneable parameters of existing hybrid retrieval (defaults reproduce baseline exactly)."""

    rrf_k: int = DEFAULT_RRF_K
    fts_weight: float = DEFAULT_FTS_WEIGHT
    vector_weight: float = DEFAULT_VECTOR_WEIGHT
    fts_pool: int = DEFAULT_FTS_POOL
    vector_pool_min: int = DEFAULT_VECTOR_POOL_MIN
    lexical_top_n: int = DEFAULT_LEXICAL_TOP_N
    min_similarity: float | None = None
    use_prefixes: bool = DEFAULT_USE_PREFIXES

    def __post_init__(self) -> None:
        if isinstance(self.rrf_k, bool) or self.rrf_k <= 0:
            raise ValueError(f"rrf_k must be a positive integer, got {self.rrf_k!r}")
        for name, val in (
            ("fts_weight", self.fts_weight),
            ("vector_weight", self.vector_weight),
        ):
            if isinstance(val, bool) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be a finite number >= 0, got {val!r}")
        for name, count in (
            ("fts_pool", self.fts_pool),
            ("vector_pool_min", self.vector_pool_min),
            ("lexical_top_n", self.lexical_top_n),
        ):
            if isinstance(count, bool) or count < 0:
                raise ValueError(f"{name} must be an integer >= 0, got {count!r}")
        if self.min_similarity is not None:
            sim = self.min_similarity
            if isinstance(sim, bool) or not math.isfinite(sim) or not (-1.0 <= sim <= 1.0):
                raise ValueError(f"min_similarity must be a finite number in [-1, 1], got {sim!r}")


def _parse_int_env(source: Mapping[str, str], var: str, default: int) -> int:
    raw = source.get(var, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{var} must be an integer, got {raw!r}") from None


def _parse_float_env(source: Mapping[str, str], var: str, default: float) -> float:
    raw = source.get(var, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{var} must be a number, got {raw!r}") from None


def _parse_bool_env(source: Mapping[str, str], var: str, default: bool) -> bool:
    raw = source.get(var, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{var} must be a boolean (0/1/true/false), got {raw!r}")


def load_retrieval_settings(env: Mapping[str, str] | None = None) -> RetrievalSettings:
    """Read `RetrievalSettings` from `env` (or `os.environ`), validating every override."""
    source = os.environ if env is None else env
    raw_min_sim = source.get(MIN_SIMILARITY_ENV, "").strip()
    min_sim: float | None = None
    if raw_min_sim:
        min_sim = _parse_float_env(source, MIN_SIMILARITY_ENV, 0.0)
    return RetrievalSettings(
        rrf_k=_parse_int_env(source, RRF_K_ENV, DEFAULT_RRF_K),
        fts_weight=_parse_float_env(source, FTS_WEIGHT_ENV, DEFAULT_FTS_WEIGHT),
        vector_weight=_parse_float_env(source, VECTOR_WEIGHT_ENV, DEFAULT_VECTOR_WEIGHT),
        fts_pool=_parse_int_env(source, FTS_POOL_ENV, DEFAULT_FTS_POOL),
        vector_pool_min=_parse_int_env(source, VECTOR_POOL_MIN_ENV, DEFAULT_VECTOR_POOL_MIN),
        lexical_top_n=_parse_int_env(source, LEXICAL_TOP_N_ENV, DEFAULT_LEXICAL_TOP_N),
        min_similarity=min_sim,
        use_prefixes=_parse_bool_env(source, USE_PREFIXES_ENV, DEFAULT_USE_PREFIXES),
    )


def default_model_dir(env: Mapping[str, str] | None = None) -> Path:
    """The model cache (outside every repo), one folder per pinned revision."""
    source = os.environ if env is None else env
    explicit = source.get(MODEL_DIR_ENV, "").strip()
    root = (
        Path(explicit).expanduser()
        if explicit
        else _home_dir(source, "XDG_CACHE_HOME", ".cache") / "vehicle-mcp" / "models"
    )
    return root / MODEL_NAME.split("/")[-1] / MODEL_REVISION
