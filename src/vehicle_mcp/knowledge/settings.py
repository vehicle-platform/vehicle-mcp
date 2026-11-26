"""Paths and the pinned embedding model for the knowledge layer."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DB_ENV = "VEHICLE_MCP_KNOWLEDGE_DB"
MODEL_DIR_ENV = "VEHICLE_MCP_MODEL_DIR"

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
