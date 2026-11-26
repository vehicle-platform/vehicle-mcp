"""CI guard (T60, D-033): no manual, PDF, index, model weight or `manuals/` path may be tracked.

The repo is public. A rider's own manual lives only in a local, git-ignored index. This test
fails when git tracks any such file; the secret scan (gitleaks) runs separately.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = (".pdf", ".sqlite", ".sqlite3", ".db", ".onnx", ".gguf", ".safetensors")


def forbidden(paths: list[str]) -> list[str]:
    """Tracked paths that must never be committed."""
    bad = []
    for path in paths:
        parts = [p.lower() for p in Path(path).parts]
        if path.lower().endswith(FORBIDDEN_SUFFIXES) or "manuals" in parts[:-1]:
            bad.append(path)
    return bad


def test_forbidden_matches_documents_models_and_indexes() -> None:
    sample = ["a/b.PDF", "manuals/x.txt", "docs/Manuals/y.md", "idx/k.sqlite", "w.onnx", "ok.md"]
    assert forbidden(sample) == sample[:5]


def test_git_tracks_no_manuals_pdfs_indexes_or_weights() -> None:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    listing = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert forbidden(listing) == [], "manuals, PDFs, indexes or model weights are tracked"
