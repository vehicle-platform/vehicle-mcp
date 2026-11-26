"""Ride LLM knowledge layer (T60, D-132 item 2c): hybrid retrieval with citations.

An optional extra (`pip install 'vehicle-mcp[rag]'`). One SQLite file holds the text chunks, an
FTS5 (BM25) table and a sqlite-vec table; `search_knowledge` fuses both rankings with
Reciprocal Rank Fusion. The core server never imports this package's heavy dependencies unless
a search is made.
"""

from __future__ import annotations


class KnowledgeUnavailable(RuntimeError):
    """The knowledge layer cannot answer; the message is a safe, human-readable reason."""
