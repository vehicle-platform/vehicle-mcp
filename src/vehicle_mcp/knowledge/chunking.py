"""Text chunking: Markdown by heading, plain text by paragraph, both in overlapping windows.

Size is counted in words; 300 words is about 400 tokens for English text with the e5 tokenizer
(a rule of thumb, not a measurement). Tables and code blocks stay whole blocks unless a single
block exceeds the window, and each window repeats the tail of the previous one (overlap).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

MAX_WORDS = 300
OVERLAP_WORDS = 50
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Piece:
    section: str
    text: str


def _words(block: str) -> int:
    return len(block.split())


def _split_long(block: str, max_words: int, overlap: int) -> list[str]:
    words = block.split()
    if len(words) <= max_words:
        return [block]
    step = max(1, max_words - overlap)
    starts = range(0, len(words) - overlap, step)  # the last piece reaches the end
    return [" ".join(words[i : i + max_words]) for i in starts]


def window_blocks(
    blocks: list[str], max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS
) -> list[str]:
    """Greedy windows of whole blocks (long blocks are split by words), with block overlap."""
    flat: list[str] = []
    for block in blocks:
        flat.extend(_split_long(block, max_words, overlap))
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for block in flat:
        n = _words(block)
        if current and size + n > max_words:
            windows.append("\n\n".join(current))
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + _words(prev) > overlap:
                    break
                carry.insert(0, prev)
                carried += _words(prev)
            current, size = carry, carried
        current.append(block)
        size += n
    if current:
        text = "\n\n".join(current)
        if not windows or text != windows[-1]:
            windows.append(text)
    return windows


def _blocks(lines: list[str]) -> list[str]:
    """Blank-line separated blocks; a fenced code block never splits."""
    blocks: list[str] = []
    buf: list[str] = []
    fenced = False
    for line in lines:
        if _FENCE.match(line):
            fenced = not fenced
        if not line.strip() and not fenced:
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line.rstrip())
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b]


def markdown_sections(text: str) -> Iterator[tuple[str, list[str]]]:
    """(heading path, body lines) per section; headings inside code fences are ignored."""
    path: list[tuple[int, str]] = []
    body: list[str] = []
    fenced = False

    def label() -> str:
        return " > ".join(title for _, title in path)

    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        match = None if fenced else _HEADING.match(line)
        if match is None:
            body.append(line)
            continue
        yield label(), body
        body = []
        level = len(match.group(1))
        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, match.group(2).strip()))
    yield label(), body


def chunk_markdown(
    text: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS
) -> list[Piece]:
    pieces: list[Piece] = []
    for section, lines in markdown_sections(text):
        for window in window_blocks(_blocks(lines), max_words, overlap):
            pieces.append(Piece(section, window))
    return pieces


def chunk_plain(text: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Plain text (a PDF page): paragraph blocks, then windows."""
    return window_blocks(_blocks(text.splitlines()), max_words, overlap)


def first_heading(text: str) -> str | None:
    fenced = False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        match = None if fenced else _HEADING.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return None
