"""Post-OCR text cleanup: strip page numbers and running headers.

Scanned books often have running headers (book title repeated on every page)
and page numbers that the vision model transcribes as standalone lines. These
pollute the output. This module provides lightweight, deterministic cleanup
that runs after OCR and before caching.
"""

from __future__ import annotations

import re

# Matches a standalone page number: Arabic-Indic digits (٠-٩), Eastern Arabic
# digits (۰-۹), or Latin digits, optionally wrapped in markdown heading markers.
_PAGE_NUMBER_RE = re.compile(
    r"^\s{0,3}#{0,6}\s*[\d\u0660-\u0669\u06F0-\u06F9]{1,4}\s*#*\s*$"
)

# Matches a line that is ONLY a page number (no other text).
_STANDALONE_NUMBER_RE = re.compile(
    r"^[\d\u0660-\u0669\u06F0-\u06F9]{1,4}$"
)


def strip_page_numbers(text: str) -> str:
    """Remove standalone page-number lines from OCR text.

    A page number is a line containing only digits (Latin, Arabic-Indic, or
    Eastern Arabic), optionally wrapped in markdown heading markers (e.g.
    `## ٦٨` or `## 70`). Real content lines are never pure numbers.
    """
    lines = text.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines (handled by caller) and pure page-number lines.
        if _PAGE_NUMBER_RE.match(stripped) or _STANDALONE_NUMBER_RE.match(stripped):
            continue
        kept.append(line)
    return "\n".join(kept)


def strip_running_headers(text: str, header: str | None) -> str:
    """Remove a known running header line from the top of the text.

    `header` is the book title (e.g. "موسوعة مؤلفات ورسائل وفتاوى الشيخ ربيع
    المدخلي") that repeats as a running header on every page. If provided, any
    leading heading line matching it is removed.
    """
    if not header:
        return text
    lines = text.splitlines()
    # Remove leading heading lines that match the running header.
    while lines and lines[0].strip().lstrip("#").strip() == header.strip():
        lines.pop(0)
    return "\n".join(lines)


def cleanup_text(text: str, running_header: str | None = None) -> str:
    """Apply all post-OCR text cleanup steps."""
    text = strip_page_numbers(text)
    if running_header:
        text = strip_running_headers(text, running_header)
    return text
