"""Post-OCR text cleanup: strip page numbers and running headers.

Scanned books often have running headers (book title repeated on every page)
and page numbers that the vision model transcribes as standalone lines. These
pollute the output. This module provides lightweight, deterministic cleanup
that runs after OCR and before caching.
"""

from __future__ import annotations

import re
import unicodedata

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


# --- Arabic text normalization (search/RAG indexing + normalized OCR scoring) ---

# Combining marks stripped by default: Arabic tashkeel, Quranic annotations, and
# honorific signs. Kept separate from the base letters they attach to.
_ARABIC_DIACRITIC_RANGES = (
    (0x0610, 0x061A),  # Arabic sign / honorific combining marks
    (0x064B, 0x065F),  # tashkeel: fatha, damma, kasra, sukun, shadda, tanwin...
    (0x0670, 0x0670),  # superscript (dagger) alef
    (0x06D6, 0x06DC),  # Quranic annotation signs
    (0x06DF, 0x06E4),
    (0x06E7, 0x06E8),
    (0x06EA, 0x06ED),
)

# Optional collapse of common orthographic variants (search-friendly).
_ARABIC_LETTER_MAP = {
    "\u0622": "\u0627",  # آ -> ا
    "\u0623": "\u0627",  # أ -> ا
    "\u0625": "\u0627",  # إ -> ا
    "\u0671": "\u0627",  # ٱ -> ا
    "\u0649": "\u064A",  # ى -> ي
    "\u0626": "\u064A",  # ئ -> ي
    "\u0624": "\u0648",  # ؤ -> و
    "\u0629": "\u0647",  # ة -> ه
    "\u06A9": "\u0643",  # Persian ک -> ك
    "\u06CC": "\u064A",  # Persian ی -> ي
}

_TATWEEL = "\u0640"  # kashida
_ARABIC_INDIC_DIGITS = {chr(0x0660 + i): str(i) for i in range(10)}
_EXTENDED_ARABIC_INDIC_DIGITS = {chr(0x06F0 + i): str(i) for i in range(10)}


def _is_diacritic(ch: str) -> bool:
    """True if `ch` is an Arabic combining mark in the strip set."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_DIACRITIC_RANGES)


def normalize_arabic(
    text: str,
    *,
    strip_diacritics: bool = True,
    normalize_letters: bool = True,
    normalize_digits: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Normalize Arabic text for search/RAG indexing and normalized OCR scoring.

    Steps (all optional except the NFKC pass):
      1. NFKC — decomposes Arabic presentation forms and ligatures in the
         FB50-FDFF / FE70-FEFF ranges (e.g. the lam-alef ligature لا) into base
         letters, so stored text uses base Unicode rather than visual glyph forms.
      2. Digits — Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) to ASCII 0-9.
      3. Letters — collapse common orthographic variants (see `_ARABIC_LETTER_MAP`).
      4. Diacritics — strip tashkeel and Quranic combining marks.
      5. Tatweel (kashida) — removed.
      6. Whitespace — collapsed to single spaces and trimmed.

    Pure and side-effect-free, so both the pipeline and the evaluation harness
    (`eval_ocr.py`) can share it.
    """
    if not text:
        return ""

    # 1. NFKC: presentation forms / ligatures -> base letters.
    out = unicodedata.normalize("NFKC", text)

    # 2-5. Character-level transforms.
    chars: list[str] = []
    for ch in out:
        if normalize_digits and (ch in _ARABIC_INDIC_DIGITS or ch in _EXTENDED_ARABIC_INDIC_DIGITS):
            chars.append(_ARABIC_INDIC_DIGITS.get(ch) or _EXTENDED_ARABIC_INDIC_DIGITS[ch])
            continue
        if strip_diacritics and _is_diacritic(ch):
            continue
        if ch == _TATWEEL:
            continue
        if normalize_letters and ch in _ARABIC_LETTER_MAP:
            chars.append(_ARABIC_LETTER_MAP[ch])
            continue
        chars.append(ch)
    out = "".join(chars)

    # 6. Whitespace collapse.
    if collapse_whitespace:
        out = re.sub(r"\s+", " ", out).strip()
    return out
