"""Table of Contents extraction and generation.

Parses the canonical Markdown for headings, maps each heading to its source page
anchor, and emits both `toc.json` (structured) and `TOC.md` (human-readable).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from models import DocumentResult, TocEntry

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_PAGE_ANCHOR_RE = re.compile(r'<div id="page-(\d+)"')


def _page_for_offset(markdown: str, offset: int) -> int:
    """Find the page number of the most recent page anchor before `offset`."""
    page = 1
    for m in _PAGE_ANCHOR_RE.finditer(markdown):
        if m.start() > offset:
            break
        page = int(m.group(1))
    return page


def extract_toc(markdown: str) -> list[TocEntry]:
    """Extract a flat list of heading entries with page anchors."""
    entries: list[TocEntry] = []
    for m in _HEADING_RE.finditer(markdown):
        level = len(m.group(1))
        title = m.group(2).strip()
        page = _page_for_offset(markdown, m.start())
        entries.append(
            TocEntry(
                level=level,
                title=title,
                page=page,
                anchor=f"#page-{page}",
            )
        )
    return entries


def _build_tree(entries: list[TocEntry]) -> list[TocEntry]:
    """Convert a flat heading list into a hierarchical tree."""
    root: list[TocEntry] = []
    stack: list[TocEntry] = []
    for entry in entries:
        while stack and stack[-1].level >= entry.level:
            stack.pop()
        if stack:
            stack[-1].children.append(entry)
        else:
            root.append(entry)
        stack.append(entry)
    return root


def _render_toc_md(entries: list[TocEntry], depth: int = 0) -> str:
    lines: list[str] = []
    for entry in entries:
        indent = "  " * depth
        lines.append(f"{indent}- [{entry.title}]({entry.anchor})")
        if entry.children:
            lines.append(_render_toc_md(entry.children, depth + 1))
    return "\n".join(lines)


def write_toc(result: DocumentResult, markdown: str) -> tuple[Path, Path]:
    """Write toc.json and TOC.md into the document's output directory."""
    entries = extract_toc(markdown)
    tree = _build_tree(entries)

    toc_json_path = result.output_dir / "toc.json"
    toc_md_path = result.output_dir / "TOC.md"

    toc_json_path.write_text(
        json.dumps(
            {
                "source": result.rel_path,
                "total_pages": result.total_pages,
                "entries": [e.to_dict() for e in tree],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    toc_md_path.write_text(
        "# فهرس المحتويات\n\n" + (_render_toc_md(tree) if tree else "_No headings detected._"),
        encoding="utf-8",
    )

    return toc_json_path, toc_md_path
