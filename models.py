"""Data models shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class FileStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PageResult:
    """OCR result for a single PDF page."""

    page_number: int  # 1-based physical page number
    text: str
    model: str
    duration_ms: int = 0
    error: str | None = None
    quality: float = 1.0  # heuristic confidence score (0.0 = bad, 1.0 = good)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentResult:
    """Aggregated result for one PDF document."""

    source_path: Path
    rel_path: str
    output_dir: Path
    file_hash: str
    total_pages: int
    pages: list[PageResult] = field(default_factory=list)
    status: FileStatus = FileStatus.PENDING
    error: str | None = None
    # Optional list of physical page numbers to process (e.g. [31,32,33,34,35]).
    # When set, only these pages are OCR'd and overlaid. Empty = all pages.
    page_filter: list[int] = field(default_factory=list)

    @property
    def markdown(self) -> str:
        """Assemble canonical Markdown with page anchors."""
        parts: list[str] = []
        for page in self.pages:
            if page.error:
                parts.append(
                    f'<div id="page-{page.page_number}" class="page-anchor" data-page="{page.page_number}"></div>\n'
                    f"<!-- PAGE_{page.page_number}_START -->\n\n"
                    f"> ⚠️ OCR failed for page {page.page_number}: {page.error}\n\n"
                )
                continue
            parts.append(
                f'<div id="page-{page.page_number}" class="page-anchor" data-page="{page.page_number}"></div>\n'
                f"<!-- PAGE_{page.page_number}_START -->\n\n"
                f"{page.text.strip()}\n\n"
            )
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "rel_path": self.rel_path,
            "output_dir": str(self.output_dir),
            "file_hash": self.file_hash,
            "total_pages": self.total_pages,
            "status": self.status.value,
            "error": self.error,
            "pages": [p.to_dict() for p in self.pages],
        }


@dataclass
class TocEntry:
    """A single Table-of-Contents entry."""

    level: int
    title: str
    page: int
    anchor: str
    children: list["TocEntry"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "title": self.title,
            "page": self.page,
            "anchor": self.anchor,
            "children": [c.to_dict() for c in self.children],
        }
