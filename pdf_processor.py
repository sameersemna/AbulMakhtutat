"""Asynchronous PDF traversal, page rendering, and Markdown assembly.

Recursively scans `input/` for PDFs, mirrors the nested hierarchy into `output/`,
renders pages at high DPI, OCRs them via the Ollama engine, and assembles the
canonical Markdown with explicit page anchors.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path

import pymupdf

from config import Config, get_config
from models import DocumentResult, FileStatus, PageResult
from ocr_engine import OCREngine
from state_tracker import StateTracker
from text_cleanup import cleanup_text
from thermal import ThermalGuard

logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_page(doc: pymupdf.Document, page_number: int, dpi: int) -> bytes:
    """Render a single page to PNG bytes at the given DPI."""
    page = doc[page_number - 1]
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


class PDFProcessor:
    """Orchestrates traversal, rendering, OCR, and Markdown assembly."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.engine = OCREngine(self.config)
        self.state = StateTracker(self.config)
        self.thermal = ThermalGuard(self.config)
        self._semaphore: asyncio.Semaphore | None = None

    async def close(self) -> None:
        await self.thermal.stop()
        await self.engine.close()

    def discover_pdfs(self) -> list[Path]:
        """Recursively find all PDFs under input_dir."""
        return sorted(self.config.pipeline.input_dir.rglob("*.pdf"))

    def _output_dir_for(self, pdf_path: Path) -> Path:
        """Mirror input hierarchy into output, using the stem as a folder."""
        rel = pdf_path.relative_to(self.config.pipeline.input_dir)
        # e.g. Author/Book/vol1.pdf -> Author/Book/vol1/
        out_rel = rel.with_suffix("")
        return self.config.pipeline.output_dir / out_rel

    def _cache_path(self, file_hash: str, page_number: int) -> Path:
        return self.config.pipeline.cache_dir / file_hash / f"page_{page_number:04d}.txt"

    async def _ocr_page_cached(self, file_hash: str, page_number: int, image_bytes: bytes) -> PageResult:
        """OCR a page, using a per-page cache to avoid re-OCR on resume."""
        cache_path = self._cache_path(file_hash, page_number)
        if self.config.pipeline.use_cache and cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
            # If the cached text is a refusal/failure message (from an older run),
            # discard it and re-OCR so we don't keep the failure prose.
            if not self.engine.is_refusal(text):
                return PageResult(page_number=page_number, text=text, model="cache")
            logger.warning("Cached page %d contains a refusal message; re-OCRing", page_number)

        start = time.monotonic()
        try:
            # Wait for the GPU to cool before starting this page's inference.
            await self.thermal.wait_if_hot()
            raw, model = await self.engine.ocr_page(image_bytes)
            text = await self.engine.postprocess(raw)
            # Clean up page numbers and running headers.
            text = cleanup_text(text, self.config.pipeline.running_header)
            duration_ms = int((time.monotonic() - start) * 1000)
            quality = self.engine.quality_score(text)
            if not text.strip():
                # OCR returned empty (all attempts refused) — treat as failure
                # so it is NOT cached and can be retried on a future run.
                result = PageResult(
                    page_number=page_number, text="", model=model,
                    duration_ms=duration_ms, error="OCR returned no text (all attempts refused)",
                    quality=0.0,
                )
            else:
                result = PageResult(page_number=page_number, text=text, model=model, duration_ms=duration_ms, quality=quality)
                if quality < 0.5:
                    logger.warning("Low-quality OCR for page %d (score %.2f)", page_number, quality)
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.error("OCR failed for page %d: %s", page_number, exc)
            result = PageResult(page_number=page_number, text="", model="", duration_ms=duration_ms, error=str(exc), quality=0.0)

        # Optional breathing room between pages to avoid sustained GPU load.
        await self.thermal.cooldown_between_pages()

        if self.config.pipeline.use_cache and not result.error:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(result.text, encoding="utf-8")
        if not result.error:
            self.state.record_page(file_hash, page_number)
        return result

    async def process_pdf(self, pdf_path: Path) -> DocumentResult:
        """Process a single PDF end-to-end."""
        rel = str(pdf_path.relative_to(self.config.pipeline.input_dir))
        fhash = _file_hash(pdf_path)
        out_dir = self._output_dir_for(pdf_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        result = DocumentResult(
            source_path=pdf_path,
            rel_path=rel,
            output_dir=out_dir,
            file_hash=fhash,
            total_pages=0,
        )

        try:
            doc = pymupdf.open(str(pdf_path))
            real_page_count = doc.page_count
            result.total_pages = real_page_count
            if self.config.pipeline.max_pages > 0:
                result.total_pages = min(real_page_count, self.config.pipeline.max_pages)

            # Idempotency: skip only if the full document was already completed.
            if self.state.is_completed(fhash):
                state = self.state.get(fhash) or {}
                done_pages = state.get("total_pages", 0)
                if done_pages >= real_page_count:
                    logger.info("Skipping already-completed file: %s", rel)
                    result.status = FileStatus.COMPLETED
                    doc.close()
                    return result

            self.state.mark(fhash, FileStatus.PROCESSING, rel_path=rel)
            completed = self.state.completed_pages(fhash)

            # Pages that still need OCR (not yet cached/completed).
            pending = [pn for pn in range(1, result.total_pages + 1) if pn not in completed]

            if pending:
                # Estimate duration: ~10s/page for the vision model, divided by workers.
                est_seconds = len(pending) * 10.0 / max(self.config.pipeline.workers, 1)
                logger.info(
                    "Processing %s: %d pages total, %d pending OCR (est. %s)",
                    rel, result.total_pages, len(pending), _fmt_duration(est_seconds),
                )
                logger.info(
                    "Starting OCR for %s — this will take a while; progress is logged per page.",
                    rel,
                )
            else:
                logger.info("Processing %s: %d pages total, all already cached", rel, result.total_pages)

            sem = asyncio.Semaphore(self.config.pipeline.workers)
            self._semaphore = sem
            done_count = 0
            start_time = time.monotonic()

            async def _process_one(page_number: int) -> PageResult:
                nonlocal done_count
                async with sem:
                    # Render lazily, just before OCR, to avoid holding all pages in memory.
                    image_bytes = _render_page(doc, page_number, self.config.pipeline.dpi)
                    result_page = await self._ocr_page_cached(fhash, page_number, image_bytes)
                    done_count += 1
                    elapsed = time.monotonic() - start_time
                    rate = done_count / elapsed if elapsed > 0 else 0.0
                    remaining = (len(pending) - done_count) / rate if rate > 0 else 0.0
                    logger.info(
                        "  [%d/%d] page %d OCR'd (%s) — elapsed %s, ETA %s",
                        done_count, len(pending), page_number, result_page.model,
                        _fmt_duration(elapsed), _fmt_duration(remaining),
                    )
                    return result_page

            # Process pending pages with bounded concurrency, preserving order.
            page_results = await asyncio.gather(*(_process_one(pn) for pn in pending))

            # Merge cached + freshly OCR'd pages in order.
            page_map = {p.page_number: p for p in page_results}
            for page_number in range(1, result.total_pages + 1):
                if page_number in page_map:
                    result.pages.append(page_map[page_number])
                else:
                    # Already completed earlier — reload from cache.
                    cache_path = self._cache_path(fhash, page_number)
                    text = cache_path.read_text(encoding="utf-8") if cache_path.exists() else ""
                    result.pages.append(PageResult(page_number=page_number, text=text, model="cache"))

            result.pages.sort(key=lambda p: p.page_number)
            result.status = FileStatus.COMPLETED
            self.state.mark(fhash, FileStatus.COMPLETED, total_pages=result.total_pages)
            doc.close()
        except Exception as exc:
            logger.exception("Failed to process %s", rel)
            result.status = FileStatus.FAILED
            result.error = str(exc)
            self.state.mark(fhash, FileStatus.FAILED, error=str(exc))

        return result

    async def run(self) -> list[DocumentResult]:
        """Process all discovered PDFs."""
        pdfs = self.discover_pdfs()
        logger.info("Discovered %d PDF(s) in %s", len(pdfs), self.config.pipeline.input_dir)
        # Start the continuous temperature monitor so we can detect spikes that
        # occur *during* inference (not just before each page).
        self.thermal.start()
        results: list[DocumentResult] = []
        for pdf in pdfs:
            results.append(await self.process_pdf(pdf))
        return results
