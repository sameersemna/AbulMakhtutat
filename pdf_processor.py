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
from hf_vlm_engine import build_ocr_engine
from ocr_engine import FALLBACK_MODEL_PREFIX
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


def _render_page(doc: pymupdf.Document, page_number: int, dpi: int, config=None) -> bytes:
    """Render a single page to PNG bytes at the given DPI.

    If preprocessing is enabled in config, applies it to the rendered image so
    the vision OCR model sees a clean, binarized page (black text on white
    background) — this improves accuracy on pixelated scans and poor paper.
    """
    page = doc[page_number - 1]
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")

    if config is not None:
        from preprocess import preprocess
        img_bytes = preprocess(img_bytes, config)
    return img_bytes


def parse_page_range(spec: str, total_pages: int) -> list[int]:
    """Parse a page-range spec like '31-35' or '1,3,5-7' into a sorted list.

    Returns [] if spec is empty (meaning all pages). Clamps to [1, total_pages].
    """
    if not spec or not spec.strip():
        return []
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            pages.update(range(max(1, lo), min(total_pages, hi) + 1))
        else:
            try:
                pages.add(int(part))
            except ValueError:
                continue
    return sorted(p for p in pages if 1 <= p <= total_pages)


class PDFProcessor:
    """Orchestrates traversal, rendering, OCR, and Markdown assembly."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.engine = build_ocr_engine(self.config)
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

    def _load_cached_pages(self, result: DocumentResult, file_hash: str, total_pages: int) -> None:
        """Populate result.pages from the per-page cache (for already-completed files)."""
        if result.page_filter:
            target_pages = result.page_filter
        else:
            target_pages = range(1, total_pages + 1)
        for page_number in target_pages:
            cache_path = self._cache_path(file_hash, page_number)
            if cache_path.exists():
                text = cache_path.read_text(encoding="utf-8")
                result.pages.append(PageResult(page_number=page_number, text=text, model="cache"))
            else:
                result.pages.append(PageResult(page_number=page_number, text="", model="cache"))

    def _cached_page_result(self, file_hash: str, page_number: int) -> PageResult | None:
        """Return a usable cached PageResult, else None.

        A cached refusal/failure message is discarded (returns None) so the page is
        re-OCR'd. Shared by `_process_one` (to skip rendering on cache hits) and
        `_ocr_page_cached`.
        """
        if not self.config.pipeline.use_cache:
            return None
        cache_path = self._cache_path(file_hash, page_number)
        if not cache_path.exists():
            return None
        text = cache_path.read_text(encoding="utf-8")
        if self.engine.is_refusal(text):
            logger.warning("Cached page %d contains a refusal message; re-OCRing", page_number)
            return None
        return PageResult(page_number=page_number, text=text, model="cache")

    async def _ocr_page_cached(self, file_hash: str, page_number: int, image_bytes: bytes, max_dim: int | None = None) -> PageResult:
        """OCR a page, using a per-page cache to avoid re-OCR on resume.

        `max_dim` is forwarded to the vision engine (larger cap for heavy pages).
        """
        cached = self._cached_page_result(file_hash, page_number)
        if cached is not None:
            return cached

        start = time.monotonic()
        is_fallback = False
        try:
            # Wait for the GPU to cool before starting this page's inference.
            await self.thermal.wait_if_hot()
            raw, model = await self.engine.ocr_page(image_bytes, max_dim=max_dim)
            is_fallback = bool(model) and model.startswith(FALLBACK_MODEL_PREFIX)
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

        # Fallback pages (kraken/HF failed -> Ollama served them) are deliberately
        # NOT cached and NOT marked done, so a later run retries them with the
        # configured engine instead of silently persisting degraded text.
        if is_fallback:
            logger.warning(
                "Page %d was served by the Ollama fallback (%s); not caching or marking it done "
                "so a later run retries it with the configured engine",
                page_number, model,
            )
        elif self.config.pipeline.use_cache and not result.error:
            cache_path = self._cache_path(file_hash, page_number)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(result.text, encoding="utf-8")
        if not result.error and not is_fallback:
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

            # Optional page-range filter (e.g. "31-35") to focus on main text.
            page_filter = parse_page_range(self.config.pipeline.pages, result.total_pages)
            if page_filter:
                result.total_pages = len(page_filter)
                result.page_filter = page_filter

            # Idempotency: skip only if the full document was already completed.
            if self.state.is_completed(fhash):
                state = self.state.get(fhash) or {}
                done_pages = state.get("total_pages", 0)
                if done_pages >= real_page_count:
                    logger.info("Skipping already-completed file: %s", rel)
                    result.status = FileStatus.COMPLETED
                    # Load cached pages so downstream consumers (e.g. overlay
                    # generation) still have the OCR text available.
                    self._load_cached_pages(result, fhash, result.total_pages)
                    doc.close()
                    return result

            self.state.mark(fhash, FileStatus.PROCESSING, rel_path=rel)
            completed = self.state.completed_pages(fhash)

            # Pages that still need OCR (not yet cached/completed).
            # If a page filter is set, only process those physical pages.
            if page_filter:
                target_pages = page_filter
            else:
                target_pages = list(range(1, result.total_pages + 1))
            pending = [pn for pn in target_pages if pn not in completed]

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
                    # Skip rendering entirely when the page is already cached.
                    result_page = self._cached_page_result(fhash, page_number)
                    if result_page is None:
                        # Render lazily, just before OCR, to avoid holding all pages in memory.
                        dpi = self.config.pipeline.dpi
                        image_bytes = _render_page(doc, page_number, dpi, self.config)
                        max_dim: int | None = None
                        # Optional: re-render diacritized / small-text pages at a higher
                        # DPI and a larger vision-model cap so diacritics survive
                        # downscaling. Off by default (base behaviour unchanged).
                        if self.config.pipeline.heavy_heuristic:
                            from preprocess import score_diacritic_density

                            score = score_diacritic_density(image_bytes)
                            if score >= self.config.pipeline.heavy_threshold:
                                dpi = self.config.pipeline.dpi_heavy
                                image_bytes = _render_page(doc, page_number, dpi, self.config)
                                max_dim = self.config.ollama.max_image_dim_heavy
                                logger.info(
                                    "  page %d: heavy page (diacritic score %.2f) -> DPI %d, image cap %d",
                                    page_number, score, dpi, max_dim,
                                )
                        result_page = await self._ocr_page_cached(fhash, page_number, image_bytes, max_dim=max_dim)
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

            # Merge cached + freshly OCR'd pages in order (respecting page filter).
            page_map = {p.page_number: p for p in page_results}
            for page_number in target_pages:
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

    async def run(self, pdfs: list[Path] | None = None) -> list[DocumentResult]:
        """Process the given PDFs, or all discovered ones when none are passed."""
        if pdfs is None:
            pdfs = self.discover_pdfs()
            logger.info("Discovered %d PDF(s) in %s", len(pdfs), self.config.pipeline.input_dir)
        # Start the continuous temperature monitor so we can detect spikes that
        # occur *during* inference (not just before each page).
        self.thermal.start()
        results: list[DocumentResult] = []
        for pdf in pdfs:
            results.append(await self.process_pdf(pdf))
        return results
