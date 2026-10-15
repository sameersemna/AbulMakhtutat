"""Regression tests for the per-page OCR cache in `pdf_processor`.

Covers the 2026-10-15 bug where `_ocr_page_cached` referenced an undefined
`cache_path` after a successful OCR, failing every page with a NameError and
leaving nothing cached.

Run:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Ensure the repo root is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _StubEngine:
    """Minimal vision engine: echoes a fixed transcript, never refuses."""

    def __init__(self, text: str = "نص تجريبي") -> None:
        self.text = text
        self.calls = 0

    async def ocr_page(self, image_bytes: bytes, max_dim: int | None = None) -> tuple[str, str]:
        self.calls += 1
        return self.text, "stub"

    async def postprocess(self, text: str) -> str:
        return text

    def quality_score(self, text: str) -> float:
        return 1.0

    def is_refusal(self, text: str) -> bool:
        return False

    async def close(self) -> None:  # pragma: no cover - nothing to close
        pass


class _StubThermal:
    async def wait_if_hot(self) -> None:
        pass

    async def cooldown_between_pages(self) -> None:
        pass

    async def stop(self) -> None:  # pragma: no cover
        pass


class TestPageCacheWrite(unittest.TestCase):
    def _make_processor(self, tmp: str, engine: _StubEngine):
        from config import get_config
        from pdf_processor import PDFProcessor

        cfg = get_config()
        cfg.pipeline.cache_dir = Path(tmp) / "cache"
        cfg.pipeline.state_file = Path(tmp) / "state.json"
        cfg.pipeline.use_cache = True
        with mock.patch("pdf_processor.build_ocr_engine", return_value=engine):
            processor = PDFProcessor(cfg)
        processor.thermal = _StubThermal()
        return processor

    def test_successful_page_is_cached_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = _StubEngine()
            processor = self._make_processor(tmp, engine)
            result = asyncio.run(processor._ocr_page_cached("abc123", 7, b"png"))
            self.assertFalse(result.error)
            self.assertEqual(result.text, "نص تجريبي")

            cache_file = Path(tmp) / "cache" / "abc123" / "page_0007.txt"
            self.assertTrue(cache_file.is_file(), "page text was not written to the cache")
            self.assertEqual(cache_file.read_text(encoding="utf-8"), "نص تجريبي")

            # Second call must be served from the cache (no engine call).
            again = asyncio.run(processor._ocr_page_cached("abc123", 7, b"png"))
            self.assertEqual(again.model, "cache")
            self.assertEqual(engine.calls, 1)

    def test_failed_page_is_not_cached(self) -> None:
        class _FailingEngine(_StubEngine):
            async def ocr_page(self, image_bytes: bytes, max_dim: int | None = None):
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            processor = self._make_processor(tmp, _FailingEngine())
            result = asyncio.run(processor._ocr_page_cached("deadbeef", 1, b"png"))
            self.assertTrue(result.error)
            self.assertFalse((Path(tmp) / "cache" / "deadbeef").exists(),
                             "a failed page must not be cached")

    def test_fallback_page_is_not_cached_and_retried(self) -> None:
        """A page served by the Ollama fallback (tagged model) must not be cached
        or marked done, so a later run retries it with the configured engine."""
        from ocr_engine import FALLBACK_MODEL_PREFIX

        class _FallbackEngine(_StubEngine):
            async def ocr_page(self, image_bytes: bytes, max_dim: int | None = None):
                self.calls += 1
                return "نص رديء من الفولباك", f"{FALLBACK_MODEL_PREFIX}qwen2.5vl:7b"

        with tempfile.TemporaryDirectory() as tmp:
            engine = _FallbackEngine()
            processor = self._make_processor(tmp, engine)
            result = asyncio.run(processor._ocr_page_cached("cafe", 5, b"png"))
            self.assertFalse(result.error)
            self.assertTrue(result.model.startswith(FALLBACK_MODEL_PREFIX))
            self.assertFalse((Path(tmp) / "cache" / "cafe").exists(),
                             "fallback text must not be cached")

            # Not recorded as done -> a later call retries (engine is called again).
            again = asyncio.run(processor._ocr_page_cached("cafe", 5, b"png"))
            self.assertNotEqual(again.model, "cache")
            self.assertEqual(engine.calls, 2)


if __name__ == "__main__":
    unittest.main()
