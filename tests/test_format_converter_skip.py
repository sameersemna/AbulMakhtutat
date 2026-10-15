"""Unit tests for FormatConverter skipping semantics.

Covers the 2026-10-15 fix: `--skip-formats` used to only prune the reported
summary while every format was still generated. `convert(..., skip=...)` must
never invoke the writer of a skipped format, and the sync overlay writer must
not duplicate the async line/bbox overlay work.

Run:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

# Ensure the repo root is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_WRITERS = {
    "_write_markdown": "md",
    "_write_html": "html",
    "_write_epub": "epub",
    "_write_pdf": "pdf",
    "_write_mobi": "mobi",
    "_write_overlay_pdf": "overlay",
}


class TestConvertSkip(unittest.TestCase):
    def _converter_with_stubs(self):
        from format_converter import FormatConverter

        converter = FormatConverter.__new__(FormatConverter)  # bypass __init__ (real config not needed)
        calls: list[str] = []

        def make_stub(fmt: str):
            def stub(*args, **kwargs):
                calls.append(fmt)
                return Path(f"/tmp/{fmt}")
            return stub

        for name, fmt in _WRITERS.items():
            setattr(converter, name, make_stub(fmt))
        return converter, calls

    def test_skip_prevents_generation(self) -> None:
        converter, calls = self._converter_with_stubs()
        out = converter.convert(result=None, markdown="# نص", skip={"epub", "mobi", "overlay"})
        self.assertEqual(set(out), {"md", "html", "pdf"})
        self.assertEqual(set(calls), {"md", "html", "pdf"})

    def test_no_skip_generates_everything(self) -> None:
        converter, calls = self._converter_with_stubs()
        out = converter.convert(result=None, markdown="# نص")
        self.assertEqual(set(out), {"md", "html", "epub", "pdf", "mobi", "overlay"})
        self.assertEqual(set(calls), {"md", "html", "epub", "pdf", "mobi", "overlay"})

    def test_skip_everything(self) -> None:
        converter, calls = self._converter_with_stubs()
        out = converter.convert(result=None, markdown="# نص", skip=set(_WRITERS.values()))
        self.assertEqual(out, {})
        self.assertEqual(calls, [])


class TestOverlaySingleBuilder(unittest.TestCase):
    def _stub_result(self):
        from models import DocumentResult

        return DocumentResult(source_path=Path("input/book.pdf"), rel_path="book.pdf",
                              output_dir=Path("/tmp/kilo/overlay-test"), file_hash="x", total_pages=0)

    def test_sync_overlay_writer_skips_async_modes(self) -> None:
        """page/word are built by the sync writer; off/bbox/line must return None."""
        from config import get_config
        from format_converter import FormatConverter

        cfg = get_config()
        converter = FormatConverter(cfg)
        for mode in ("off", "bbox", "line"):
            cfg.pipeline.overlay_mode = mode
            with self.subTest(mode=mode):
                self.assertIsNone(converter._write_overlay_pdf(self._stub_result()))


if __name__ == "__main__":
    unittest.main()
