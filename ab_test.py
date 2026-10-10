"""A/B comparison: OCR with vs. without OpenCV preprocessing.

Renders a few pages from a PDF and OCRs each page twice — once with
preprocessing disabled (baseline) and once enabled — writing both outputs
side-by-side for manual comparison. Bypasses the per-page cache so the two
runs are independent.

Usage:
    .venv/bin/python ab_test.py --pdf input/rabee/majmua/1.pdf --pages 3,4,5
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import click
import pymupdf

from config import get_config
from hf_vlm_engine import build_ocr_engine
from thermal import ThermalGuard

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ab_test")


def _render(doc: pymupdf.Document, page_number: int, dpi: int) -> bytes:
    page = doc[page_number - 1]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


@click.command()
@click.option("--pdf", required=True, type=click.Path(path_type=Path), help="Input PDF.")
@click.option("--pages", default="3,4,5", help="Comma-separated page numbers to test.")
@click.option("--out", default="./ab_test_results", type=click.Path(path_type=Path), help="Output directory.")
@click.option("--mode", default="full", type=click.Choice(["light", "full", "sharpen"]), help="Preprocessing variant to A/B against baseline (off).")
def main(pdf: Path, pages: str, out: Path, mode: str) -> None:
    page_numbers = [int(x) for x in pages.split(",") if x.strip()]
    out.mkdir(parents=True, exist_ok=True)

    config = get_config()
    config.pipeline.workers = 1  # single worker to avoid GPU overload
    engine = build_ocr_engine(config)
    thermal = ThermalGuard(config)

    async def run() -> None:
        doc = pymupdf.open(str(pdf))
        results = []
        for pn in page_numbers:
            img = _render(doc, pn, config.pipeline.dpi)

            # Baseline: preprocessing OFF
            config.pipeline.preprocess.enabled = False
            config.pipeline.preprocess.mode = "off"
            await thermal.wait_if_hot()
            text_off, model_off = await engine.ocr_page(img)

            # Variant: preprocessing ON in the requested mode
            config.pipeline.preprocess.enabled = True
            config.pipeline.preprocess.mode = mode
            await thermal.wait_if_hot()
            text_on, model_on = await engine.ocr_page(img)

            entry = {
                "page": pn,
                "mode": mode,
                "baseline_model": model_off,
                "preprocess_model": model_on,
                "baseline_chars": len(text_off),
                "preprocess_chars": len(text_on),
            }
            results.append(entry)

            # Write side-by-side text files for manual diff.
            (out / f"page_{pn:04d}_baseline.md").write_text(text_off, encoding="utf-8")
            (out / f"page_{pn:04d}_{mode}.md").write_text(text_on, encoding="utf-8")

            logger.info(
                "page %d: baseline=%d chars (%s), %s=%d chars (%s)",
                pn, len(text_off), model_off, mode, len(text_on), model_on,
            )

        (out / "summary.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        doc.close()
        await engine.close()
        logger.info("A/B results written to %s", out)

    asyncio.run(run())


if __name__ == "__main__":
    main()
