"""CLI entry point for the Arabic OCR pipeline.

Usage:
    python main.py --input-dir ./input --output-dir ./output --workers 4
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

from config import get_config
from database_sync import DatabaseSync
from format_converter import FormatConverter
from log_utils import setup_logging
from pdf_processor import PDFProcessor
from toc_builder import write_toc

setup_logging(level=logging.INFO)
logger = logging.getLogger("main")


@click.command()
@click.option("--input-dir", default="./input", type=click.Path(path_type=Path), help="Root input directory.")
@click.option("--output-dir", default="./output", type=click.Path(path_type=Path), help="Root output directory.")
@click.option("--workers", default=1, type=int, help="Concurrent OCR workers (keep low to avoid GPU overheating).")
@click.option("--dpi", default=300, type=int, help="Render DPI for page images.")
@click.option("--skip-formats", default="", help="Comma-separated formats to skip (md,html,epub,mobi,pdf).")
@click.option("--no-postprocess", is_flag=True, help="Disable text post-processing.")
@click.option("--correct", is_flag=True, help="Enable post-OCR correction pass (fixes OCR artifacts with a text model).")
@click.option("--preprocess", "preprocess_mode", flag_value="full", default="off", help="Enable full OpenCV preprocessing (deskew, binarize, denoise, contrast).")
@click.option("--preprocess-light", "preprocess_mode", flag_value="light", help="Enable light preprocessing (global contrast normalization only).")
@click.option("--preprocess-sharpen", "preprocess_mode", flag_value="sharpen", help="Enable mild unsharp-mask preprocessing (VLM-friendly).")
@click.option("--overlay", "overlay_mode", flag_value="page", default="off", help="Generate a searchable overlay PDF (original scan + invisible OCR text layer).")
@click.option("--overlay-bbox", "overlay_mode", flag_value="bbox", help="Generate overlay PDF with model-precise bounding-box alignment (most accurate).")
@click.option("--overlay-line", "overlay_mode", flag_value="line", help="Generate overlay PDF with line-level alignment (robust for Arabic).")
@click.option("--max-pages", default=0, type=int, help="Limit pages per PDF (0 = all). Useful for testing.")
@click.option("--pages", default="", help="Page range to process, e.g. '31-35' or '1,3,5-7'. Empty = all pages. Useful for focusing on main text.")
@click.option("--only", default="", help="Process only this PDF filename (e.g. 1.pdf). Empty = all PDFs.")
@click.option("--dry-run", is_flag=True, help="List discovered PDFs without processing.")
def main(input_dir: Path, output_dir: Path, workers: int, dpi: int, skip_formats: str, no_postprocess: bool, correct: bool, preprocess_mode: str, overlay_mode: str, max_pages: int, pages: str, only: str, dry_run: bool) -> None:
    config = get_config()
    config.pipeline.input_dir = input_dir
    config.pipeline.output_dir = output_dir
    config.pipeline.workers = workers
    config.pipeline.dpi = dpi
    config.pipeline.max_pages = max_pages
    config.pipeline.pages = pages
    if no_postprocess:
        config.ollama.postprocess_enabled = False
    if overlay_mode in ("page", "word", "bbox", "line"):
        config.pipeline.overlay_mode = overlay_mode
    if correct:
        config.ollama.postprocess_enabled = True
    if preprocess_mode in ("light", "full", "sharpen"):
        config.pipeline.preprocess.mode = preprocess_mode
        config.pipeline.preprocess.enabled = True
    config.ensure_dirs()

    processor = PDFProcessor(config)
    converter = FormatConverter(config)
    db_sync = DatabaseSync(config)
    skip = {s.strip() for s in skip_formats.split(",") if s.strip()}

    async def _run() -> None:
        try:
            pdfs = processor.discover_pdfs()
            if only:
                pdfs = [p for p in pdfs if p.name == only]
                if not pdfs:
                    click.echo(f"No PDF found matching --only {only}", err=True)
                    return
            if dry_run:
                for p in pdfs:
                    click.echo(str(p))
                return

            results = await processor.run()
            summary = []
            for result in results:
                if result.status.value == "failed":
                    logger.error("FAILED: %s (%s)", result.rel_path, result.error)
                    summary.append({"rel_path": result.rel_path, "status": "failed", "error": result.error})
                    continue

                markdown = result.markdown
                toc_json_path, toc_md_path = write_toc(result, markdown)

                logger.info(
                    "Converting %s to output formats (md/html/epub/mobi/pdf) — this may take a minute for large documents...",
                    result.rel_path,
                )
                outputs = converter.convert(result, markdown)
                for fmt in skip:
                    outputs.pop(fmt, None)

                # BBox/line overlay modes use the model's grounding (async, needs engine).
                if config.pipeline.overlay_mode == "bbox":
                    outputs["overlay"] = await converter.write_overlay_bbox(result, processor.engine)
                elif config.pipeline.overlay_mode == "line":
                    outputs["overlay"] = await converter.write_overlay_line(result, processor.engine, thermal=processor.thermal)

                # Load toc entries for DB sync.
                toc_entries = json.loads(toc_json_path.read_text(encoding="utf-8")).get("entries", [])

                await db_sync.sync(result, toc_entries, processor.engine)

                produced = {k: (str(v) if v else None) for k, v in outputs.items()}
                summary.append(
                    {
                        "rel_path": result.rel_path,
                        "status": "completed",
                        "pages": result.total_pages,
                        "outputs": produced,
                    }
                )
                logger.info("COMPLETED: %s (%d pages) -> %s", result.rel_path, result.total_pages, result.output_dir)

            # Write a run summary manifest.
            manifest = config.pipeline.output_dir / "run_summary.json"
            manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            click.echo(f"\nDone. Processed {len(summary)} document(s). Summary: {manifest}")
        finally:
            await processor.close()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
