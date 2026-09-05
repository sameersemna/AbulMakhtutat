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
from pdf_processor import PDFProcessor
from toc_builder import write_toc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


@click.command()
@click.option("--input-dir", default="./input", type=click.Path(path_type=Path), help="Root input directory.")
@click.option("--output-dir", default="./output", type=click.Path(path_type=Path), help="Root output directory.")
@click.option("--workers", default=4, type=int, help="Concurrent OCR workers.")
@click.option("--dpi", default=300, type=int, help="Render DPI for page images.")
@click.option("--skip-formats", default="", help="Comma-separated formats to skip (md,html,epub,mobi,pdf).")
@click.option("--no-postprocess", is_flag=True, help="Disable text post-processing.")
@click.option("--max-pages", default=0, type=int, help="Limit pages per PDF (0 = all). Useful for testing.")
@click.option("--dry-run", is_flag=True, help="List discovered PDFs without processing.")
def main(input_dir: Path, output_dir: Path, workers: int, dpi: int, skip_formats: str, no_postprocess: bool, max_pages: int, dry_run: bool) -> None:
    config = get_config()
    config.pipeline.input_dir = input_dir
    config.pipeline.output_dir = output_dir
    config.pipeline.workers = workers
    config.pipeline.dpi = dpi
    config.pipeline.max_pages = max_pages
    if no_postprocess:
        config.ollama.postprocess_enabled = False
    config.ensure_dirs()

    processor = PDFProcessor(config)
    converter = FormatConverter(config)
    db_sync = DatabaseSync(config)
    skip = {s.strip() for s in skip_formats.split(",") if s.strip()}

    async def _run() -> None:
        pdfs = processor.discover_pdfs()
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

            outputs = converter.convert(result, markdown)
            for fmt in skip:
                outputs.pop(fmt, None)

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

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(processor.close())


if __name__ == "__main__":
    main()
