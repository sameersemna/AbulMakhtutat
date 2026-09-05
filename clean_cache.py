"""Purge bad OCR cache entries (refusal messages / empty text).

Scans `.ocr_cache/` for pages whose cached text is a refusal/failure message
(e.g. "overexposed", "unreadable") or has no Arabic content, deletes those
cache files, and removes the corresponding page numbers from the state file's
`completed_pages` list so they are re-OCR'd on the next run.

Usage:
    .venv/bin/python clean_cache.py            # dry-run: list what would be purged
    .venv/bin/python clean_cache.py --apply    # actually delete + update state
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys

import click

from config import get_config
from ocr_engine import OCREngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clean_cache")


def _find_bad_cache_files(cache_dir: pathlib.Path, engine: OCREngine) -> list[pathlib.Path]:
    """Return cache files whose text is a refusal/failure message."""
    bad: list[pathlib.Path] = []
    for f in cache_dir.rglob("*.txt"):
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        if engine.is_refusal(text):
            bad.append(f)
    return bad


def _page_number_from_filename(path: pathlib.Path) -> int | None:
    """Extract the page number from a cache filename like page_0014.txt."""
    import re

    m = re.search(r"page_(\d+)\.txt$", path.name)
    return int(m.group(1)) if m else None


@click.command()
@click.option("--apply", is_flag=True, help="Actually delete bad cache files and update state (default is dry-run).")
def main(apply: bool) -> None:
    config = get_config()
    engine = OCREngine(config)
    cache_dir = config.pipeline.cache_dir
    state_file = config.pipeline.state_file

    bad_files = _find_bad_cache_files(cache_dir, engine)
    logger.info("Found %d bad cache file(s)", len(bad_files))

    if not bad_files:
        click.echo("Nothing to purge — cache is clean.")
        return

    # Group bad page numbers by file-hash (cache subdirectory name).
    bad_by_hash: dict[str, list[int]] = {}
    for f in bad_files:
        fhash = f.parent.name
        pn = _page_number_from_filename(f)
        if pn is not None:
            bad_by_hash.setdefault(fhash, []).append(pn)

    for f in sorted(bad_files):
        click.echo(f"  {'DELETE' if apply else 'would delete'}: {f}")

    if not apply:
        click.echo("\nDry-run complete. Re-run with --apply to purge.")
        return

    # Delete bad cache files.
    for f in bad_files:
        f.unlink(missing_ok=True)

    # Remove bad page numbers from the state file's completed_pages lists.
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}

        changed = False
        for fhash, pages in bad_by_hash.items():
            if fhash in state:
                completed = set(state[fhash].get("completed_pages", []))
                new_completed = completed - set(pages)
                if new_completed != completed:
                    state[fhash]["completed_pages"] = sorted(new_completed)
                    # If the file was marked completed, revert to processing so it re-runs.
                    if state[fhash].get("status") == "completed":
                        state[fhash]["status"] = "processing"
                    changed = True

        if changed:
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Updated state file: removed %d bad page(s) from completed_pages", sum(len(p) for p in bad_by_hash.values()))

    click.echo(f"\nPurged {len(bad_files)} bad cache file(s). Re-run the pipeline to re-OCR them.")


if __name__ == "__main__":
    main()
