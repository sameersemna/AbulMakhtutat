"""Rank un-annotated line crops by model uncertainty to prioritize annotation.

For every line image that has no non-empty transcript yet, runs the in-repo
Kraken model (see `kraken_engine.py`) and records the mean per-character
confidence. The worklist is written sorted by ascending confidence (most
uncertain lines first), which is the most informative order in which to spend
human annotation effort.

With `--prefill`, the model's transcription is also written to
`<data>/text/<stem>.txt` and `<data>/images/<stem>.gt.txt` (the same places
`prefill_line_gt.py` writes) so the annotator only has to correct mistakes.

*** These are MACHINE labels -- correct them by hand before training. ***

Typical workflow:
    .venv/bin/python make_line_crops.py --pdf input/rabee/majmua/9.pdf --pages 100-140
    .venv/bin/python prioritize_lines.py --data annotation/ --prefill --top 400
    # correct annotation/images/*.gt.txt worst-first (worklist order), then retrain

Usage:
    .venv/bin/python prioritize_lines.py --data annotation/ --out annotation/worklist.tsv
    .venv/bin/python prioritize_lines.py --data annotation/ --prefill --top 400
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("prioritize_lines")

_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _annotated(data_dir: Path, img: Path) -> bool:
    txt = data_dir / "text" / f"{img.stem}.txt"
    gt = img.with_suffix(".gt.txt")
    for candidate in (txt, gt):
        if candidate.exists() and candidate.read_text(encoding="utf-8").strip():
            return True
    return False


async def run(data_dir: Path, out: Path, top: int, prefill: bool, include_annotated: bool,
              min_chars: int, max_chars: int) -> int:
    from config import get_config
    from hf_vlm_engine import build_ocr_engine
    from kraken_engine import KrakenEngine

    img_dir = data_dir / "images"
    if not img_dir.is_dir():
        logger.error("image directory not found: %s", img_dir)
        return 1

    cfg = get_config()
    cfg.pipeline.workers = 1
    cfg.vision.backend = "kraken"
    engine = build_ocr_engine(cfg)
    if not isinstance(engine, KrakenEngine):
        logger.error("kraken backend unavailable (model %s); is VISION_KRAKEN_MODEL set correctly?",
                     cfg.vision.kraken_model)
        await engine.close()
        return 2
    logger.info("model: %s", cfg.vision.kraken_model)

    candidates = [p for p in sorted(img_dir.iterdir())
                  if p.suffix.lower() in _IMAGE_EXT and (include_annotated or not _annotated(data_dir, p))]
    logger.info("%d candidate lines to score", len(candidates))
    if not candidates:
        await engine.close()
        logger.info("nothing to do (all lines annotated?)")
        return 0

    rows: list[tuple[Path, str, float | None]] = []
    try:
        for i, img in enumerate(candidates, 1):
            text, confidence = await engine.ocr_line_scored(img.read_bytes())
            text = " ".join((text or "").split())
            if len(text) > max_chars:
                logger.warning("%s -> %d chars, looks like a repetition loop; discarded", img.name, len(text))
                text = ""
            rows.append((img, text, confidence))
            if i % 25 == 0:
                logger.info("scored %d/%d lines", i, len(candidates))
    finally:
        await engine.close()

    rows.sort(key=lambda r: (r[2] is None, r[2] if r[2] is not None else 1.0))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        fp.write("image\tconfidence\tchars\ttext\n")
        for img, text, conf in rows:
            conf_s = f"{conf:.3f}" if conf is not None else ""
            fp.write(f"{img}\t{conf_s}\t{len(text)}\t{text}\n")
    logger.info("worklist written: %s (%d lines, most uncertain first)", out, len(rows))

    if prefill:
        limit = top if top > 0 else len(rows)
        written = 0
        txt_dir = data_dir / "text"
        txt_dir.mkdir(parents=True, exist_ok=True)
        for img, text, _ in rows[:limit]:
            if len(text) < min_chars:
                continue
            (txt_dir / f"{img.stem}.txt").write_text(text, encoding="utf-8")
            img.with_suffix(".gt.txt").write_text(text, encoding="utf-8")
            written += 1
        logger.info("prefilled %d/%d worst lines with machine transcripts", written, min(limit, len(rows)))
        if written:
            logger.warning("MACHINE labels -- correct %s/images/*.gt.txt by hand before training.", data_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rank/fill line crops by kraken model uncertainty.")
    p.add_argument("--data", type=Path, default=Path("annotation"), help="Dataset dir (images/ + text/).")
    p.add_argument("--out", type=Path, default=None, help="Worklist path (default: <data>/worklist.tsv).")
    p.add_argument("--top", type=int, default=0, help="Prefill only the N most uncertain lines (0 = all).")
    p.add_argument("--prefill", action="store_true", help="Also write machine transcripts for the selected lines.")
    p.add_argument("--all", dest="include_annotated", action="store_true", help="Re-score annotated lines too.")
    p.add_argument("--force", action="store_true",
                   help="Required with `--all --prefill`: overwriting existing transcripts is destructive "
                        "(2026-10-15: this wiped the diacritized GT of the whole corpus).")
    p.add_argument("--min-chars", type=int, default=2, help="Treat shorter output as empty (no .gt.txt written).")
    p.add_argument("--max-chars", type=int, default=300, help="Treat longer output as a repetition loop (empty).")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.prefill and args.include_annotated and not args.force:
        raise SystemExit(
            "refusing to run `--all --prefill`: it overwrites the transcript of every line in the "
            "dataset (this destroyed the diacritized GT corpus on 2026-10-15). Pass --force if you "
            "really mean it, or drop --all to prefill only unannotated lines, optionally with --top N."
        )
    out = args.out or (args.data / "worklist.tsv")
    return asyncio.run(run(args.data, out, args.top, args.prefill, args.include_annotated,
                           args.min_chars, args.max_chars))


if __name__ == "__main__":
    raise SystemExit(main())
