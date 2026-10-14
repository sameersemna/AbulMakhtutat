"""Pre-annotate line crops with the OCR engine to bootstrap ground truth (Task 6).

Runs the configured vision engine's line OCR (`ocr_line`: line prompt + token cap)
over each line image that has no non-empty transcript and writes the result to BOTH places the trainers expect:
    annotation/text/<stem>.txt        (used by train_lora.py)
    annotation/images/<stem>.gt.txt   (used by ketos 'path' mode)

`<stem>` is the image filename without its extension (e.g. p0031_l001.png ->
p0031_l001.gt.txt).

*** These are MACHINE labels. Correct them by hand before training. ***
Using uncorrected model output as ground truth teaches the model its own errors
(confirmation bias). Prefer a model that is NOT the one you plan to fine-tune,
and always review the output.

Usage:
    .venv/bin/python prefill_line_gt.py --data annotation/
    .venv/bin/python prefill_line_gt.py --data annotation/ --force   # overwrite existing
    VISION_BACKEND=hf .venv/bin/python prefill_line_gt.py --data annotation/
    VISION_BACKEND=kraken .venv/bin/python prefill_line_gt.py --data annotation/  # in-repo kraken model
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("prefill_line_gt")

_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


async def run(data_dir: Path, force: bool, min_chars: int, max_chars: int) -> int:
    from config import get_config
    from hf_vlm_engine import build_ocr_engine

    img_dir = data_dir / "images"
    txt_dir = data_dir / "text"
    if not img_dir.is_dir():
        logger.error("image directory not found: %s", img_dir)
        return 1
    txt_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_config()
    cfg.pipeline.workers = 1
    engine = build_ocr_engine(cfg)
    if cfg.vision.backend == "hf":
        model_desc = cfg.vision.hf_model
    elif cfg.vision.backend == "kraken":
        model_desc = cfg.vision.kraken_model
    else:
        model_desc = cfg.ollama.vision_model
    logger.info("engine: %s | model: %s", type(engine).__name__, model_desc)

    written = empty = skipped = synced = 0
    try:
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in _IMAGE_EXT:
                continue
            txt = txt_dir / f"{img.stem}.txt"
            gt = img.with_suffix(".gt.txt")

            existing = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
            if existing and not force:
                # Already transcribed: just make sure ketos' .gt.txt exists too.
                if not (gt.exists() and gt.read_text(encoding="utf-8").strip()):
                    gt.write_text(existing, encoding="utf-8")
                    synced += 1
                skipped += 1
                continue

            text, model = await engine.ocr_line(img.read_bytes())
            text = " ".join((text or "").split())  # one line: no newlines or runs of spaces
            if len(text) > max_chars:
                # A printed line is ~100-150 chars; far longer output is a repetition loop.
                logger.warning("%s -> %d chars, looks like a repetition loop; discarded", img.name, len(text))
                text = ""
            txt.write_text(text, encoding="utf-8")

            if len(text) < min_chars:
                # Never leave an empty .gt.txt behind: ketos aborts on empty ground truth.
                if gt.exists() and not gt.read_text(encoding="utf-8").strip():
                    gt.unlink()
                empty += 1
                logger.warning(
                    "%s -> EMPTY/short (model=%s); no .gt.txt written — transcribe or drop this line",
                    img.name, model or "?",
                )
            else:
                gt.write_text(text, encoding="utf-8")
                written += 1
                logger.info("%s -> %s", img.name, text[:70])
    finally:
        await engine.close()

    logger.info(
        "done: %d written, %d empty/short, %d skipped, %d .gt.txt synced", written, empty, skipped, synced
    )
    if written == 0 and skipped == 0:
        logger.error(
            "no non-empty transcripts produced. The vision model may be unavailable or "
            "refusing the crops — check `ollama list` / `VISION_BACKEND`, and try a different "
            "model or the HF backend, then re-run."
        )
        return 2
    if written:
        logger.warning("These are MACHINE labels — correct annotation/images/*.gt.txt by hand before training.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-annotate line crops to bootstrap ground truth.")
    p.add_argument("--data", type=Path, default=Path("annotation"), help="Dataset dir (images/ + text/).")
    p.add_argument("--force", action="store_true", help="Overwrite existing non-empty transcripts.")
    p.add_argument("--min-chars", type=int, default=2, help="Treat shorter output as empty.")
    p.add_argument("--max-chars", type=int, default=300, help="Treat longer output as a repetition loop (empty).")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(run(args.data, args.force, args.min_chars, args.max_chars))


if __name__ == "__main__":
    raise SystemExit(main())
