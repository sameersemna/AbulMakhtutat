"""Cut PDF pages into single-line crops for the annotation corpus (Task 6).

Renders each page, detects text lines with Surya, and saves one PNG per line to
    annotation/images/<pdf stem>_p<page>_l<line>.png
ready for `prefill_line_gt.py` and hand correction.

A line model needs exactly one text line per crop, so detector boxes are
filtered before saving (rejections are logged with the reason):
  - too short  (< --min-height x median line height): slivers, diacritic strips
  - too tall   (> --max-height x median): merged lines; split at the ink valleys
                between rows when the split is clean, otherwise rejected
  - too narrow (< --min-width px) or nearly blank

Pages listed in the eval manifest (probe/manifest.json) are skipped by default:
training on eval pages makes the before/after CER meaningless.

Usage:
    .venv/bin/python make_line_crops.py --pdf input/rabee/majmua/1.pdf --pages 40-79
    .venv/bin/python make_line_crops.py --pdf input/rabee/majmua/9.pdf --pages 100-120 --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from pdf_processor import parse_page_range

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("make_line_crops")

_INK = 120  # grayscale threshold for "ink" pixels


def eval_pages(manifest: Path, pdf: Path) -> set[int]:
    """Return the page numbers of `pdf` that the eval manifest uses."""
    if not manifest.exists():
        return set()
    entries = json.loads(manifest.read_text(encoding="utf-8")).get("entries", [])
    target = pdf.resolve()
    return {int(e["page"]) for e in entries if Path(e["pdf"]).resolve() == target}


def render(doc: pymupdf.Document, page: int, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    pix = doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def row_profile(gray: np.ndarray) -> np.ndarray:
    """Ink pixels per row, lightly smoothed."""
    ink = (gray < _INK).sum(1).astype(float)
    return np.convolve(ink, np.ones(5) / 5, "same")


def split_rows(gray: np.ndarray, n: int) -> list[tuple[int, int]] | None:
    """Split a merged box into `n` rows at the n-1 deepest ink valleys.

    Returns None when a valley is not clearly emptier than the rows it
    separates, i.e. the split would cut through text.
    """
    prof = row_profile(gray)
    h = len(prof)
    cuts = [0]
    for k in range(1, n):
        centre = h * k // n
        lo, hi = max(1, centre - h // (2 * n)), min(h - 1, centre + h // (2 * n))
        y = lo + int(np.argmin(prof[lo:hi]))
        if prof[y] > 0.15 * prof.max():
            return None
        cuts.append(y)
    cuts.append(h)
    return list(zip(cuts, cuts[1:]))


def crop_page(
    img: Image.Image, boxes: list[tuple[int, int, int, int]], args: argparse.Namespace
) -> tuple[list[Image.Image], list[str]]:
    """Filter/split detector boxes; return (line crops, rejection reasons)."""
    if not boxes:
        return [], ["no lines detected"]
    widest = max(b[2] - b[0] for b in boxes)
    body = [b[3] - b[1] for b in boxes if b[2] - b[0] > 0.5 * widest] or [b[3] - b[1] for b in boxes]
    med = float(np.median(body))
    pad_y = int(args.pad * med)
    W, H = img.size

    crops: list[Image.Image] = []
    rejects: list[str] = []
    for x0, y0, x1, y1 in boxes:
        h, w = y1 - y0, x1 - x0
        tag = f"box y={y0}-{y1} h={h} (median {med:.0f})"
        if w < args.min_width:
            rejects.append(f"{tag}: too narrow ({w}px)")
            continue
        if h < args.min_height * med:
            rejects.append(f"{tag}: too short (sliver / diacritics only)")
            continue
        region = (max(0, x0 - 10), max(0, y0 - pad_y), min(W, x1 + 10), min(H, y1 + pad_y))
        piece = img.crop(region)
        gray = np.asarray(piece.convert("L"))
        if (gray < _INK).mean() < 0.005:
            rejects.append(f"{tag}: nearly blank")
            continue
        if h > args.max_height * med:
            n = max(2, round(h / med))
            rows = split_rows(gray, n)
            if rows is None:
                rejects.append(f"{tag}: {n} merged lines, no clean split")
                continue
            crops.extend(piece.crop((0, a, piece.width, b)) for a, b in rows)
            continue
        crops.append(piece)
    return crops, rejects


def main() -> int:
    p = argparse.ArgumentParser(description="Cut PDF pages into single-line crops.")
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--pages", required=True, help='Page spec, e.g. "40-79" or "40,42,50-60".')
    p.add_argument("--out", type=Path, default=Path("annotation/images"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--device", default="cuda", help="Surya device: cuda | cpu.")
    p.add_argument("--manifest", type=Path, default=Path("probe/manifest.json"))
    p.add_argument("--include-eval-pages", action="store_true", help="Do not skip eval manifest pages.")
    p.add_argument("--min-height", type=float, default=0.6, help="Reject boxes shorter than this x median.")
    p.add_argument("--max-height", type=float, default=1.6, help="Split/reject boxes taller than this x median.")
    p.add_argument("--min-width", type=int, default=60)
    p.add_argument("--pad", type=float, default=0.12, help="Vertical padding as a fraction of median height.")
    p.add_argument("--dry-run", action="store_true", help="Detect and report only; write nothing.")
    args = p.parse_args()

    from surya_detection import detect_line_boxes_surya

    doc = pymupdf.open(args.pdf)
    pages = parse_page_range(args.pages, doc.page_count)
    skip = set() if args.include_eval_pages else eval_pages(args.manifest, args.pdf)
    args.out.mkdir(parents=True, exist_ok=True)

    saved = rejected = 0
    for page in pages:
        if page in skip:
            logger.info("p%d: eval manifest page, skipped", page)
            continue
        prefix = f"{args.pdf.stem}_p{page:04d}_l"
        if any(args.out.glob(f"{prefix}*.png")):
            logger.info("p%d: crops already exist, skipped", page)
            continue
        img = render(doc, page, args.dpi)
        crops, rejects = crop_page(img, detect_line_boxes_surya(img, device=args.device), args)
        for r in rejects:
            logger.info("p%d: rejected %s", page, r)
        if not args.dry_run:
            for i, c in enumerate(crops, 1):
                c.save(args.out / f"{prefix}{i:03d}.png")
        saved += len(crops)
        rejected += len(rejects)
        logger.info("p%d: %d lines, %d rejected", page, len(crops), len(rejects))

    logger.info("done: %d line crops %s, %d boxes rejected", saved, "found" if args.dry_run else "saved", rejected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
