"""Expand honourific ligatures in dataset ground truth into spelled-out phrases.

The corpus GT (extracted from the PDFs) contains Arabic presentation-form
ligatures for honourific formulae, e.g. U+FDFA -> "صلى الله عليه وسلم". A
recognition model can only learn them as ultra-rare single output classes
(1-60 occurrences corpus-wide), which it mostly ends up dropping, and they never
occur in the holdout split, so the metrics ignore them entirely (the
`alphabet mismatch` warning during training).

This tool rewrites a dataset copy with every ligature expanded to its
spelled-out phrase, so the model expresses the same content with ordinary,
frequent letters:

    annotation_nodiac/        -> annotation_canon/      (PNGs hardlinked, GT rewritten)
    train.txt / holdout.txt   -> train_canon.txt / holdout_canon.txt

Ornate Quranic parentheses (U+FD3E/U+FD3F) are punctuation and are kept as-is.
NFKC does NOT decompose most of these (only U+FDFA/FDFB), which is why an
explicit mapping is needed.

Note: expanding ligatures to phrases changes word counts, so CER/WER before and
after canonicalization are not directly comparable -- compare models within the
same dataset version.

Usage:
    .venv/bin/python canonicalize_gt.py --dry-run     # report only
    .venv/bin/python canonicalize_gt.py               # write annotation_canon/ + *_canon.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Presentation-form ligature -> spelled-out phrase (verified against corpus GT).
MAPPING: dict[str, str] = {
    "\ufdfa": "صلى الله عليه وسلم",   # SALLALLAHOU ALAYHE WASALLAM
    "\ufdfb": "جل جلاله",             # JALLAJALALOUHOU
    "\ufdff": "عز وجل",               # AZZA WA JALL
    "\ufd40": "رحمه الله",           # RAHIMAHU ALLAAH
    "\ufd47": "عليه السلام",         # ALAYHI AS-SALAAM
    "\ufd41": "رضي الله عنه",        # RADI ALLAAHU ANH(U)
    "\ufd42": "رضي الله عنها",       # RADI ALLAAHU ANHAA
    "\ufd43": "رضي الله عنهم",       # RADI ALLAAHU ANHUM
    "\ufd44": "رضي الله عنهما",      # RADI ALLAAHU ANHUMAA
    "\ufd49": "عليهما السلام",       # ALAYHIMAA AS-SALAAM
    "\ufdfe": "سبحانه وتعالى",       # SUBHAANAHU WA TAAALAA
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("canonicalize_gt")


def canonicalize(text: str) -> str:
    """Replace honourific ligatures with their spelled-out phrases."""
    if not any(ch in MAPPING for ch in text):
        return text
    return "".join(MAPPING.get(ch, ch) for ch in text)


def rewrite_manifest(src: Path, dst: Path, manifest: Path, out: Path) -> int:
    """Rewrite image paths in a manifest from the source to the destination dataset."""
    src_abs = src.resolve()
    dst_abs = dst.resolve()
    lines = []
    changed = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            p = Path(stripped)
            resolved = p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
            if src_abs == resolved.parents[0] or src_abs in resolved.parents:
                rel = resolved.relative_to(src_abs)
                new_p = dst_abs / rel if p.is_absolute() else Path(os.path.relpath(dst_abs / rel, manifest.parent))
                lines.append(str(new_p))
                changed += 1
                continue
        lines.append(line)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def run(src: Path, dst: Path, train: Path, holdout: Path, dry_run: bool) -> int:
    src_images = src / "images"
    dst_images = dst / "images"
    gt_files = sorted(src_images.glob("*.gt.txt"))
    if not gt_files:
        logger.error("no ground-truth files under %s", src_images)
        return 1

    stats: Counter = Counter()
    changed_files = 0
    chars_before = chars_after = 0
    for gt in gt_files:
        text = gt.read_text(encoding="utf-8")
        new = canonicalize(text)
        chars_before += len(text)
        chars_after += len(new)
        if new != text:
            changed_files += 1
            for ch, phrase in MAPPING.items():
                stats[ch] += text.count(ch)
        if dry_run:
            continue
        dst_images.mkdir(parents=True, exist_ok=True)
        png = gt.with_suffix("")  # foo.gt.txt -> foo, keep the image name below
        image_name = gt.name[: -len(".gt.txt")] + ".png"
        src_png = src_images / image_name
        dst_png = dst_images / image_name
        if src_png.exists() and not dst_png.exists():
            try:
                os.link(src_png, dst_png)  # hardlink: same bytes, no extra disk
            except OSError:
                shutil.copy2(src_png, dst_png)
        (dst_images / gt.name).write_text(new, encoding="utf-8")

    if not dry_run:
        train_written = rewrite_manifest(src, dst, train, ROOT / "train_canon.txt")
        holdout_written = rewrite_manifest(src, dst, holdout, ROOT / "holdout_canon.txt")
    else:
        train_written = holdout_written = 0

    logger.info("files: %d GT files, %d changed", len(gt_files), changed_files)
    for ch, n in stats.most_common():
        logger.info("  U+%04X x%d -> %s", ord(ch), n, MAPPING[ch])
    logger.info("characters: %d -> %d (+%d)", chars_before, chars_after, chars_after - chars_before)
    if dry_run:
        logger.info("dry-run: nothing written")
    else:
        logger.info("dataset written to %s", dst)
        logger.info("manifests: train_canon.txt (%d paths), holdout_canon.txt (%d paths)",
                    train_written, holdout_written)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Expand honourific ligatures in dataset GT.")
    p.add_argument("--src", type=Path, default=ROOT / "annotation_nodiac", help="Source dataset dir.")
    p.add_argument("--dst", type=Path, default=ROOT / "annotation_canon", help="Destination dataset dir.")
    p.add_argument("--train", type=Path, default=ROOT / "train.txt", help="Training manifest to rewrite.")
    p.add_argument("--holdout", type=Path, default=ROOT / "holdout.txt", help="Holdout manifest to rewrite.")
    p.add_argument("--dry-run", action="store_true", help="Report changes without writing anything.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run(args.src, args.dst, args.train, args.holdout, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
