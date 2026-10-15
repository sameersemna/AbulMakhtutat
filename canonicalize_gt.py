"""Build a Kraken training dataset from the primary annotated corpus.

Two build modes:

* **Canonical** (default): honourific ligatures are expanded to spelled-out
  phrases. The corpus GT contains Arabic presentation-form ligatures, e.g.
  U+FDFA -> "صلى الله عليه وسلم"; a recognition model can only learn them as
  ultra-rare single output classes (1-60 occurrences corpus-wide) which it mostly
  drops. Ornate Quranic parentheses (U+FD3E/U+FD3F) are punctuation and are kept.
  NFKC decomposes only two of the ligatures (U+FDFA/FDFB), hence the explicit map.
* **Ligature-preserving** (`--no-expand-ligatures`): harakat stripped, ligatures
  kept as single code points. This is the convention of `annotation_nodiac` /
  `train.txt` / `holdout.txt` and what the shipped model is trained on; the
  ligatures are expanded *after* recognition by `text_cleanup.normalize_arabic`
  on the search path. (Training canonical directly was measured worse: the
  spelled-out phrases are too sparse for a small CTC model — 2026-10-14,
  `run_x_canon` 94.67% vs `run_x_3e4` 96.92% char.)

The tool always writes PNGs (hardlinked, no extra disk) and rewritten GT into
`<dst>/images/`, then rewrites the manifests so they point at `<dst>`. With
`--append-new`, annotated files present in `<dst>` but missing from both
manifests are appended to the *training* manifest (the holdout stays frozen) --
that is how hand-corrected additions to `annotation/` are folded in.

Usage:
    # canonical build (from the primary diacritized corpus; strips harakat first)
    .venv/bin/python canonicalize_gt.py --src annotation --strip-diacritics

    # fold-in build for the shipped (ligature-preserving) model -- rebuilds
    # annotation_nodiac in place and adds the new hand-corrected lines to train:
    .venv/bin/python canonicalize_gt.py --src annotation --strip-diacritics \\
        --no-expand-ligatures --dst annotation_nodiac \\
        --train train.txt --holdout holdout.txt \\
        --train-out train.txt --holdout-out holdout.txt --append-new

    .venv/bin/python canonicalize_gt.py --dry-run     # report only
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from text_cleanup import HONOURIFIC_LIGATURES, expand_honourifics

ROOT = Path(__file__).resolve().parent

# Presentation-form ligature -> spelled-out phrase. Single source of truth lives
# in `text_cleanup.HONOURIFIC_LIGATURES` (shared with the runtime search path so
# the dataset and the indexer can never drift apart). Alias kept for callers/tests.
MAPPING: dict[str, str] = HONOURIFIC_LIGATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("canonicalize_gt")


def canonicalize(text: str) -> str:
    """Replace honourific ligatures with their spelled-out phrases."""
    return expand_honourifics(text)


# Diacritic spans matching the original corpus build (keeps U+0670, the Quranic
# superscript alef).
_TASHKEEL_RE = re.compile(r"[\u064B-\u065F\u0610-\u061A\u06D6-\u06ED]")


def strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritics (harakat) from text."""
    return _TASHKEEL_RE.sub("", text)


def rewrite_manifest(dst: Path, manifest: Path, out: Path) -> int:
    """Point manifest entries at the destination dataset (matched by file name).

    Works regardless of which dataset prefix the manifest uses (e.g.
    `annotation_nodiac/...` or `annotation/...`): entries whose file exists in
    `<dst>/images/` are rewritten, everything else is passed through.
    """
    dst_images = dst / "images"
    lines = []
    changed = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            target = dst_images / Path(stripped).name
            if target.exists():
                new_p = target if Path(stripped).is_absolute() else Path(os.path.relpath(target, manifest.parent))
                lines.append(str(new_p))
                changed += 1
                continue
        lines.append(line)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def append_new_files(dst: Path, train_out: Path, holdout_out: Path) -> int:
    """Append dataset files missing from both manifests to the training manifest.

    Only files present in `<dst>/images/` are considered, matched by file name,
    so this is safe to re-run: already-listed files (and anything in the holdout)
    are never touched. Returns the number of appended paths.
    """
    dst_images = dst / "images"
    known = set()
    for manifest in (train_out, holdout_out):
        if manifest.is_file():
            known.update(Path(line.strip()).name for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    new_entries = []
    for gt in sorted(dst_images.glob("*.gt.txt")):
        image_name = gt.name[: -len(".gt.txt")] + ".png"
        if image_name in known:
            continue
        target = dst_images / image_name
        new_entries.append(str(target if train_out.is_absolute() else Path(os.path.relpath(target, train_out.parent))))
    if new_entries:
        existing = train_out.read_text(encoding="utf-8") if train_out.is_file() else ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        train_out.write_text(existing + "\n".join(new_entries) + "\n", encoding="utf-8")
    return len(new_entries)


def run(src: Path, dst: Path, train: Path, holdout: Path, dry_run: bool, strip_diacritics: bool = False,
        expand_ligatures: bool = True, append_new: bool = False,
        train_out: Path | None = None, holdout_out: Path | None = None) -> int:
    src_images = src / "images"
    dst_images = dst / "images"
    gt_files = sorted(src_images.glob("*.gt.txt"))
    if not gt_files:
        logger.error("no ground-truth files under %s", src_images)
        return 1

    train_out = train_out or ROOT / "train_canon.txt"
    holdout_out = holdout_out or ROOT / "holdout_canon.txt"

    stats: Counter = Counter()
    changed_files = 0
    chars_before = chars_after = 0
    for gt in gt_files:
        text = gt.read_text(encoding="utf-8")
        if strip_diacritics:
            text = strip_tashkeel(text)
        new = canonicalize(text) if expand_ligatures else text
        chars_before += len(text)
        chars_after += len(new)
        if new != text:
            changed_files += 1
            for ch, phrase in MAPPING.items():
                stats[ch] += text.count(ch)
        if dry_run:
            continue
        dst_images.mkdir(parents=True, exist_ok=True)
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
        train_written = rewrite_manifest(dst, train, train_out)
        holdout_written = rewrite_manifest(dst, holdout, holdout_out)
        appended = append_new_files(dst, train_out, holdout_out) if append_new else 0
        if appended:
            logger.warning("%d new file(s) appended to %s -- these must be HAND-CORRECTED "
                           "ground truth; machine-generated prefills are not training data",
                           appended, train_out.name)
    else:
        train_written = holdout_written = appended = 0

    logger.info("files: %d GT files, %d changed%s", len(gt_files), changed_files,
                "" if expand_ligatures else " (ligatures kept)")
    for ch, n in stats.most_common():
        logger.info("  U+%04X x%d -> %s", ord(ch), n, MAPPING[ch])
    logger.info("characters: %d -> %d (+%d)", chars_before, chars_after, chars_after - chars_before)
    if dry_run:
        logger.info("dry-run: nothing written")
    else:
        logger.info("dataset written to %s", dst)
        logger.info("manifests: %s (%d rewritten), %s (%d rewritten), %d new appended",
                    train_out.name, train_written, holdout_out.name, holdout_written, appended)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a Kraken dataset from the primary annotated corpus.")
    p.add_argument("--src", type=Path, default=ROOT / "annotation_nodiac", help="Source dataset dir.")
    p.add_argument("--dst", type=Path, default=ROOT / "annotation_canon", help="Destination dataset dir.")
    p.add_argument("--train", type=Path, default=ROOT / "train.txt", help="Training manifest to rewrite.")
    p.add_argument("--holdout", type=Path, default=ROOT / "holdout.txt", help="Holdout manifest to rewrite.")
    p.add_argument("--train-out", type=Path, default=None,
                   help="Training manifest to write (default: train_canon.txt).")
    p.add_argument("--holdout-out", type=Path, default=None,
                   help="Holdout manifest to write (default: holdout_canon.txt).")
    p.add_argument("--append-new", action="store_true",
                   help="Append files present in <dst> but missing from both manifests to the "
                        "training manifest (the holdout stays frozen). For folding in newly "
                        "annotated lines.")
    p.add_argument("--dry-run", action="store_true", help="Report changes without writing anything.")
    p.add_argument("--strip-diacritics", action="store_true",
                   help="Remove harakat/tashkeel (build from the primary annotation/ corpus).")
    p.add_argument("--no-expand-ligatures", action="store_true",
                   help="Keep honourific ligatures as-is (the annotation_nodiac / shipped-model "
                        "convention; expanded post-OCR by text_cleanup.normalize_arabic).")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run(args.src, args.dst, args.train, args.holdout, args.dry_run, args.strip_diacritics,
               expand_ligatures=not args.no_expand_ligatures, append_new=args.append_new,
               train_out=args.train_out, holdout_out=args.holdout_out)


if __name__ == "__main__":
    raise SystemExit(main())
