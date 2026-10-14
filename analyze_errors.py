"""Per-line error analysis of a Kraken recognition model on a line manifest.

Runs the trained model over every image listed in a manifest (default:
`holdout.txt`), diffs each prediction against its `<stem>.gt.txt` ground truth
with kraken's own `global_align` / `compute_confusions` helpers, and writes a
JSON report plus a human-readable summary:

  - aggregate character/word error rates (should reproduce the `ketos test`
    numbers for the same model and manifest),
  - insertions / deletions / substitutions split,
  - top confusion pairs and error counts grouped by character category
    (digits, Quranic marks, honourific/presentation forms, spaces, letters),
  - worst lines ranked by errors, to steer the next annotation batch.

Ground truth is whitespace-normalized; the model runs through kraken's standard
inference path (`RecognitionTaskModel.predict`, BiDi reordering on), which
returns logical-order text, so both sides are compared in logical order (the
same text a human reads out of the pipeline).

Usage:
    .venv/bin/python analyze_errors.py
    .venv/bin/python analyze_errors.py --model models/run_x_3e4/best_0.9690.safetensors \
        --manifest holdout.txt --out models/run_x_3e4/error_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "run_x_3e4" / "best_0.9690.safetensors"
DEFAULT_MANIFEST = ROOT / "holdout.txt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("analyze_errors")

_CATEGORY_RANGES = (
    ("quranic mark", ((0x0600, 0x0605), (0x0610, 0x061A), (0x06D6, 0x06ED), (0x08D3, 0x08FF), (0x0670, 0x0670))),
    ("honorific/presentation", ((0xFB50, 0xFD3D), (0xFD50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("ornate paren", ((0xFD3E, 0xFD3F),)),
)


def load_manifest(path: Path) -> list[Path]:
    """Manifest lines -> existing image paths (cwd- or manifest-relative)."""
    files: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if not p.exists():
            p = path.parent / line
        if p.exists():
            files.append(p)
        else:
            logger.warning("manifest entry missing on disk: %s", line)
    return files


def line_segmentation(im) -> "Segmentation":
    """Whole-image bbox segmentation (mirrors kraken's `-s/--no-segmentation` mode)."""
    from kraken.containers import BBoxLine, Segmentation

    return Segmentation(type="bbox",
                        imagename="",
                        text_direction="horizontal-tb",
                        script_detection=False,
                        lines=[BBoxLine(id="line-0", bbox=(0, 0, im.width, im.height))])


def reference_text(raw: str) -> str:
    """GT text with whitespace normalization (as stored: logical order)."""
    from kraken.lib import functional_im_transforms as F_t

    return F_t.text_whitespace_normalize(raw)


def hypothesis_text(raw: str) -> str:
    from kraken.lib import functional_im_transforms as F_t

    return F_t.text_whitespace_normalize(raw)


def character_category(ch: str) -> str:
    """Coarse Unicode category used to attribute errors to content classes."""
    if not ch:
        return "empty"
    if ch == " ":
        return "space"
    cp = ord(ch)
    if 0x0660 <= cp <= 0x0669 or 0x06F0 <= cp <= 0x06F9 or 0x30 <= cp <= 0x39:
        return "digit"
    for name, ranges in _CATEGORY_RANGES:
        if any(lo <= cp <= hi for lo, hi in ranges):
            return name
    if unicodedata.category(ch).startswith("P"):
        return "punctuation"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
        return "arabic letter"
    return "other"


def _display(ch: str) -> str:
    if ch == "":
        return "{  }"
    if ch == " ":
        return "{ SPACE }"
    return "{" + " " + ch + " " + "}"


def analyze(model_path: Path, manifest: Path, device: str | None, batch_size: int) -> dict:
    from kraken.configs import RecognitionInferenceConfig
    from kraken.lib.dataset.utils import compute_confusions, global_align
    from kraken.lib.util import open_image
    from kraken.tasks import RecognitionTaskModel

    model = RecognitionTaskModel.load_model(str(model_path))
    kwargs: dict = {"batch_size": batch_size, "num_line_workers": 0, "padding": 16,
                    "bidi_reordering": True, "text_direction": "horizontal-tb"}
    if device:
        kwargs["device"] = device
    config = RecognitionInferenceConfig(**kwargs)

    files = load_manifest(manifest)
    if not files:
        raise SystemExit(f"no existing images in manifest {manifest}")

    per_line: list[dict] = []
    counts: Counter = Counter()
    gt_chars: Counter = Counter()
    total_errors = total_chars = 0
    total_word_errors = total_words = 0
    exact = 0

    for i, img_path in enumerate(files, 1):
        gt_path = img_path.with_suffix(".gt.txt")
        if not gt_path.exists():
            logger.warning("no ground truth for %s, skipped", img_path.name)
            continue
        ref = reference_text(gt_path.read_text(encoding="utf-8"))
        if not ref:
            logger.warning("empty ground truth for %s, skipped", img_path.name)
            continue
        im = open_image(img_path)
        records = list(model.predict(im=im, segmentation=line_segmentation(im), config=config))
        hyp = hypothesis_text(" ".join(r.prediction for r in records))

        errors, algn_ref, algn_hyp = global_align(ref, hyp)
        line_counts, _, _, _, _ = compute_confusions(algn_ref, algn_hyp)
        counts.update(line_counts)
        gt_chars.update(character_category(ch) for ch in ref)

        w_dist, _, _ = global_align(ref.split(), hyp.split())
        conf = None
        try:
            values = records[0].confidences if records else None
            if values:
                conf = round(float(sum(values) / len(values)), 4)
        except Exception:
            conf = None

        total_errors += errors
        total_chars += len(ref)
        total_word_errors += w_dist
        total_words += len(ref.split())
        if errors == 0:
            exact += 1
        per_line.append({
            "id": img_path.stem,
            "chars": len(ref),
            "errors": errors,
            "cer": round(errors / len(ref), 4) if ref else 0.0,
            "word_errors": w_dist,
            "words": len(ref.split()),
            "confidence": conf,
            "reference": ref,
            "hypothesis": hyp,
        })
        if i % 20 == 0:
            logger.info("processed %d/%d lines", i, len(files))

    deletions = sum(n for (u, v), n in counts.items() if v == "" and u != "")
    insertions = sum(n for (u, v), n in counts.items() if u == "" and v != "")
    substitutions = sum(n for (u, v), n in counts.items() if u != "" and v != "" and u != v)

    missed: Counter = Counter()
    for (u, v), n in counts.items():
        if u == v or u == "":
            continue
        missed[character_category(u)] += n
    category_rows = []
    for cat, total in gt_chars.most_common():
        m = missed.get(cat, 0)
        category_rows.append({"category": cat, "count": total, "missed": m,
                              "right_pct": round(100.0 * (total - m) / total, 2) if total else 100.0})

    confusions = sorted(((u, v, n) for (u, v), n in counts.items() if u != v),
                        key=lambda t: -t[2])

    aggregate = {
        "lines": len(per_line),
        "lines_exact": exact,
        "characters": total_chars,
        "errors": total_errors,
        "character_accuracy": round(1.0 - total_errors / total_chars, 4) if total_chars else 0.0,
        "words": total_words,
        "word_errors": total_word_errors,
        "word_accuracy": round(1.0 - total_word_errors / total_words, 4) if total_words else 0.0,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
    }
    return {
        "model": str(model_path),
        "manifest": str(manifest),
        "aggregate": aggregate,
        "categories": category_rows,
        "top_confusions": [{"reference": u, "hypothesis": v, "count": n} for u, v, n in confusions],
        "per_line": sorted(per_line, key=lambda r: -r["errors"]),
    }


def print_summary(report: dict, top: int, worst: int) -> None:
    agg = report["aggregate"]
    print("=== error analysis ===")
    print(f"model: {report['model']}")
    print(f"manifest: {report['manifest']}")
    print(f"lines: {agg['lines']} ({agg['lines_exact']} exact)")
    print(f"characters: {agg['characters']}  errors: {agg['errors']}  "
          f"character accuracy: {agg['character_accuracy']:.2%}")
    print(f"words: {agg['words']}  errors: {agg['word_errors']}  "
          f"word accuracy: {agg['word_accuracy']:.2%}")
    print(f"insertions: {agg['insertions']}  deletions: {agg['deletions']}  "
          f"substitutions: {agg['substitutions']}")
    print("\n--- per category (ground-truth content) ---")
    print(f"{'category':<24} {'count':>7} {'missed':>7} {'%right':>7}")
    for row in report["categories"]:
        print(f"{row['category']:<24} {row['count']:>7} {row['missed']:>7} {row['right_pct']:>7.2f}")
    print(f"\n--- top {top} mismatches (reference - hypothesis) ---")
    for item in report["top_confusions"][:top]:
        print(f"{item['count']:>4}   {_display(item['reference'])} - {_display(item['hypothesis'])}")
    print(f"\n--- worst {worst} lines ---")
    for row in report["per_line"][:worst]:
        conf = f" conf={row['confidence']:.2f}" if row["confidence"] is not None else ""
        print(f"{row['id']}: {row['errors']}/{row['chars']} char errors{conf}")
        print(f"   ref: {row['reference'][:110]}")
        print(f"   hyp: {row['hypothesis'][:110]}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-line Kraken error analysis on a line manifest.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Recognition model (.safetensors/.mlmodel).")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Text file with one image path per line.")
    p.add_argument("--out", type=Path, default=None, help="JSON report path (default: <model dir>/error_report.json).")
    p.add_argument("--device", default=None, help="Torch device (default: kraken's auto).")
    p.add_argument("-B", "--batch-size", type=int, default=16, help="Lines per forward pass.")
    p.add_argument("--top", type=int, default=25, help="Number of confusion pairs to print.")
    p.add_argument("--worst", type=int, default=10, help="Number of worst lines to print.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    report = analyze(args.model, args.manifest, args.device, args.batch_size)
    out = args.out or args.model.parent / "error_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report, args.top, args.worst)
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
