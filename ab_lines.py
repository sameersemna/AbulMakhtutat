"""A/B one or more OCR engines on a hand-corrected *line* manifest (CER/WER).

Unlike `eval_ocr.py` (page-level probe set), this scores each engine's
`ocr_line` path on the same hand-corrected line holdout that gates model
training, so the comparison is against trustworthy labels. Engines are built
through `build_ocr_engine` with the backend overridden per engine:

    VISION_BACKEND=kraken .venv/bin/python ab_lines.py --engines kraken
    .venv/bin/python ab_lines.py --engines kraken,ollama --manifest holdout.txt
    .venv/bin/python ab_lines.py --engines ollama --limit 20

Metrics are character/word-weighted (total errors / total characters), like
`ketos test`, in verbatim and Arabic-normalized modes.

Output: a console table plus a JSON report (default probe/ab_lines_report.json).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from eval_ocr import _collapse, _edit_distance
from text_cleanup import normalize_arabic

ROOT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ab_lines")

_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def load_manifest(path: Path) -> list[Path]:
    return [Path(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


async def run_engine(engine_name: str, files: list[Path]) -> dict:
    from config import get_config
    from hf_vlm_engine import build_ocr_engine
    from thermal import ThermalGuard

    cfg = get_config()
    cfg.pipeline.workers = 1
    cfg.vision.backend = engine_name
    engine = build_ocr_engine(cfg)
    actual = type(engine).__name__
    logger.info("=== engine %s (requested %s) ===", actual, engine_name)

    thermal = ThermalGuard(cfg)
    rows: list[dict] = []
    chars_v = errs_v = chars_n = errs_n = 0
    words_v = werr_v = words_n = werr_n = 0
    try:
        for i, img in enumerate(files, 1):
            gt = img.with_suffix(".gt.txt")
            if not gt.exists() or not gt.read_text(encoding="utf-8").strip():
                continue
            await thermal.wait_if_hot()
            hyp, _ = await engine.ocr_line(img.read_bytes())
            ref = gt.read_text(encoding="utf-8")

            ref_v, hyp_v = _collapse(ref), _collapse(hyp)
            ref_n, hyp_n = normalize_arabic(ref), normalize_arabic(hyp)
            dv, dn = _edit_distance(ref_v, hyp_v), _edit_distance(ref_n, hyp_n)
            wv = _edit_distance(ref_v.split(), hyp_v.split())
            wn = _edit_distance(ref_n.split(), hyp_n.split())
            chars_v += len(ref_v); errs_v += dv
            chars_n += len(ref_n); errs_n += dn
            words_v += len(ref_v.split()); werr_v += wv
            words_n += len(ref_n.split()); werr_n += wn
            rows.append({"id": img.stem, "chars": len(ref_v), "errors": dv,
                         "hyp_chars": len(hyp_v)})
            if i % 25 == 0:
                logger.info("%s: processed %d/%d lines", engine_name, i, len(files))
    finally:
        await engine.close()

    return {
        "engine": actual,
        "requested": engine_name,
        "lines": len(rows),
        "characters": chars_v,
        "errors": errs_v,
        "cer_verbatim": round(errs_v / chars_v, 4) if chars_v else 0.0,
        "cer_normalized": round(errs_n / chars_n, 4) if chars_n else 0.0,
        "wer_verbatim": round(werr_v / words_v, 4) if words_v else 0.0,
        "wer_normalized": round(werr_n / words_n, 4) if words_n else 0.0,
        "per_line": rows,
    }


async def run(engines: list[str], manifest: Path, limit: int, out: Path) -> int:
    files = [p for p in load_manifest(manifest) if p.suffix.lower() in _IMAGE_EXT]
    if limit > 0:
        files = files[:limit]
    if not files:
        logger.error("no images in manifest %s", manifest)
        return 1

    results = []
    for name in engines:
        results.append(await run_engine(name, files))

    print(f"\n{'engine':<12} {'lines':>5} {'chars':>6} {'CERv':>7} {'CERn':>7} {'WERv':>7} {'WERn':>7}")
    for r in results:
        print(f"{r['engine']:<12} {r['lines']:>5} {r['characters']:>6} "
              f"{r['cer_verbatim']:>7.4f} {r['cer_normalized']:>7.4f} "
              f"{r['wer_verbatim']:>7.4f} {r['wer_normalized']:>7.4f}")

    report = {"manifest": str(manifest), "limit": limit, "engines": results}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A/B OCR engines on a line manifest (CER/WER).")
    p.add_argument("--engines", default="kraken,ollama", help="Comma-separated engine names.")
    p.add_argument("--manifest", type=Path, default=ROOT / "holdout.txt", help="Line manifest (image paths).")
    p.add_argument("--limit", type=int, default=0, help="Score at most N lines (0 = all).")
    p.add_argument("--out", type=Path, default=ROOT / "probe" / "ab_lines_report.json", help="JSON report path.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    return asyncio.run(run(engines, args.manifest, args.limit, args.out))


if __name__ == "__main__":
    raise SystemExit(main())
