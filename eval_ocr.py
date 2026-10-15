"""Probe-set OCR evaluation harness: CER / WER (+ optional overlay LineMatch).

Scores an OCR engine's output against a small ground-truth probe set built from
our own corpus. This is the metric gate that lets us safely switch models or
enable preprocessing (see documentation/08_IMPLEMENTATION_PLAN.md, Task 1).

Two scoring modes per pair:
  - verbatim   : counts diacritics and punctuation (only whitespace collapsed)
  - normalized : Arabic-normalized via `text_cleanup.normalize_arabic`
                 (presentation forms, digits, orthographic variants, diacritics)

The OCR output is cached per entry so re-scoring never re-runs the model.

Usage:
    # Run the current Ollama engine on the probe set, then score
    .venv/bin/python eval_ocr.py

    # Re-score previously cached OCR output without re-running the model
    .venv/bin/python eval_ocr.py --score-only

    # Export current OCR output as ground-truth templates for manual correction
    .venv/bin/python eval_ocr.py --export-templates

    # Run a specific backend (overrides VISION_BACKEND for the run):
    .venv/bin/python eval_ocr.py --engine hf
    .venv/bin/python eval_ocr.py --engine kraken   # in-repo kraken model (VISION_KRAKEN_MODEL)

    # Overlay LineMatch is computed automatically for manifest entries that
    # declare "overlay_pdf" and "overlay_page"; no flag is needed.

Metric notes:
  - CER/WER are reported as fractions (0.0 = perfect). Lower is better.
  - "normalized" CER is the fair comparison across engines, because some engines
    silently drop diacritics (which lowers verbatim CER artificially).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Sequence

import pymupdf

from config import get_config
from text_cleanup import normalize_arabic

try:  # optional C-accelerated edit distance (falls back to pure Python)
    from rapidfuzz.distance import Levenshtein as _rapidfuzz_lev  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _rapidfuzz_lev = None

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "probe" / "manifest.json"
DEFAULT_CACHE = ROOT / "probe" / "_ocr_cache"
DEFAULT_REPORT = ROOT / "probe" / "eval_report.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval_ocr")


# --- Path helpers -----------------------------------------------------------


def _resolve(path_str: str, manifest_dir: Path) -> Path:
    """Resolve a manifest path: absolute as-is, else cwd-relative, else manifest-relative."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (manifest_dir / p).resolve()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load and validate the probe manifest; returns the entry list."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest {path} has no 'entries' list")
    for i, e in enumerate(entries):
        for key in ("id", "pdf", "page"):
            if key not in e:
                raise ValueError(f"Manifest entry #{i} missing '{key}': {e}")
    return entries


# --- Edit distance / metrics ------------------------------------------------


def _edit_distance(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> int:
    """Levenshtein distance between two sequences.

    Uses rapidfuzz (C-accelerated) for string inputs when available; otherwise an
    O(n*m) pure-Python DP. Pass strings for full-page CER and token lists for WER.
    """
    if seq_a == seq_b:
        return 0
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)
    if _rapidfuzz_lev is not None and isinstance(seq_a, str) and isinstance(seq_b, str):
        return int(_rapidfuzz_lev.distance(seq_a, seq_b))

    prev = list(range(len(seq_b) + 1))
    for i, ca in enumerate(seq_a, 1):
        cur = [i]
        for j, cb in enumerate(seq_b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _cer(ref: str, hyp: str) -> float:
    """Character error rate (fraction), 0.0 = perfect."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def _wer(ref: str, hyp: str) -> float:
    """Word error rate (fraction), 0.0 = perfect."""
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def _collapse(text: str) -> str:
    """Whitespace-only normalization (used for verbatim scoring)."""
    return re.sub(r"\s+", " ", text).strip()


def score_pair(ref_raw: str, hyp_raw: str) -> dict[str, float | int]:
    """Score one reference/hypothesis pair in both verbatim and normalized modes."""
    ref_v, hyp_v = _collapse(ref_raw), _collapse(hyp_raw)
    ref_n, hyp_n = normalize_arabic(ref_raw), normalize_arabic(hyp_raw)
    return {
        "cer_verbatim": round(_cer(ref_v, hyp_v), 4),
        "wer_verbatim": round(_wer(ref_v, hyp_v), 4),
        "cer_normalized": round(_cer(ref_n, hyp_n), 4),
        "wer_normalized": round(_wer(ref_n, hyp_n), 4),
        "ref_chars": len(ref_n),
        "hyp_chars": len(hyp_n),
    }


# --- OCR execution ----------------------------------------------------------


def _render_page(doc: pymupdf.Document, page_number: int, dpi: int) -> bytes:
    """Render a 1-based PDF page to PNG bytes at the given DPI."""
    page = doc[page_number - 1]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


async def _run_ocr(
    entries: list[dict[str, Any]], manifest_dir: Path, cache_dir: Path, force: bool, engine_name: str
) -> str:
    """OCR each probe entry (unless cached) and write raw model text to the cache.

    Returns the ACTUAL backend used ("ollama" or "hf"), after any fail-soft fallback.
    """
    # Imports are local so that --score-only / --export-templates never need a model.
    from hf_vlm_engine import HFVisionEngine, build_ocr_engine
    from thermal import ThermalGuard

    try:
        from kraken_engine import KrakenEngine
    except ImportError:  # kraken unavailable: --engine kraken will fail-soft via the factory
        KrakenEngine = None  # type: ignore[assignment,misc]

    config = get_config()
    config.pipeline.workers = 1  # thermal safety (see 06_AGENT_GUIDE.md)
    if engine_name in ("ollama", "hf", "kraken"):
        config.vision.backend = engine_name  # make --engine actually select the backend
    engine = build_ocr_engine(config)
    if KrakenEngine is not None and isinstance(engine, KrakenEngine):
        actual_backend = "kraken"
    elif isinstance(engine, HFVisionEngine):
        actual_backend = "hf"
    else:
        actual_backend = "ollama"
    thermal = ThermalGuard(config)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        for entry in entries:
            out_path = cache_dir / f"{entry['id']}.txt"
            if out_path.exists() and not force:
                logger.info("cached: %s", entry["id"])
                continue
            pdf_path = _resolve(entry["pdf"], manifest_dir)
            if not pdf_path.exists():
                logger.warning("skipping %s: PDF not found: %s", entry["id"], pdf_path)
                continue
            doc = pymupdf.open(str(pdf_path))
            image = _render_page(doc, int(entry["page"]), config.pipeline.dpi)
            doc.close()
            await thermal.wait_if_hot()
            text, model = await engine.ocr_page(image)
            out_path.write_text(text, encoding="utf-8")
            logger.info("ocr: %s (page %s) -> %d chars via %s", entry["id"], entry["page"], len(text), model or "?")
    finally:
        await engine.close()
    if isinstance(engine, HFVisionEngine) and getattr(engine, "_fallback", None) is not None:
        actual_backend = "hf (fell back to ollama)"
    if KrakenEngine is not None and isinstance(engine, KrakenEngine) and getattr(engine, "_fallback", None) is not None:
        actual_backend = "kraken (fell back to ollama)"
    return actual_backend


# --- Overlay LineMatch (reuses measure_alignment.py) ------------------------


def _overlay_line_match(entry: dict[str, Any], manifest_dir: Path, dpi: int) -> float | None:
    """Compute overlay LineMatch for an entry, or None if not applicable.

    Reuses `measure_alignment.measure_page`. Requires the entry to declare
    "overlay_pdf" and the 0-based "overlay_page" index within that PDF.
    """
    if not entry.get("overlay_pdf") or entry.get("overlay_page") is None:
        return None
    try:
        import measure_alignment as ma

        overlay = _resolve(entry["overlay_pdf"], manifest_dir)
        scan = _resolve(entry["pdf"], manifest_dir)
        if not overlay.exists() or not scan.exists():
            return None
        idx = int(entry["overlay_page"])
        overlay_rgb = ma._render_page(overlay, idx, dpi)
        scan_rgb = ma._render_page(scan, idx, dpi)
        return float(ma.measure_page(overlay_rgb, scan_rgb)["line_match_ratio"])
    except Exception as exc:  # fail-soft: overlay metrics are optional
        logger.warning("overlay LineMatch failed for %s: %s", entry.get("id"), exc)
        return None


# --- Templates --------------------------------------------------------------


def _export_templates(entries: list[dict[str, Any]], manifest_dir: Path, cache_dir: Path) -> None:
    """Seed ground-truth files from cached OCR output for manual correction."""
    written = 0
    for entry in entries:
        gt_path = _resolve(entry["ground_truth"], manifest_dir) if entry.get("ground_truth") else None
        if gt_path is None:
            logger.warning("entry %s has no 'ground_truth' path; skipping", entry["id"])
            continue
        if gt_path.exists() and gt_path.read_text(encoding="utf-8").strip():
            logger.info("ground truth already present: %s", gt_path)
            continue
        cached = cache_dir / f"{entry['id']}.txt"
        text = cached.read_text(encoding="utf-8") if cached.exists() else ""
        if not text.strip():
            logger.warning(
                "no cached OCR for %s yet — run `python eval_ocr.py` first, then export templates",
                entry["id"],
            )
            continue
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text(text, encoding="utf-8")
        written += 1
        logger.info("template written (correct it by hand): %s", gt_path)
    logger.info("exported %d ground-truth template(s)", written)


# --- Main -------------------------------------------------------------------


def _gt_looks_synthetic(text: str) -> bool:
    """Heuristic for machine-generated ground truth (markdown/templated/oversized).

    The probe set once scored against LLM/markdown output and inflated every
    engine's error rate; warn loudly instead of silently scoring junk.
    """
    if "```" in text or text.lstrip().startswith("# ") or "\n# " in text:
        return True
    return len(text) > 20000


def _aggregate(per_entry: list[dict[str, Any]]) -> dict[str, float]:
    """Mean metrics over scored entries (those with ground truth)."""
    keys = ("cer_verbatim", "wer_verbatim", "cer_normalized", "wer_normalized")
    scored = [e for e in per_entry if e.get("scored")]
    if not scored:
        return {}
    agg = {k: round(sum(e["metrics"][k] for e in scored) / len(scored), 4) for k in keys}
    agg["entries"] = len(scored)
    return agg


def _print_table(per_entry: list[dict[str, Any]], agg: dict[str, float]) -> None:
    print(f"{'id':<14} {'CERv':>7} {'WERv':>7} {'CERn':>7} {'WERn':>7} {'LM':>6}  status")
    for e in per_entry:
        m = e.get("metrics")
        lm = e.get("line_match")
        lm_s = f"{lm:.3f}" if lm is not None else "  -  "
        if m:
            print(
                f"{e['id']:<14} {m['cer_verbatim']:>7.3f} {m['wer_verbatim']:>7.3f} "
                f"{m['cer_normalized']:>7.3f} {m['wer_normalized']:>7.3f} {lm_s:>6}  scored"
            )
        else:
            print(f"{e['id']:<14} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {lm_s:>6}  {e.get('status', 'skipped')}")
    print("\n--- Aggregate (scored entries) ---")
    if agg:
        for k, v in agg.items():
            print(f"  {k}: {v}")
    else:
        print("  (no entries with ground truth yet — run --export-templates, then correct them)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe-set OCR evaluation (CER/WER).")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Probe manifest JSON.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Where per-entry OCR text is cached.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Where to write eval_report.json.")
    parser.add_argument("--engine", choices=("ollama", "hf", "kraken", "none"), default="ollama", help="OCR engine to run before scoring.")
    parser.add_argument("--score-only", action="store_true", help="Skip OCR; score cached outputs only.")
    parser.add_argument("--force", action="store_true", help="Re-run OCR even if a cached output exists.")
    parser.add_argument("--export-templates", action="store_true", help="Write GT templates from cached OCR and exit.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report instead of a table.")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest_dir = manifest_path.parent
    entries = _load_manifest(manifest_path)
    cache_dir = args.cache_dir
    config = get_config()

    # 1. Export templates if requested (no model needed).
    if args.export_templates:
        _export_templates(entries, manifest_dir, cache_dir)
        return

    # 2. Optionally run OCR to populate the cache.
    engine_used = "cached"
    if not args.score_only and args.engine in ("ollama", "hf", "kraken"):
        engine_used = asyncio.run(_run_ocr(entries, manifest_dir, cache_dir, args.force, args.engine))

    # 3. Score cached outputs against ground truth.
    per_entry: list[dict[str, Any]] = []
    for entry in entries:
        record: dict[str, Any] = {"id": entry["id"], "page": entry.get("page"), "scored": False}

        gt_path = _resolve(entry["ground_truth"], manifest_dir) if entry.get("ground_truth") else None
        cache_path = cache_dir / f"{entry['id']}.txt"

        if gt_path is None or not gt_path.exists() or not gt_path.read_text(encoding="utf-8").strip():
            record["status"] = "no ground truth"
        elif not cache_path.exists():
            record["status"] = "no ocr output (run without --score-only)"
        else:
            ref = gt_path.read_text(encoding="utf-8")
            hyp = cache_path.read_text(encoding="utf-8")
            if _gt_looks_synthetic(ref):
                logger.warning(
                    "ground truth for %s looks machine-generated (markdown/heading/oversized, %d chars) — "
                    "metrics may be meaningless; see probe/README.md", entry["id"], len(ref))
                record["gt_warning"] = True
            record["metrics"] = score_pair(ref, hyp)
            record["scored"] = True

        lm = _overlay_line_match(entry, manifest_dir, config.pipeline.dpi)
        if lm is not None:
            record["line_match"] = round(lm, 4)

        per_entry.append(record)

    agg = _aggregate(per_entry)
    report = {
        "manifest": str(manifest_path),
        "engine": engine_used,
        "aggregate": agg,
        "per_entry": per_entry,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_table(per_entry, agg)
        print(f"\nReport: {args.report}")


if __name__ == "__main__":
    main()
