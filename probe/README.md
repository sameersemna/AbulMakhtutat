# Probe set — Arabic OCR evaluation

Small, locally-annotated evaluation set used to gate model swaps and
preprocessing changes. See `documentation/08_IMPLEMENTATION_PLAN.md` (Task 1).

## What lives here

| Path | Purpose |
|------|---------|
| `manifest.json` | List of probe entries (PDF, page, ground-truth path). |
| `ground_truth/` | Hand-corrected reference text for each entry (`<id>.txt`). |
| `_ocr_cache/` | Raw OCR output per entry, cached so scoring never re-runs the model. |
| `eval_report.json` | Last evaluation report (per-entry + aggregate metrics). |

## Workflow

```bash
# 1. Run the current engine over the probe pages (populates probe/_ocr_cache/).
.venv/bin/python eval_ocr.py

# 2. Seed ground-truth templates from the cached OCR.
.venv/bin/python eval_ocr.py --export-templates

# 3. Open each probe/ground_truth/*.txt and CORRECT it by hand against the scan.
#    These become the reference. Accuracy of the reference is what everything
#    else is measured against, so take care here.

# 4. Score (no model run).
.venv/bin/python eval_ocr.py --score-only
```

Adding pages: append an entry to `manifest.json` with a unique `id`, the `pdf`,
the 1-based `page`, and a `ground_truth` path. Prefer diacritized and
footnote-heavy pages in addition to the canonical pages 31–35.

## Metrics

- **CER / WER** (lower is better, 0.0 = perfect) in two modes:
  - `*_verbatim` — diacritics and punctuation count.
  - `*_normalized` — Arabic-normalized via `text_cleanup.normalize_arabic`.
    The fair cross-engine comparison (some engines drop diacritics).
- **LineMatch** (higher is better) — overlay searchability, only computed for
  entries that declare `overlay_pdf` + `overlay_page`; reused from
  `measure_alignment.py`.

## Notes

- Add `probe/_ocr_cache/` and `probe/eval_report.json` to `.gitignore` if they
  should not be committed; the hand-corrected `ground_truth/` files **should** be
  committed so results stay comparable over time.
- The harness never needs a running model for `--score-only` / `--export-templates`.
