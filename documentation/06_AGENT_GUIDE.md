# 06 — Agent Guide: Harness & Guardrails

> **Read this before making ANY change.** This document is the harness and
> guardrail set for future agentic development or vibe-coding agents using other
> LLMs. It tells you how to work on this project safely without breaking
> current functionality.

## Your mission

This project must be **the most accurate Arabic Language OCR available,
especially for Arabic Language Religious books**. The operating constraint is
**accuracy > speed, thermal via queueing**.

Any change you make must:
1. **Not break** the current working pipeline.
2. **Improve** OCR accuracy or output quality.
3. **Not** cause GPU overheating.

## The golden rule: verify before and after

**Always test on pages 31-35 of `input/rabee/majmua/1.pdf`** (the main text)
before and after any change. This is the canonical test set.

```bash
# 1. Unset stale preprocess env vars (CRITICAL — see below)
unset PREPROCESS_MODE PREPROCESS_ENABLED PREPROCESS_DESKEW \
       PREPROCESS_BINARIZE PREPROCESS_DENOISE PREPROCESS_NORMALIZE_CONTRAST

# 2. Run the pipeline on the test pages
python main.py --input-dir ./input --output-dir ./output \
    --only 1.pdf --pages 31-35 --preprocess --overlay-line

# 3. Verify OCR text quality (read the .md output — should be sensible Arabic)
cat output/rabee/majmua/1/1.md

# 4. If you changed the overlay, measure alignment objectively
OVERLAY_DEBUG=true python main.py --input-dir ./input --output-dir ./output \
    --only 1.pdf --pages 31-35 --preprocess --overlay-line
python measure_alignment.py --overlay output/rabee/majmua/1/1_overlay.pdf \
    --scan input/rabee/majmua/1.pdf --pages 1-5 --offset 30
```

## CRITICAL gotcha: stale environment variables

`config.py` loads `.env` via `os.environ.setdefault`, which means **any
variable already set in the shell overrides `.env`**. If you (or a previous
agent) exported `PREPROCESS_MODE=off` in the shell, preprocessing will be
silently disabled even though `.env` says `full`.

**Always `unset` the preprocess vars before running** (as shown above). If you
see preprocessing not taking effect, check the shell environment first.

## The three hard-won fixes (DO NOT REVERT)

These fixes were hard-won through extensive debugging. Reverting any of them
breaks the overlay:

### 1. Preprocess the overlay image too
`format_converter.write_overlay_line()` must preprocess its image so the
background and ink-height calibration match the OCR text. If the OCR text comes
from the preprocessed image but the overlay geometry comes from the raw scan,
they diverge and the overlay misaligns.

### 2. Run Surya on the RAW image (NOT preprocessed)
Binarization strips the subtle gradients Surya relies on to find text lines.
On page 33: RAW gives 33 lines, preprocessed gives only 5. **Never run Surya
on the preprocessed image.** Since deskew preserves dimensions, raw line boxes
map 1:1 onto the preprocessed background.

### 3. `scan_base_h_px` fallback
When a line box is empty (`max_density == 0`), `scan_base_h_px` must be set to
a fallback (`rect.height * (dpi/72)`), or the code crashes with "cannot access
local variable 'scan_base_h_px'".

## Guardrails by area

### OCR engine (`ocr_engine.py`)
- **Never change the vision prompt** to allow translation, summarization, or
  commentary — the model must output ONLY transcribed text.
- **Never remove refusal phrases** (English + Arabic) — they prevent "I can't
  read this" from being stored as valid OCR.
- **Keep `OLLAMA_TEMPERATURE=0.0`** — nonzero temperature risks hallucinated
  Arabic.
- **Don't switch to a larger model** without verifying thermal safety (the 32b
  model caused crashes).

### Preprocessing (`preprocess.py`)
- **Never run Surya on the preprocessed image** — it breaks line detection.
- **Keep preprocessing consistent** between OCR and overlay (both must use the
  same preprocessed image).
- **Deskew must preserve dimensions** — if you change `_deskew` to crop or
  resize, the raw→preprocessed coordinate mapping breaks.
- **Use `ab_test.py`** to verify preprocessing helps before enabling it.

### Overlay (`format_converter.py`)
- **LineMatch is the metric that matters**, not IoU. IoU is cosmetic
  glyph-overlap; LineMatch is searchability.
- **`insert_htmlbox` is required** for searchability — don't switch to
  `insert_text` (it breaks searchability).
- **Don't "fix" the font-size ambiguity** by guessing — Arabic diacritics make
  it fundamentally ambiguous. The closed-loop base-letter calibration is the
  best known.
- **Use `measure_alignment.py`** to verify any overlay change objectively.

### Thermal safety
- **Keep `OCR_WORKERS=1`** — higher concurrency overheats the GPU.
- **Keep `OLLAMA_QUEUE_INFERENCE=true`** — serializes inference.
- **Keep the thermal guard enabled** — it prevents system crashes.

## How to make a change safely

1. **Read the relevant doc** (00-05) for the area you're changing.
2. **Test the current state** on pages 31-35 (baseline).
3. **Make a small, focused change.**
4. **Re-test** on pages 31-35.
5. **Compare** OCR text quality and (if overlay) LineMatch before/after.
6. **Only keep the change** if it improves accuracy without breaking anything.

## Common pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Stale `PREPROCESS_MODE` in shell | Preprocessing silently off | `unset` the preprocess vars |
| Surya on preprocessed image | Overlay missing most lines | Run Surya on raw image |
| Overlay geometry from raw, text from preprocessed | Overlay misaligned | Preprocess the overlay image too |
| `scan_base_h_px` unbound | "cannot access local variable" crash | Add the fallback |
| Nonzero temperature | Hallucinated Arabic | Keep `OLLAMA_TEMPERATURE=0.0` |
| `insert_text` instead of `insert_htmlbox` | Overlay not searchable | Use `insert_htmlbox` |
| Higher `OCR_WORKERS` | GPU overheating/crash | Keep `OCR_WORKERS=1` |

## Testing commands cheat-sheet

```bash
# Run pipeline on test pages with preprocessing + line overlay
python main.py --input-dir ./input --output-dir ./output \
    --only 1.pdf --pages 31-35 --preprocess --overlay-line

# Generate red debug overlay for alignment measurement
OVERLAY_DEBUG=true python main.py --input-dir ./input --output-dir ./output \
    --only 1.pdf --pages 31-35 --preprocess --overlay-line

# Measure alignment
python measure_alignment.py --overlay output/rabee/majmua/1/1_overlay.pdf \
    --scan input/rabee/majmua/1.pdf --pages 1-5 --offset 30

# A/B test preprocessing
python ab_test.py --pdf input/rabee/majmua/1.pdf --pages 3,4,5

# Dry-run (list PDFs)
python main.py --dry-run
```

## What's already been tried (so you don't repeat it)

- **Font-size matching**: 6 approaches tried (height-based, width-based,
  per-word, closed-loop, diacritic-excluding base-letter, base-letter
  closed-loop). NONE converge to a visually-correct size. The closed-loop
  base-letter calibration is the best so far.
- **Overlay detectors**: Surya (best line detection), EasyOCR (best
  recognition), OpenCV (fallback). Surya is the default.
- **Vision models**: qwen2.5vl:7b (primary), qwen3-vl:latest (fallback),
  qwen3-vl:32b (too hot), llama3.2-vision:11b (broken), minicpm-v:8b (poor).

## Good next steps (if you want to improve)

1. **Improve overlay font-size matching** — the closed-loop calibration is good
   but not perfect. A dedicated Arabic text-layer library might help.
2. **Improve pages 33-34 LineMatch** (currently 0.7) — these are the lower
   performers.
3. **Run the full 510-page book** with preprocessing enabled for a complete
   result.
4. **Add more refusal phrases** if you find the model refusing in new ways.
5. **Improve post-processing** — `OLLAMA_POSTPROCESS_ENABLED` is currently off;
   a good post-correction pass could improve accuracy further.
