# 03 — Preprocessing

> OpenCV image preprocessing applied **before** vision OCR to improve accuracy
> on pixelated scans and poor paper quality. Read [01_ARCHITECTURE.md] first.

## Why preprocessing matters

The user's core complaint was:

> "I still see that it's not accurate, maybe it's due to these scans being
> pixelated... we need preprocessing to change it to black and white, so the
> text is full black and background paper is full white."

Binarizing to **black text on white background** dramatically improves OCR
accuracy on degraded scans. The vision model sees a clean page instead of
noisy gray.

## Modes

`preprocess.py` supports three modes (via `PREPROCESS_MODE`):

| Mode | What it does | When to use |
|------|--------------|-------------|
| `off` | No preprocessing | Clean scans; vision LLM handles them fine |
| `light` | Global contrast normalization only (fast, low-risk) | Mildly washed-out pages |
| `full` | Deskew + binarize + denoise + contrast (heavy) | Pixelated scans, poor paper |

## The steps (in `full` mode)

1. **Deskew** (`_deskew`): corrects small rotation using the min-area rectangle
   of text pixels. Skips if the angle is below 0.3° to avoid over-rotating
   near-straight pages. Uses `warpAffine` with the **same output dimensions** —
   this is important (see below).

2. **Binarize** (`_adaptive_binarize`): adaptive thresholding to clean uneven
   illumination / bleed-through. Gaussian blur first, then
   `cv2.adaptiveThreshold`.

3. **Denoise** (`_denoise`): non-local means denoising (preserves text edges
   better than blur).

4. **Normalize contrast** (`_normalize_contrast`): CLAHE on the L channel
   (LAB colorspace).

## Verified effect

On page 33 of `input/rabee/majmua/1.pdf`:

| Metric | Raw scan | Preprocessed |
|--------|----------|--------------|
| Black text | 0.1% | **1.4%** |
| White paper | 96.6% | **98.6%** |
| Gray (noise) | 3.3% | **0.0%** |

The page becomes clean black-on-white.

## Where preprocessing is applied

Preprocessing is applied in **two places** (this is critical for consistency):

1. **`pdf_processor._render_page()`** — the image sent to the vision OCR model.
2. **`format_converter.write_overlay_line()`** — the image used as the overlay
   background AND for ink-height calibration.

**Why both?** If the OCR text comes from the preprocessed image but the overlay
geometry (line boxes, ink height) comes from the raw scan, they diverge and the
overlay misaligns. Both must use the same preprocessed image.

## CRITICAL: Surya detection must run on the RAW image

This is a hard-won lesson. **Surya line detection must run on the RAW image,
NOT the preprocessed one.**

Binarization strips the subtle gradients Surya relies on to find text lines.
On page 33:
- RAW image: **33 lines** detected
- Preprocessed image: **5 lines** detected (mostly giant merged boxes)

Since deskew preserves image dimensions, raw line boxes map 1:1 onto the
preprocessed background. So the code does:
- `raw_bytes` = rendered page (no preprocessing)
- `img_bytes` = `preprocess(raw_bytes)` (for background + ink height)
- Surya detection runs on `raw_bytes` (for line boxes)

**Guardrail:** do NOT "simplify" this by running Surya on the preprocessed
image — it will silently drop most lines and break the overlay.

## Config

| Variable | Default | Purpose |
|----------|---------|---------|
| `PREPROCESS_MODE` | `off` | `off` / `light` / `full` |
| `PREPROCESS_ENABLED` | `false` | Master switch |
| `PREPROCESS_DESKEW` | `true` | Skew correction |
| `PREPROCESS_BINARIZE` | `false` | Adaptive thresholding |
| `PREPROCESS_DENOISE` | `false` | Non-local means denoising |
| `PREPROCESS_NORMALIZE_CONTRAST` | `false` | CLAHE contrast |

The current `.env` enables `full` mode with all steps on.

## A/B testing

`ab_test.py` renders a few pages and OCRs each twice — once with preprocessing
disabled (baseline) and once enabled — writing both outputs side-by-side for
manual comparison. Use it to verify preprocessing helps before committing to it.

```bash
python ab_test.py --pdf input/rabee/majmua/1.pdf --pages 3,4,5
```

## Guardrails for agents

1. **Never run Surya on the preprocessed image** — it breaks line detection.
2. **Keep preprocessing consistent** between OCR and overlay (both must use the
   same preprocessed image).
3. **Test on pages 31-35** before/after any preprocessing change.
4. **Use `ab_test.py`** to verify preprocessing actually helps before enabling
   it globally.
5. **Deskew must preserve dimensions** — if you change `_deskew` to crop or
   resize, the raw→preprocessed coordinate mapping breaks.
