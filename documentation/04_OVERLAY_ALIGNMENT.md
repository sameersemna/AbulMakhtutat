# 04 — Overlay Alignment

> The searchable overlay PDF is the most complex and hard-won part of this
> project. This doc explains the modes, the Surya detection, the font
> calibration, and how to measure alignment objectively.

## What the overlay is

`{stem}_overlay.pdf` is a **searchable PDF**: the original scan as the visible
image, with an **invisible OCR text layer** on top (opacity=0). The PDF looks
identical to the scan but is searchable and selectable.

## Overlay modes

`OVERLAY_MODE` selects the strategy:

| Mode | Description | Status |
|------|-------------|--------|
| `off` | No overlay | Default |
| `page` | Full-page text layer (single invisible block) | Basic |
| `word` | Word-aligned (OpenCV line) | Legacy |
| `bbox` | Model word bounding boxes | Legacy |
| `line` | **Line-level alignment** (BEST) | ✅ Current |

**`line` mode** is the recommended and current mode. It:
- Uses the **complete main OCR text** (from the cached result), not the LLM's
  line-box text (which under-segments Arabic).
- Splits the text across **Surya-detected line boxes** (pixel-accurate
  positions).
- Sizes each line's font to fill the scan's text line height.

## Why Surya for line detection

Surya (datalab-to/surya) has the **best Arabic text-line detection**
(mAP@0.50 79.67% on KITAB-Bench), but its *recognition* is poor (CER 4.95).
We only use its **detection boxes** and pair them with the LLM's accurate
Arabic text.

EasyOCR has the best Arabic *recognition* (CER 0.20) and is kept as a
selectable alternative (`OVERLAY_DETECTOR=easyocr`) for A/B comparison.

## The overlay pipeline (in `write_overlay_line`)

For each page:

1. **Render** the page at the configured DPI → `raw_bytes`.
2. **Preprocess** `raw_bytes` → `img_bytes` (background + ink-height
   calibration). **Must match the OCR text's preprocessing.**
3. **Detect line boxes** on the **RAW** image (Surya/EasyOCR/OpenCV). Raw line
   boxes map 1:1 onto the preprocessed background (deskew preserves dims).
4. **Filter** line boxes: drop boxes taller than half the page (decorative
   title/cover artifacts).
5. **Create the output page**: insert the preprocessed image as background.
6. **Split the complete OCR text** into words, fitting them into each line box
   by width.
7. **Closed-loop font calibration** (see below): size each line's font so the
   rendered base-letter height matches the scan's base-letter height.
8. **Insert invisible text** via `insert_htmlbox` (opacity=0, still searchable).

## Closed-loop font calibration

Arabic diacritics (Harakat) make font sizing ambiguous. The approach:

1. Measure the scan's **base-letter height** (dense rows, excluding diacritics)
   within each line box. Diacritics are sparse; base letters are dense.
2. Render the line text at a test size in a **separate temp document**.
3. Measure the rendered base-letter height.
4. Scale the font size so rendered base height matches scan base height.

This is the best of 6 approaches tried. None converge to a visually-perfect
size (Arabic diacritics make "correct" size ambiguous), but this is the most
accurate.

## Alignment measurement (`measure_alignment.py`)

This is an **independent verification tool**. It renders the overlay PDF and
the original scan at the SAME DPI, then compares red overlay-text pixels
(requires `OVERLAY_DEBUG=true` to render red text) against dark scan-text
pixels.

### Metrics

| Metric | Meaning | Good threshold |
|--------|---------|----------------|
| **IoU** | \|red ∩ dark\| / \|red ∪ dark\| | >0.7 good, 0.5-0.7 fair, <0.5 poor |
| **Precision** | fraction of overlay text on scan text | higher = better |
| **Recall** | fraction of scan text covered | higher = better |
| **F1** | harmonic mean of precision & recall | higher = better |
| **LineMatch** | fraction of overlay lines whose center falls within a scan line band | **>0.8 good** — this is what matters for searchable PDFs |
| **OffPx** | mean \|overlay line center - nearest scan line center\| | lower = better |

**LineMatch is the metric that matters** for a searchable PDF: does each
overlay text line sit at the same vertical position as a scan text line? It
does NOT require glyph-shape overlap (impossible without the original font).

### Usage

```bash
# Generate overlay with red debug text
OVERLAY_DEBUG=true python main.py --input-dir ./input --output-dir ./output \
    --only 1.pdf --pages 31-35 --preprocess --overlay-line

# Measure alignment (--offset maps overlay subset pages to scan pages)
python measure_alignment.py --overlay output/rabee/majmua/1/1_overlay.pdf \
    --scan input/rabee/majmua/1.pdf --pages 1-5 --offset 30
```

`--offset 30` means overlay page 1 maps to scan page 31 (0-based offset).

## Current measured results (pages 31-35, binarized overlay)

| Page | LineMatch | OffPx |
|------|-----------|-------|
| 31 | **0.929** | 6.0 |
| 32 | **0.882** | 5.7 |
| 33 | 0.706 | 22.4 |
| 34 | 0.735 | 12.6 |
| 35 | **0.879** | 8.1 |

Pages 31, 32, 35 have **good** LineMatch (>0.8). Pages 33, 34 are lower but
acceptable. IoU is low (0.03-0.07) — this is cosmetic glyph-overlap, not the
searchability metric.

## Known issues & history

### Font-size matching (unresolved)
6 approaches tried (height-based, width-based, per-word, closed-loop,
diacritic-excluding base-letter, base-letter closed-loop). NONE converge to a
visually-correct size. Root cause: Arabic diacritics make "correct" size
ambiguous. The current closed-loop base-letter calibration is the best so far.

### Selection gaps
`insert_htmlbox` splits words into spans, which can create selection gaps.
`insert_text` gives contiguous text but breaks searchability. We kept
`insert_htmlbox`.

### Page 33 missing words (fixed)
The LLM's line-box text under-segmented Arabic (6 boxes vs 33 visual lines),
dropping most lines. Fixed by using the **complete main OCR text** split across
Surya line boxes.

### Alignment regression after preprocessing (fixed)
The overlay geometry used the raw scan while OCR text came from the preprocessed
image → divergence. Fixed by preprocessing the overlay image too, running Surya
on the raw image, and adding a `scan_base_h_px` fallback.

## Guardrails for agents

1. **Never run Surya on the preprocessed image** — it drops most lines.
2. **Keep preprocessing consistent** between OCR and overlay.
3. **Use `measure_alignment.py`** to verify any overlay change objectively.
4. **LineMatch is the metric that matters**, not IoU.
5. **Test on pages 31-35** before/after any overlay change.
6. **Don't "fix" the font-size ambiguity** by guessing — Arabic diacritics make
   it fundamentally ambiguous. The closed-loop calibration is the best known.
7. **`insert_htmlbox` is required** for searchability — don't switch to
   `insert_text` (it breaks searchability).
