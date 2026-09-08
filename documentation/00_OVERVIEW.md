# 00 — Project Overview

> **Read this first.** This document is the entry point for any agent (human or
> AI) working on this codebase. It explains what the project is, why it exists,
> and how the pieces fit together. For deep dives, see the numbered docs that
> follow.

## What this project is

An **automated, fault-tolerant, asynchronous Arabic PDF OCR pipeline**. It takes
scanned Arabic religious books (image-only PDFs with **no text layer**) and
produces clean, structured, multi-format output:

- `.md` — canonical Markdown with page anchors
- `.html` — standalone HTML5 (RTL, Amiri font)
- `.epub` — EPUB3 with navigation
- `.mobi` — MOBI (if Calibre is installed)
- `.pdf` — searchable PDF (WeasyPrint)
- `{stem}_overlay.pdf` — **searchable overlay PDF**: the original scan as the
  visible image, with an invisible OCR text layer on top so the PDF is
  searchable/selectable while looking identical to the scan.

## Why it exists

The user's mission, stated explicitly:

> "Our system must be the most accurate Arabic Language OCR available,
> especially for Arabic Language Religious books."

And the operating constraint:

> "Accuracy > speed, thermal via queueing."

The books are complex: classical Arabic with **diacritical marks (Harakat)**,
footnotes, running headers, page numbers, and sometimes **pixelated scans or
poor paper quality**. The pipeline must handle all of this while keeping the
GPU from overheating (the host is a Dell Pro Max GB10 with a single GPU).

## Core design decisions (the "why")

| Decision | Why |
|----------|-----|
| **Vision LLM (Ollama) for OCR**, not Tesseract/EasyOCR | Vision LLMs (qwen2.5vl) produce far more accurate Arabic with diacritics than traditional OCR engines. |
| **`qwen2.5vl:7b` as primary model** | Excellent Arabic OCR, ~5-15s/page, low GPU heat. Larger models (qwen3-vl:32b) caused thermal crashes. |
| **OpenCV preprocessing (binarization)** | Pixelated scans and poor paper → binarize to black-on-white improves OCR accuracy dramatically. |
| **Surya for line detection** | Best Arabic line detection (mAP 79.67%). We use only its *detection* boxes, paired with the LLM's accurate *text*. |
| **Thermal guard + inference queue** | Prevents GPU overheating/crashes. All inference serialized through one queue; GPU temp monitored and throttled. |
| **Per-page cache + state tracking** | Idempotent resume: re-running skips already-completed pages. |
| **Searchable overlay via `insert_htmlbox`** | Invisible (opacity=0) but searchable text layer over the scan. |

## The pipeline at a glance

```
[ input/ Nested PDFs ]
        │  PyMuPDF render @300 DPI
        ▼
   Page images ──► [OpenCV preprocess: deskew/binarize/denoise/contrast]
        │
        ▼
   Ollama Vision OCR (qwen2.5vl:7b) ──► Arabic text + Markdown
        │
        ▼
   Canonical Markdown (page anchors) ──► .md .html .epub .mobi .pdf
        │
        └──► Searchable overlay PDF (scan + invisible text layer)
```

## Key files (module map)

| File | Responsibility |
|------|----------------|
| `main.py` | CLI entry point |
| `config.py` | Env-driven configuration (all tunables) |
| `pdf_processor.py` | Async traversal, page rendering, Markdown assembly |
| `ocr_engine.py` | Ollama vision OCR + post-processing + refusal detection |
| `preprocess.py` | OpenCV preprocessing (deskew, binarize, denoise, contrast) |
| `format_converter.py` | Multi-format output + **searchable overlay** |
| `surya_detection.py` | Surya line detection (best Arabic line boxes) |
| `easyocr_detection.py` | EasyOCR line detection (alternative) |
| `measure_alignment.py` | **Independent** overlay alignment measurement |
| `thermal.py` | GPU thermal guard |
| `text_cleanup.py` | Strip page numbers / running headers |
| `state_tracker.py` | Redis + JSON idempotent state |
| `toc_builder.py` | ToC extraction |
| `database_sync.py` | Optional DB/vector/graph ingestion |
| `ab_test.py` | A/B test: OCR with vs. without preprocessing |
| `models.py` | Dataclasses (`PageResult`, `DocumentResult`, `TocEntry`) |

## Documentation index

| Doc | Contents |
|-----|----------|
| [00_OVERVIEW.md](00_OVERVIEW.md) | This file — what/why/how at a glance |
| [01_ARCHITECTURE.md](01_ARCHITECTURE.md) | Data flow, module responsibilities, orchestration |
| [02_OCR_ENGINE.md](02_OCR_ENGINE.md) | Vision model, prompts, refusal detection, post-processing |
| [03_PREPROCESSING.md](03_PREPROCESSING.md) | OpenCV preprocessing: modes, steps, when to use |
| [04_OVERLAY_ALIGNMENT.md](04_OVERLAY_ALIGNMENT.md) | Searchable overlay: modes, Surya detection, font calibration, alignment measurement |
| [05_CONFIGURATION.md](05_CONFIGURATION.md) | Every env var, its default, and what it does |
| [06_AGENT_GUIDE.md](06_AGENT_GUIDE.md) | **Harness + guardrails for future agents** — how to work on this safely |

## Current status (as of this writing)

- ✅ Core pipeline: render → preprocess → OCR → Markdown → multi-format output
- ✅ Thermal safety: thermal guard + inference queue
- ✅ Preprocessing: binarization wired into both OCR and overlay
- ✅ Searchable overlay: Surya line detection + LLM text + closed-loop font calibration
- ✅ Alignment measurement: `measure_alignment.py` (objective, independent)
- ✅ Text cleanup: page numbers / running headers stripped
- ✅ A/B testing: `ab_test.py`
- ⚠️ Font-size matching in overlay: 6 approaches tried, none converge to a
  visually-perfect size (Arabic diacritics make "correct" size ambiguous). The
  current closed-loop base-letter calibration is the best so far.
- ⚠️ Overlay IoU is low (0.03-0.07) — this is cosmetic glyph-overlap, not the
  searchability metric. LineMatch (the metric that matters) is 0.7-0.93.

## How to run

```bash
# Activate venv
source .venv/bin/activate

# IMPORTANT: unset stale preprocess env vars (see 06_AGENT_GUIDE.md)
unset PREPROCESS_MODE PREPROCESS_ENABLED PREPROCESS_DESKEW \
       PREPROCESS_BINARIZE PREPROCESS_DENOISE PREPROCESS_NORMALIZE_CONTRAST

# Test on a few pages with preprocessing + line overlay
python main.py --input-dir ./input --output-dir ./output \
    --only 1.pdf --pages 31-35 --preprocess --overlay-line

# Measure overlay alignment (needs OVERLAY_DEBUG=true to render red text)
OVERLAY_DEBUG=true python main.py --input-dir ./input --output-dir ./output \
    --only 1.pdf --pages 31-35 --preprocess --overlay-line
python measure_alignment.py --overlay output/rabee/majmua/1/1_overlay.pdf \
    --scan input/rabee/majmua/1.pdf --pages 1-5 --offset 30
```

See [06_AGENT_GUIDE.md](06_AGENT_GUIDE.md) for the full harness and guardrails.
