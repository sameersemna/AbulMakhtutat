# Arabic PDF OCR & Multi-Format Conversion Pipeline

An automated, fault-tolerant, asynchronous document-processing pipeline that OCRs
complex Arabic PDF books (image scans with no text layer) and produces clean,
structured multi-format output (`.md`, `.html`, `.pdf`, `.epub`, `.mobi`) plus a
**searchable overlay PDF** (original scan + invisible OCR text layer).

> **For agents working on this codebase:** read
> [`documentation/00_OVERVIEW.md`](documentation/00_OVERVIEW.md) first, then
> [`documentation/06_AGENT_GUIDE.md`](documentation/06_AGENT_GUIDE.md) for the
> harness and guardrails.

## Architecture

```
[ input/ Nested PDFs ]  →  Python Orchestration (asyncio)  →  Ollama Vision OCR (local)
        │                                                          │
        │  PyMuPDF render @300 DPI                                 │  qwen2.5vl:7b
        │  OpenCV preprocess (deskew/binarize/denoise/contrast)   │
        ▼                                                          ▼
   Page images ────────────────────────────────────────────►  Arabic text + Markdown
                                                                   │
        ┌──────────────────────────────────────────────────────────┤
        ▼                                                          ▼
  Canonical Markdown (page anchors)                    Optional DB/vector sync
        │                                              (Postgres, Redis, ES,
        ▼                                               Qdrant, Neo4j, MySQL)
  Pandoc / WeasyPrint  →  .md .html .epub .mobi .pdf
        │
        └──► Searchable overlay PDF (scan + invisible text layer)
```

## Verified Model Findings (important)

| Model | Status |
|-------|--------|
| `qwen2.5vl:7b` | ✅ **Primary** — excellent Arabic OCR, ~5-15s/page, low GPU heat |
| `qwen3-vl:latest` | ✅ Fallback — 6.1GB, good quality |
| `qwen3-vl:32b-instruct-q8_0` | ⚠️ 35GB — best quality but caused thermal crashes (~86s/page @ 95% GPU) |
| `llama3.2-vision:11b` | ❌ Broken on this Ollama build (`mllama` arch → HTTP 500) |
| `minicpm-v:8b` | ⚠️ Poor Arabic (hallucinates watermarks) |

## Requirements

- Python 3.11+
- System: `pandoc`, `poppler` (optional), `ebook-convert` (Calibre, for MOBI)
- Ollama running locally with `qwen2.5vl:7b` pulled
- OpenCV (`cv2`), Surya (`surya`), EasyOCR (`easyocr`) for preprocessing and
  overlay line detection

```bash
pip install -r requirements.txt
ollama pull qwen2.5vl:7b
```

## Usage

```bash
# IMPORTANT: unset stale preprocess env vars (see documentation/06_AGENT_GUIDE.md)
unset PREPROCESS_MODE PREPROCESS_ENABLED PREPROCESS_DESKEW \
       PREPROCESS_BINARIZE PREPROCESS_DENOISE PREPROCESS_NORMALIZE_CONTRAST

# Full run (all PDFs, all formats, with preprocessing + line overlay)
python main.py --input-dir ./input --output-dir ./output --preprocess --overlay-line

# Test on a few pages first
python main.py --only 1.pdf --pages 31-35 --preprocess --overlay-line

# Skip a format
python main.py --skip-formats mobi

# List PDFs without processing
python main.py --dry-run
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | `./input` | Root input directory |
| `--output-dir` | `./output` | Root output directory |
| `--workers` | `1` | Concurrent OCR workers (keep low to avoid GPU overheating) |
| `--dpi` | `300` | Render DPI |
| `--max-pages` | `0` (all) | Limit pages per PDF (testing) |
| `--pages` | `""` | Page range, e.g. `31-35` or `1,3,5-7` |
| `--only` | `""` | Process only this PDF filename |
| `--skip-formats` | `""` | Comma-separated formats to skip |
| `--no-postprocess` | off | Disable text post-processing |
| `--correct` | off | Enable post-OCR correction pass |
| `--preprocess` | off | Enable full OpenCV preprocessing (deskew, binarize, denoise, contrast) |
| `--preprocess-light` | off | Enable light preprocessing (global contrast only) |
| `--overlay` | off | Generate searchable overlay PDF (page mode) |
| `--overlay-bbox` | off | Overlay with model-precise bounding-box alignment |
| `--overlay-line` | off | Overlay with line-level alignment (BEST for Arabic) |
| `--dry-run` | off | List PDFs only |

## Output Structure

Input hierarchy is mirrored into `output/`:

```
input/rabee/majmua/1.pdf
  → output/rabee/majmua/1/
      1.md            # canonical Markdown with page anchors
      1.html          # standalone HTML5 (RTL, Amiri font)
      1.epub          # EPUB3 with nav
      1.mobi          # MOBI (if Calibre installed)
      1.pdf           # searchable PDF (WeasyPrint)
      1_overlay.pdf   # searchable overlay (scan + invisible text layer)
      toc.json        # structured ToC
      TOC.md          # human-readable ToC
      style.css       # RTL stylesheet
```

## Searchable Overlay

The `{stem}_overlay.pdf` is a **searchable PDF**: the original scan as the
visible image, with an **invisible OCR text layer** on top (opacity=0). It
looks identical to the scan but is searchable and selectable.

- **`--overlay-line`** (recommended): line-level alignment using Surya line
  detection (best Arabic line detection, mAP 79.67%) + the complete LLM OCR
  text + closed-loop font calibration.
- **`--overlay-bbox`**: model-precise bounding-box alignment.
- **`--overlay`**: full-page text layer.

To verify alignment objectively, use `measure_alignment.py` (requires
`OVERLAY_DEBUG=true` to render red debug text):

```bash
OVERLAY_DEBUG=true python main.py --input-dir ./input --output-dir ./output \
    --only 1.pdf --pages 31-35 --preprocess --overlay-line
python measure_alignment.py --overlay output/rabee/majmua/1/1_overlay.pdf \
    --scan input/rabee/majmua/1.pdf --pages 1-5 --offset 30
```

**LineMatch** (>0.8 good) is the metric that matters for searchable PDFs — it
measures whether each overlay text line sits at the same vertical position as a
scan text line. Current results on pages 31-35: 0.93, 0.88, 0.71, 0.74, 0.88.

## Preprocessing

OpenCV preprocessing binarizes pixelated scans to **black text on white
background**, dramatically improving OCR accuracy on poor paper. Enabled via
`--preprocess` (full) or `--preprocess-light`.

**Critical:** Surya line detection must run on the **raw** image, not the
preprocessed one (binarization strips the gradients Surya needs). See
[`documentation/03_PREPROCESSING.md`](documentation/03_PREPROCESSING.md).

## Configuration

All settings are environment-driven (see `config.py` and
[`documentation/05_CONFIGURATION.md`](documentation/05_CONFIGURATION.md)). Key
variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Primary vision model |
| `OLLAMA_VISION_FALLBACK` | `qwen3-vl:latest` | Fallback vision model |
| `OLLAMA_POSTPROCESS_ENABLED` | `false` | Enable text post-processing |
| `OLLAMA_MAX_IMAGE_DIM` | `1500` | Downscale long edge before OCR |
| `OLLAMA_KEEP_ALIVE` | `5m` | How long models stay resident in GPU memory |
| `OLLAMA_QUEUE_INFERENCE` | `true` | Serialize all inference calls through one queue |
| `OCR_DPI` | `300` | Render DPI |
| `OCR_WORKERS` | `1` | Concurrency (keep low to avoid GPU overheating) |
| `OCR_PAGES` | `""` | Page range, e.g. `31-35` |
| `OVERLAY_MODE` | `off` | `off` / `page` / `word` / `bbox` / `line` |
| `OVERLAY_DETECTOR` | `surya` | `surya` / `easyocr` / `opencv` |
| `OVERLAY_DEVICE` | `cuda` | `cuda` / `cpu` for Surya/EasyOCR |
| `OVERLAY_DEBUG` | `false` | Render overlay text in red (visible) for alignment verification |
| `PREPROCESS_MODE` | `off` | `off` / `light` / `full` |
| `PREPROCESS_ENABLED` | `false` | Enable OpenCV preprocessing |
| `PREPROCESS_BINARIZE` | `false` | Adaptive thresholding |
| `THERMAL_GUARD_ENABLED` | `true` | Pause OCR when GPU is too hot |
| `THERMAL_MAX_TEMP` | `70.0` | Max GPU temp (°C) before pausing |
| `REDIS_ENABLED` | `false` | Remote state tracking |
| `POSTGRES_ENABLED` | `false` | Execution logs / metadata |
| `ELASTICSEARCH_ENABLED` | `false` | Full-text Arabic search |
| `QDRANT_ENABLED` | `false` | Vector RAG |
| `NEO4J_ENABLED` | `false` | Knowledge graph |
| `MYSQL_ENABLED` | `false` | Secondary relational store |

All remote backends are **fail-soft**: if a service is unreachable, the pipeline
still produces local output and logs a warning.

## Inference Queueing (thermal safety)

To prevent GPU overheating, the pipeline **queues inference calls** so only one
runs at a time, while keeping models resident in GPU memory (no reload latency):

- A global asyncio lock serializes all Ollama calls (vision, fallback,
  post-process, embed) so only one inference runs at a time.
- Models stay loaded (`keep_alive=5m` default) so we don't waste time
  reloading them between pages.

This is controlled by `OLLAMA_QUEUE_INFERENCE` (default `true`) and
`OLLAMA_KEEP_ALIVE` (default `5m`).

## Resume / Idempotency

- Per-page OCR results are cached in `.ocr_cache/<file_hash>/page_NNNN.txt`.
- State is tracked in `.ocr_state.json` (and Redis when enabled).
- Re-running skips already-completed pages and files, resuming where it left off.

## Module Map

| File | Responsibility |
|------|----------------|
| `config.py` | Env-driven configuration |
| `models.py` | Dataclasses (`PageResult`, `DocumentResult`, `TocEntry`) |
| `state_tracker.py` | Redis + JSON idempotent state |
| `ocr_engine.py` | Ollama vision OCR + post-processing + refusal detection |
| `preprocess.py` | OpenCV image preprocessing (deskew, binarize, denoise, contrast) |
| `thermal.py` | GPU thermal guard (pauses OCR when GPU overheats) |
| `pdf_processor.py` | Async traversal, rendering, Markdown assembly |
| `toc_builder.py` | ToC extraction → `toc.json` / `TOC.md` |
| `format_converter.py` | Pandoc/WeasyPrint multi-format output + searchable overlay |
| `surya_detection.py` | Surya line detection (best Arabic line boxes) |
| `easyocr_detection.py` | EasyOCR line detection (alternative) |
| `measure_alignment.py` | Independent overlay alignment measurement |
| `text_cleanup.py` | Strip page numbers / running headers |
| `database_sync.py` | Optional DB/vector/graph ingestion |
| `ab_test.py` | A/B test: OCR with vs. without preprocessing |
| `main.py` | CLI entry point |

## Documentation

See the [`documentation/`](documentation/) folder:

- [00_OVERVIEW.md](documentation/00_OVERVIEW.md) — what/why/how at a glance
- [01_ARCHITECTURE.md](documentation/01_ARCHITECTURE.md) — data flow, modules, orchestration
- [02_OCR_ENGINE.md](documentation/02_OCR_ENGINE.md) — vision model, prompts, refusal detection
- [03_PREPROCESSING.md](documentation/03_PREPROCESSING.md) — OpenCV preprocessing
- [04_OVERLAY_ALIGNMENT.md](documentation/04_OVERLAY_ALIGNMENT.md) — searchable overlay
- [05_CONFIGURATION.md](documentation/05_CONFIGURATION.md) — every env var
- [06_AGENT_GUIDE.md](documentation/06_AGENT_GUIDE.md) — **harness + guardrails for future agents**
