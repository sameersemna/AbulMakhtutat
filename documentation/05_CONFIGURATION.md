# 05 — Configuration Reference

> Every environment variable, its default, and what it does. All settings are
> read from environment variables (via `config.py`), loaded from `.env` at
> import time.

## How config loading works

`config.py` calls `_load_dotenv()` at import time, which reads `.env` and sets
variables via `os.environ.setdefault`.

**CRITICAL GOTCHA:** because it uses `setdefault`, any variable already set in
the shell environment **overrides** `.env`. This caused a real bug where stale
`PREPROCESS_MODE=off` in the shell silently disabled preprocessing despite
`.env` saying `full`. **Always `unset` the preprocess vars before running** (see
[06_AGENT_GUIDE.md]).

## Ollama

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Primary vision OCR model |
| `OLLAMA_VISION_FALLBACK` | `qwen3-vl:latest` | Fallback vision model |
| `OLLAMA_POSTPROCESS_MODEL` | `qwen2.5-coder:14b` | Text post-correction model |
| `OLLAMA_POSTPROCESS_ENABLED` | `false` | Enable post-OCR correction |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | Embedding model (for vector backends) |
| `OLLAMA_TIMEOUT` | `300` | Per-request timeout (s) |
| `OLLAMA_MAX_RETRIES` | `3` | Retry count |
| `OLLAMA_BACKOFF_BASE` | `2.0` | Exponential backoff base |
| `OLLAMA_TEMPERATURE` | `0.0` | **Keep 0.0** — deterministic OCR |
| `OLLAMA_NUM_CTX` | `8192` | Context window |
| `OLLAMA_KEEP_ALIVE` | `5m` | How long models stay in GPU memory |
| `OLLAMA_QUEUE_INFERENCE` | `true` | Serialize all inference through one queue |
| `OLLAMA_MAX_IMAGE_DIM` | `1500` | Downscale long edge before OCR |
| `OLLAMA_MAX_IMAGE_DIM_HEAVY` | `2500` | Larger cap for heuristic-flagged "heavy" (diacritized) pages |
| `OLLAMA_RETRY_MODE` | `contrast` | Refusal retry transform: `contrast` / `sharpen` / `both` |

## Vision backend

| Variable | Default | Purpose |
|----------|---------|---------|
| `VISION_BACKEND` | `kraken` | `kraken` (default; in-repo recognition model — best accuracy and fastest), `ollama` (local VLM server) or `hf` (HuggingFace VLM, in-process) |
| `VISION_HF_MODEL` | `NAMAA-Space/Qari-OCR-v0.2-VL-2B-Instruct` | HF model id (QARI, DIMI, Fanar, …) |
| `VISION_HF_DEVICE` | `cuda` | Device for the HF model |
| `VISION_HF_MAX_NEW_TOKENS` | `2048` | Generation budget |
| `VISION_HF_LOAD_IN_4BIT` | `false` | 4-bit quantization for the HF model |
| `VISION_KRAKEN_MODEL` | `models/run_ppocrv6_m/best_0.9948.safetensors` | Kraken weights trained with `ketos train` (PP-OCRv6-medium, 99.40% char / 96.83% word on the holdout; `models/run_x_3e4/best_0.9690.safetensors` = earlier VGSL model) |
| `VISION_KRAKEN_DEVICE` | `auto` | Device for the Kraken model |
| `VISION_KRAKEN_SEG_MODEL` | *(empty)* | Custom BLLA segmentation model; empty = kraken's bundled `blla.mlmodel` |
| `VISION_KRAKEN_TEXT_DIRECTION` | `horizontal-rl` | Principal page direction for line reading order (Arabic RTL) |
| `VISION_KRAKEN_BATCH_SIZE` | `16` | Lines per forward pass |

The HF and Kraken backends are **fail-soft**: if their packages or weights are
unavailable, the engine factory falls back to Ollama. Neither supports
word/line boxes or embeddings, so keep `OVERLAY_DETECTOR=surya`.

The Kraken backend OCRs full pages by running kraken's BLLA line segmenter and
then the recognition model. The segmenter output is converted to bbox lines
because the in-repo models are trained on bbox line crops; feeding raw baseline
polygons drops accuracy drastically (2026-10-14: 1/7 vs 7/7 target words on a
book-9 page). For pre-cropped line images use `ocr_line` (via
`prefill_line_gt.py` / `prioritize_lines.py`), which is the accurate path —
96.8% character accuracy on the hand-corrected holdout.

## Pipeline

| Variable | Default | Purpose |
|----------|---------|---------|
| `INPUT_DIR` | `./input` | Root input directory |
| `OUTPUT_DIR` | `./output` | Root output directory |
| `OCR_DPI` | `300` | Render DPI |
| `OCR_DPI_HEAVY` | `400` | Re-render DPI for heuristic-flagged heavy pages |
| `OCR_HEAVY_HEURISTIC` | `false` | Enable per-page diacritic/small-mark detection |
| `OCR_HEAVY_THRESHOLD` | `0.35` | Small-mark fraction above which a page is "heavy" |
| `OCR_NORMALIZE_SEARCH` | `false` | Normalize the search/RAG copy (opt-in; never the canonical `.md`; requires a reindex). Normalization expands the 11 honourific ligatures (ﷺ → صلى الله عليه وسلم) so ligature glyphs and spelled-out text match in search |
| `OCR_WORKERS` | `1` | Concurrency (keep low to avoid GPU overload) |
| `OCR_MAX_PAGES` | `0` (all) | Limit pages per PDF (testing) |
| `OCR_PAGES` | `""` | Page range, e.g. `31-35` or `1,3,5-7` |
| `OCR_CACHE_DIR` | `./.ocr_cache` | Per-page cache directory |
| `OCR_USE_CACHE` | `true` | Enable idempotent resume |
| `OCR_STATE_FILE` | `./.ocr_state.json` | State tracking file |
| `OCR_RTL` | `true` | Right-to-left typography |
| `OCR_FONT_FAMILY` | `'Amiri', ...` | Output font family |
| `OCR_RUNNING_HEADER` | `""` | Running header to strip from output |
| `EBOOK_CONVERT` | `""` | Calibre binary (for MOBI) |

## Overlay

| Variable | Default | Purpose |
|----------|---------|---------|
| `OVERLAY_MODE` | `off` | `off` / `page` / `word` / `bbox` / `line` |
| `OVERLAY_FONT` | `/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf` | Arabic font for text layer |
| `OVERLAY_DEBUG` | `false` | Render overlay text in RED (visible) for alignment verification |
| `OVERLAY_USE_SURYA` | `true` | Use Surya for line boxes |
| `OVERLAY_DETECTOR` | `surya` | `surya` / `easyocr` / `opencv` |
| `OVERLAY_DEVICE` | `cuda` | `cuda` / `cpu` for Surya/EasyOCR |

## Preprocessing

| Variable | Default | Purpose |
|----------|---------|---------|
| `PREPROCESS_MODE` | `off` | `off` / `light` / `sharpen` / `full` |
| `PREPROCESS_ENABLED` | `false` | Master switch |
| `PREPROCESS_DESKEW` | `true` | Skew correction |
| `PREPROCESS_BINARIZE` | `false` | Adaptive thresholding |
| `PREPROCESS_DENOISE` | `false` | Non-local means denoising |
| `PREPROCESS_NORMALIZE_CONTRAST` | `false` | CLAHE contrast |
| `PREPROCESS_SHARPEN` | `false` | Mild unsharp mask (VLM-friendly) |
| `PREPROCESS_VLM_SAFE` | `false` | Skip binarize/denoise/contrast on the VLM path |

> **Evidence (arXiv 2608.22366):** for vision-LLM Arabic OCR, sharpening is the
> one preprocessing step shown to *help*; binarization, denoising and contrast
> enhancement *degrade* it. Prefer `PREPROCESS_MODE=sharpen` or set
> `PREPROCESS_VLM_SAFE=true` rather than `full` when OCR-ing with the VLM.

## Thermal guard

| Variable | Default | Purpose |
|----------|---------|---------|
| `THERMAL_GUARD_ENABLED` | `true` | Enable thermal throttling |
| `THERMAL_MAX_TEMP` | `70.0` | Pause threshold (°C) |
| `THERMAL_CRITICAL_TEMP` | `75.0` | Peak that forces cooldown (°C) |
| `THERMAL_POLL_INTERVAL` | `5.0` | Cooldown wait (s) |
| `THERMAL_MONITOR_INTERVAL` | `2.0` | Background sampling (s) |
| `THERMAL_INTER_PAGE_DELAY` | `2.0` | Delay between pages (s) |

## Remote backends (all fail-soft)

| Variable | Default | Purpose |
|----------|---------|---------|
| `REDIS_ENABLED` | `false` | Idempotent state tracking |
| `POSTGRES_ENABLED` | `false` | Execution logs / metadata |
| `ELASTICSEARCH_ENABLED` | `false` | Full-text Arabic search |
| `QDRANT_ENABLED` | `false` | Vector RAG |
| `NEO4J_ENABLED` | `false` | Knowledge graph |
| `MYSQL_ENABLED` | `false` | Secondary relational store |

Each has host/port/credentials vars (e.g. `REDIS_HOST`, `POSTGRES_HOST`, etc.).
All are **fail-soft**: if a service is unreachable, the pipeline still produces
local output and logs a warning.

## Current `.env` (as of this writing)

The `.env` enables:
- `PREPROCESS_MODE=full` with all steps on (binarize/denoise/contrast/deskew)
- `OVERLAY_MODE=line` with `OVERLAY_DETECTOR=surya`, `OVERLAY_DEVICE=cuda`
- `THERMAL_MAX_TEMP=75.0`, `THERMAL_CRITICAL_TEMP=80.0`
- `OLLAMA_QUEUE_INFERENCE=true`
