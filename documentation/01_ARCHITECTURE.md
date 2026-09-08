# 01 — Architecture

> Deep dive into the data flow, module responsibilities, and orchestration
> pattern. Read [00_OVERVIEW.md](00_OVERVIEW.md) first.

## High-level data flow

```
[ input/ Nested PDFs ]
        │  PDFProcessor.discover_pdfs() — recursive rglob("*.pdf")
        ▼
   For each PDF:
        │  PDFProcessor._render_page() — PyMuPDF render @300 DPI
        ▼
   Page image (PNG bytes)
        │  preprocess() — OpenCV deskew/binarize/denoise/contrast (if enabled)
        ▼
   Cleaned page image
        │  OCREngine.ocr_page() — Ollama vision model
        ▼
   Arabic text + Markdown
        │  text_cleanup.cleanup_text() — strip page numbers / running headers
        ▼
   PageResult (cached to .ocr_cache/<hash>/page_NNNN.txt)
        │
        ▼
   Canonical Markdown (page anchors) ──► FormatConverter.convert()
        │                                    │
        │                                    ├──► .md
        │                                    ├──► .html (WeasyPrint/Pandoc)
        │                                    ├──► .epub (Pandoc)
        │                                    ├──► .mobi (Calibre)
        │                                    └──► .pdf  (WeasyPrint)
        │
        └──► FormatConverter.write_overlay_line() ──► {stem}_overlay.pdf
```

## Module responsibilities

### `main.py` — CLI entry point
- Parses CLI flags (input/output dirs, workers, DPI, page range, preprocess,
  overlay mode, dry-run, etc.).
- Builds the `Config`, instantiates `PDFProcessor`, `FormatConverter`,
  `DatabaseSync`.
- Runs the async pipeline, writes a `run_summary.json` manifest.

### `config.py` — Env-driven configuration
- All tunables come from environment variables (see [05_CONFIGURATION.md]).
- `_load_dotenv()` reads `.env` at import time using `os.environ.setdefault`.
  **Important gotcha:** because it uses `setdefault`, any variable already set
  in the shell environment **overrides** `.env`. This caused a real bug (see
  [06_AGENT_GUIDE.md]).
- `get_config()` returns a singleton `Config`.

### `pdf_processor.py` — Orchestration
- `discover_pdfs()`: recursive scan of `input/`.
- `_render_page()`: renders a page to PNG at the configured DPI, then applies
  `preprocess()` if enabled.
- `_ocr_page_cached()`: OCRs a page, caching the result to
  `.ocr_cache/<file_hash>/page_NNNN.txt` for idempotent resume.
- `process_pdf()`: iterates pages (respecting `--pages` range), assembles
  canonical Markdown with page anchors, returns a `DocumentResult`.
- `parse_page_range()`: parses `"31-35"` or `"1,3,5-7"` specs.

### `ocr_engine.py` — Vision OCR
- Queries the local Ollama server with a vision model.
- Detects **refusal phrases** (English + Arabic) so "I can't read this"
  responses are never stored as valid OCR.
- Optional post-processing with a text model.
- See [02_OCR_ENGINE.md].

### `preprocess.py` — OpenCV preprocessing
- `preprocess(image_bytes, config)`: applies deskew, binarize, denoise,
  contrast per config.
- `enhance_contrast()`: retry strategy for washed-out pages.
- `detect_line_boxes()`: OpenCV fallback line detection.
- See [03_PREPROCESSING.md].

### `format_converter.py` — Output + overlay
- `convert()`: produces `.md`, `.html`, `.epub`, `.mobi`, `.pdf`.
- `write_overlay_line()`: the searchable overlay PDF (the most complex part).
- See [04_OVERLAY_ALIGNMENT.md].

### `surya_detection.py` / `easyocr_detection.py` — Line detection
- `detect_line_boxes_surya(image, device)`: Surya line boxes (best Arabic
  detection, mAP 79.67%).
- `detect_line_boxes_easyocr(image, device)`: EasyOCR line boxes (alternative).
- Both are lazy-loaded and cached per device.

### `measure_alignment.py` — Independent verification
- Renders overlay + scan at the same DPI, compares red overlay text vs dark
  scan text.
- Metrics: IoU, Precision, Recall, F1, mean/median distance, **LineMatch**.
- See [04_OVERLAY_ALIGNMENT.md].

### `thermal.py` — GPU thermal guard
- Monitors GPU temperature; pauses OCR when too hot.
- `wait_if_hot()` gates GPU inference (Surya/EasyOCR) too.

### `text_cleanup.py` — Post-OCR cleanup
- `strip_page_numbers()`: removes standalone page-number lines.
- `strip_running_headers()`: removes a known running header.

### `state_tracker.py` — Idempotent state
- Tracks per-file/per-page completion in `.ocr_state.json` (and optionally
  Redis).

### `toc_builder.py` — Table of contents
- Extracts headings → `toc.json` / `TOC.md`.

### `database_sync.py` — Optional backends
- Postgres, Redis, Elasticsearch, Qdrant, Neo4j, MySQL. All **fail-soft**.

## Orchestration pattern

The pipeline is **asynchronous** (`asyncio`). Key points:

1. **Inference queueing**: all Ollama calls (vision, fallback, post-process,
   embed) are serialized through a single global asyncio lock
   (`OLLAMA_QUEUE_INFERENCE=true`). This prevents GPU overload.
2. **Thermal guard**: `ThermalGuard` samples GPU temp in the background and
   pauses inference when it exceeds `THERMAL_MAX_TEMP`.
3. **Idempotency**: per-page results are cached; re-running skips completed
   pages.
4. **Fail-soft backends**: if a DB/vector service is unreachable, the pipeline
   still produces local output.

## Concurrency & thermal safety

- `OCR_WORKERS` defaults to **1** to avoid GPU overload.
- `OLLAMA_KEEP_ALIVE=5m` keeps models resident in GPU memory (no reload latency).
- `THERMAL_MAX_TEMP=75.0`, `THERMAL_CRITICAL_TEMP=80.0` — pause thresholds.
- `THERMAL_INTER_PAGE_DELAY=2.0` — delay between pages to let GPU cool.

## Output structure

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
