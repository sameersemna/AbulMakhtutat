# Arabic PDF OCR & Multi-Format Conversion Pipeline

An automated, fault-tolerant, asynchronous document-processing pipeline that OCRs
complex Arabic PDF books (image scans with no text layer) and produces clean,
structured multi-format output (`.md`, `.html`, `.pdf`, `.epub`, `.mobi`).

## Architecture

```
[ input/ Nested PDFs ]  →  Python Orchestration (asyncio)  →  Ollama Vision OCR (local)
        │                                                          │
        │  PyMuPDF render @300 DPI                                 │  qwen3-vl:32b
        ▼                                                          ▼
   Page images ────────────────────────────────────────────►  Arabic text + Markdown
                                                                   │
        ┌──────────────────────────────────────────────────────────┤
        ▼                                                          ▼
  Canonical Markdown (page anchors)                    Optional DB/vector sync
        │                                              (Postgres, Redis, ES,
        ▼                                               Qdrant, Neo4j, MySQL)
  Pandoc / WeasyPrint  →  .md .html .epub .mobi .pdf
```

## Verified Model Findings (important)

| Model | Status |
|-------|--------|
| `qwen3-vl:32b-instruct-q8_0` | ✅ **Primary** — excellent Arabic OCR with diacritics |
| `llama3.2-vision:11b` | ❌ Broken on this Ollama build (`mllama` arch → HTTP 500) |
| `minicpm-v:8b` | ⚠️ Poor Arabic (hallucinates watermarks) — fallback only |
| `Kimi-VL-A3B-Thinking` | ❌ No multimodal support via `/api/generate` |

## Requirements

- Python 3.11+
- System: `pandoc`, `poppler` (optional), `ebook-convert` (Calibre, for MOBI)
- Ollama running locally with `qwen3-vl:32b-instruct-q8_0` pulled

```bash
pip install -r requirements.txt
ollama pull qwen3-vl:32b-instruct-q8_0
```

## Usage

```bash
# Full run (all PDFs, all formats)
python main.py --input-dir ./input --output-dir ./output --workers 4

# Test on a few pages first
python main.py --max-pages 2 --workers 1

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
| `--workers` | `4` | Concurrent OCR workers |
| `--dpi` | `300` | Render DPI |
| `--max-pages` | `0` (all) | Limit pages per PDF (testing) |
| `--skip-formats` | `""` | Comma-separated formats to skip |
| `--no-postprocess` | off | Disable text post-processing |
| `--dry-run` | off | List PDFs only |

## Output Structure

Input hierarchy is mirrored into `output/`:

```
input/rabee/majmua/1.pdf
  → output/rabee/majmua/1/
      1.md        # canonical Markdown with page anchors
      1.html      # standalone HTML5 (RTL, Amiri font)
      1.epub      # EPUB3 with nav
      1.mobi      # MOBI (if Calibre installed)
      1.pdf       # searchable PDF (WeasyPrint)
      toc.json    # structured ToC
      TOC.md      # human-readable ToC
      style.css   # RTL stylesheet
```

## Configuration

All settings are environment-driven (see `config.py`). Key variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_VISION_MODEL` | `qwen3-vl:32b-instruct-q8_0` | Primary vision model |
| `OLLAMA_VISION_FALLBACK` | `minicpm-v:8b` | Fallback vision model |
| `OLLAMA_POSTPROCESS_ENABLED` | `false` | Enable text post-processing |
| `OLLAMA_MAX_IMAGE_DIM` | `1500` | Downscale long edge before OCR |
| `OCR_DPI` | `300` | Render DPI |
| `OCR_WORKERS` | `4` | Concurrency |
| `REDIS_ENABLED` | `false` | Remote state tracking |
| `POSTGRES_ENABLED` | `false` | Execution logs / metadata |
| `ELASTICSEARCH_ENABLED` | `false` | Full-text Arabic search |
| `QDRANT_ENABLED` | `false` | Vector RAG |
| `NEO4J_ENABLED` | `false` | Knowledge graph |
| `MYSQL_ENABLED` | `false` | Secondary relational store |

All remote backends are **fail-soft**: if a service is unreachable, the pipeline
still produces local output and logs a warning.

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
| `ocr_engine.py` | Ollama vision OCR + post-processing |
| `pdf_processor.py` | Async traversal, rendering, Markdown assembly |
| `toc_builder.py` | ToC extraction → `toc.json` / `TOC.md` |
| `format_converter.py` | Pandoc/WeasyPrint multi-format output |
| `database_sync.py` | Optional DB/vector/graph ingestion |
| `main.py` | CLI entry point |
