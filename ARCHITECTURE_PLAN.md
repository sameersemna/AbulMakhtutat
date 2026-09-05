### 1. Recommended Architecture & Tech Stack

```
   [ input/ Nested PDF Structure ]
                  │
                  ▼
  ┌───────────────────────────────┐
  │  Python Orchestration Engine  │  ──> System Libraries (Poppler / Pandoc)
  └───────────────┬───────────────┘
                  │ (HTTP / Ollama API)
                  ▼
  ┌───────────────────────────────┐
  │ Dell Pro Max (Ollama Server)  │  ──> Local Execution (Vision OCR Engine)
  └───────────────────────────────┘
                  │ (Network Storage & Search Sync)
                  ▼
  ┌───────────────────────────────┐
  │   `latitude` (Network Host)   │  ──> Storage & Indexing (PostgreSQL & Elasticsearch)
  └───────────────────────────────┘
                  │
                  ▼
   [ output/ Structured Format Bundles (.pdf, .epub, .mobi, .html, .md) ]

```

* **Primary Programming Language:** **Python 3.11+**. Python provides native binding access for PDF processing (`PyMuPDF`/`fitz`, `pdf2image`), document structure conversion (`Pandoc`, `pypub`), network client communication (`httpx`, `asyncio`), and vector/RAG database SDKs (`qdrant-client`, `neo4j`).
* **System Libraries:** `poppler-utils` (for `pdf2image`), `pandoc` (for document compilation to EPUB/MOBI), `tesseract-ocr-ara` (optional CPU fallback).
* **Orchestration Pattern:** **Asynchronous Directory Traversal Pipeline** using `asyncio` to read files recursively from `input/`, convert PDF pages into high-resolution images ($300\text{ DPI}$), query Ollama Vision models, inject page anchors (`<!-- PAGE_X -->`), and synthesize multi-format output into `output/`.
* **Metadata & RAG Pipeline (Dual-Node):**
* **Local Node (Dell Pro Max):** Executes Ollama vision extraction and indexing to local Qdrant (vector embeddings) and Neo4J (graph-based table of contents & cross-references).
* **Network Node (`latitude`):** Persists batch run logs in PostgreSQL, caches execution states in Redis (to allow pause/resume support), and pushes full-text searchable Markdown/Arabic text into Elasticsearch.



---

### 2. Recommended Local Ollama Models

To OCR complex, multi-column Arabic script with diacritics (Harakat) and classical formatting (such as the provided text from *Majmu' Kutub wa Rasa'il wa Fatawa*), you should pull the following models onto your Dell Pro Max GB10:

* `ollama pull llama3.2-vision:11b` (Primary Vision OCR Engine - best structured text extraction from complex Arabic page layouts).
* `ollama pull minicpm-v:8b` (Secondary/Fallback Vision OCR Engine - extremely efficient at multilingual optical character recognition).
* `ollama pull qwen2.5-coder:14b` or `qwen2.5:32b` (Text post-processing engine - cleans up OCR artifacts, structures Markdown formatting, extracts hierarchical Table of Contents, and formats metadata without hallucinations).

---

### 3. Output Formats & Page Anchor Architecture

To guarantee exact page-number tracking across all export target types (`.pdf`, `.epub`, `.mobi`, `.html`, `.md`):

1. **Markdown Intermediate Representation:** Every document is converted into a canonical Markdown (`.md`) file first.
2. **Explicit Page Anchors:** Insert explicit HTML page markers at every page boundary: `<div id="page-X" class="page-break" data-page="X"></div>` and `<!-- PAGE_X_START -->`.
3. **Table of Contents (ToC) Indexing:** Parse headings (`#`, `##`, `###`) and generate a unified `TOC.json` containing hierarchical titles and mapped physical/logical page numbers.
4. **Format Compilation via Pandoc & WeasyPrint:**
* **MD $\rightarrow$ HTML:** Direct rendering with custom CSS for right-to-left (RTL) typography (`direction: rtl; text-align: right;`).
* **HTML $\rightarrow$ EPUB/MOBI:** Generated via Pandoc using structural HTML headers, ensuring anchor points map directly to EPUB readers.
* **MD $\rightarrow$ PDF:** Rendered via WeasyPrint or Pandoc with CSS print styles (`@page { @bottom-right { content: counter(page); } }`) enforcing target alignment with the original source pages.



---

## Action Plan for Copilot (MCP Execution Guidelines)

Please use your active MCP server tools (FileSystem MCP, Terminal/Shell Execution, Database Connectors, HTTP Clients) to complete this project step-by-step:

1. **Environment Setup & Verification:**
* Inspect local workspace and generate `requirements.txt` (`pymupdf`, `pdf2image`, `httpx`, `asyncio`, `pandoc-run`, `qdrant-client`, `neo4j`, `redis`, `psycopg2-binary`, `elasticsearch`).
* Check availability of system utilities (`pandoc`, `pdftoppm`, `ollama`) using terminal commands.


2. **Core Pipeline Implementation:**
* Create `config.py` handling connection parameters for `dell-pro-max` (localhost Ollama, Qdrant, Neo4j) and `latitude` (Redis, Postgres, Elasticsearch).
* Write `ocr_engine.py` to handle Ollama REST API calls (`llama3.2-vision:11b`) with exponential backoff and batching.
* Write `pdf_processor.py` for asynchronous directory traversal, image conversion, and Markdown assembly with `<div id="page-X">` injection.
* Write `format_converter.py` using `Pandoc` and `WeasyPrint` to compile Markdown into `.html`, `.epub`, `.mobi`, and `.pdf`.
* Write `database_sync.py` to handle asynchronous ingestion into Redis, PostgreSQL, Elasticsearch, Qdrant, and Neo4J.


3. **CLI Interface & Verification:**
* Create a CLI entry point `main.py` using `argparse` or `click` supporting `--input-dir ./input --output-dir ./output --workers 4`.
* Execute a test run using the sample PDF in `input/1.pdf` and verify all 5 output formats maintain accurate Arabic layout, RTL direction, and valid page number anchors.

