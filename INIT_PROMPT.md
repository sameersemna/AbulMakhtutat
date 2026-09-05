
# TASK PROMPT: Enterprise Arabic PDF OCR & Multi-Format Conversion Architecture

## Role & Goal
You are an expert AI Systems Architect and Python Engineer working with GitHub Copilot (Deepseek v4 Pro) and Model Context Protocol (MCP) tools. Your goal is to build an automated, fault-tolerant, asynchronous document processing pipeline in Python to OCR complex Arabic PDF books (like the attached sample from "مجموع كتب ورسائل وفتاوى الشيخ ربيع المدخلي") and output clean, structured multi-format documents (`.md`, `.html`, `.pdf`, `.epub`, `.mobi`).

The system must run on a dual-node infrastructure:
- **Local Machine (Dell Pro Max GB10):** Runs Ollama (llama3.2-vision:11b, qwen2.5:32b), local Neo4J, and Qdrant.
- **Remote Host (`latitude`):** Runs PostgreSQL, Redis, Elasticsearch, and MySQL.

---

## Technical Architectural Requirements

### 1. Directory Preservation & Traversal
- Recursively scan the `./input/` directory for `.pdf` files.
- Mirror the exact nested folder hierarchy (e.g., `./input/Author_Name/Book_Name/vol1.pdf`) into `./output/Author_Name/Book_Name/vol1/`.
- Use Redis on `latitude` to track processed, pending, and failed files (enabling idempotent execution and resume capability).

### 2. Page Extraction & OCR Engine (Ollama Vision)
- Render PDF pages as high-DPI images ($300\text{ DPI}$) using `pdf2image` and `poppler`.
- Query the local Ollama vision model (`llama3.2-vision:11b`) on Dell Pro Max to OCR Arabic text page-by-page.
- Preserve Arabic typography, classical text, footnotes, diacritical marks (Harakat), and structural layouts (e.g., headers, sub-headers, lists).
- Post-process text using `qwen2.5:32b` via Ollama to ensure right-to-left logical reading order and clean markdown syntax.

### 3. Page Anchor & Table of Contents (ToC) Maintenance
- Inject explicit page markers into the canonical Markdown file:
```html
<div id="page-N" class="page-anchor" data-page="N"></div>
<!-- PAGE_N_START -->

```

* Build a structured Table of Contents (`toc.json` and `TOC.md`) capturing section titles, volume numbers, and their exact corresponding source page anchors (`#page-N`).

### 4. Multi-Format Output Generation

Convert the canonical Markdown into five output formats per PDF within its mapped output directory:

* `.md`: Raw canonical Markdown with inline HTML page anchors and CSS styling hints for RTL.
* `.html`: Standalone HTML5 document with embedded CSS (`direction: rtl; font-family: 'Amiri', 'Traditional Arabic', serif;`).
* `.epub`: Custom EPUB document with nav.xhtml mapping page targets, synthesized via Pandoc or `pypub`.
* `.mobi`: MOBI conversion using Pandoc / Calibre CLI tools.
* `.pdf`: Searchable, re-formatted PDF generated via Pandoc/WeasyPrint retaining exact page break boundaries matching the input.

### 5. Multi-Database & Vector Indexing Pipeline

* **PostgreSQL (`latitude`):** Store execution logs, document metadata, processing status, and ToC hierarchies.
* **Elasticsearch (`latitude`):** Index full-text page content for multi-lingual fuzzy search (Arabic analyzer enabled).
* **Qdrant (Local):** Store vector embeddings of page-level text snippets for semantic RAG search.
* **Neo4J (Local):** Construct a knowledge graph linking Authors $\rightarrow$ Books $\rightarrow$ Volumes $\rightarrow$ Chapters $\rightarrow$ Pages.