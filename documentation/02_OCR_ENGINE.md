# 02 — OCR Engine

> How the vision model transcribes Arabic pages, detects refusals, and
> post-processes text. Read [01_ARCHITECTURE.md](01_ARCHITECTURE.md) first.

## Overview

`ocr_engine.py` queries the local Ollama server with a **vision model** to
transcribe Arabic page images. It is the heart of the OCR accuracy story.

## Vision model selection

| Model | Status | Notes |
|-------|--------|-------|
| `qwen2.5vl:7b` | ✅ **Primary** | Excellent Arabic OCR, ~5-15s/page, low GPU heat (6GB) |
| `qwen3-vl:latest` | ✅ Fallback | 6.1GB, good quality |
| `qwen3-vl:32b-instruct-q8_0` | ⚠️ | 35GB, best quality but caused thermal crashes (~86s/page @ 95% GPU) |
| `llama3.2-vision:11b` | ❌ | Broken on this Ollama build (`mllama` arch → HTTP 500) |
| `minicpm-v:8b` | ⚠️ | Poor Arabic (hallucinates watermarks) |

**Why `qwen2.5vl:7b`?** It balances accuracy and thermal safety. The 32b model
is more accurate but overheats the GPU. The 7b model is the sweet spot.

## The vision prompt

The `_VISION_PROMPT` instructs the model to:

- Transcribe Arabic **exactly** as it appears.
- **Preserve all diacritical marks** (Harakat: fatha, damma, kasra, sukun,
  shadda, tanwin).
- Preserve classical typography, footnotes, and structural layout.
- Output clean Markdown (`#`, `##`, `###` for headings, `-` for lists, `>` for
  quotes).
- Keep right-to-left reading order.
- **Not** translate, summarize, or add commentary.
- **Not** repeat watermarks/stamps/library logos more than once.
- Skip unreadable regions silently rather than guessing.

## Refusal detection (critical)

The vision model sometimes **refuses** to transcribe (instead of returning
Arabic text), especially on poor-quality scans. It emits English phrases like
"overexposed", "washed out", "no readable text", or Arabic phrases like
"لا يوجد نص قابل للقراءة" (no readable text).

`ocr_engine.py` maintains `_REFUSAL_PHRASES` (English) and
`_ARABIC_REFUSAL_PHRASES` (Arabic). If the model's output matches a refusal
phrase, the pipeline treats it as a **failed page** and retries (with
`enhance_contrast()` as a retry strategy) rather than storing garbage.

**Guardrail:** never remove or weaken these refusal lists. They are what
prevent "I can't read this" from being stored as valid OCR output.

## Post-processing (optional)

`_POSTPROCESS_PROMPT` uses a text model (`qwen2.5-coder:14b`) to correct OCR
artifacts:

- Fix broken words, misread letters, stray Latin characters.
- **Do not** add/remove/reorder content — only correct what is clearly wrong.
- Preserve all diacritics and Markdown structure.
- If a word is ambiguous, leave it unchanged.

Controlled by `OLLAMA_POSTPROCESS_ENABLED` (default `false`).

## Bounding-box prompts

`ocr_engine.py` also has prompts for **word-level** (`_BBOX_PROMPT`) and
**line-level** (`_LINE_BBOX_PROMPT`) OCR with bounding boxes. These are used
for overlay alignment. The line-level prompt is preferred because Arabic word
boundaries are unreliable (cursive script + diacritics), while line boundaries
are detected reliably.

## Retry & backoff

- `OLLAMA_MAX_RETRIES=3`, `OLLAMA_BACKOFF_BASE=2.0` — exponential backoff.
- `OLLAMA_TIMEOUT=300` — per-request timeout.
- `OLLAMA_TEMPERATURE=0.0` — deterministic output (critical for OCR accuracy).

## Key functions

- `ocr_page(image_bytes)` — OCR a single page image.
- `ocr_page_with_line_boxes(image_bytes)` — OCR + line bounding boxes.
- `postprocess(text)` — optional text correction.
- `close()` — cleanup.

## Guardrails for agents

1. **Never change the vision prompt** to allow translation, summarization, or
   commentary — the model must output ONLY transcribed text.
2. **Never remove refusal phrases** — they prevent garbage from being stored.
3. **Keep `OLLAMA_TEMPERATURE=0.0`** — any nonzero temperature risks
   hallucinated Arabic.
4. **Don't switch to a larger model** without verifying thermal safety (the 32b
   model caused crashes).
5. **Test on pages 31-35** of `input/rabee/majmua/1.pdf` (the main text) before
   and after any OCR-engine change.
