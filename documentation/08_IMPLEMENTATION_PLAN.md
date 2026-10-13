# 08 — Implementation Plan: Arabic OCR Accuracy Improvements

> Derived from §4.7 of [`07_ARABRIC_OCR_RESEARCH.md`](07_ARABRIC_OCR_RESEARCH.md).
> Execute in order; each task lists concrete deliverables and acceptance criteria.
> **Guardrail:** read [`06_AGENT_GUIDE.md`](06_AGENT_GUIDE.md) before changing any
> code. Any change must not break the pipeline and must not overheat the GPU.

## Status legend
`TODO` · `IN PROGRESS` · `DONE` · `BLOCKED` (needs data/hardware/user input) · `DEFERRED`

---

## Task 1 — Probe set + CER/WER evaluation harness  [DONE]
**Why:** Nothing else can be validated without a local metric. Enables every
other task.

**Deliverables**
- `eval_ocr.py` — computes CER, WER, and LineMatch for a model/engine against a
  ground-truth probe set. Two scoring modes:
  - `verbatim` (diacritics/punctuation count)
  - `normalized` (Unicode NFC, strip tatweel, normalize alef/hamza/ya/ta-marbuta,
    normalize Arabic-Indic + Latin digits)
- `probe/manifest.json` — schema listing PDF, page, and ground-truth `.txt` path.
- `probe/README.md` — how to add pages and where ground truth lives.
- Ground-truth capture helper: export current OCR output as a `.txt` template per
  page for human correction (`--export-templates`).
- Reuse existing `measure_alignment.py` LineMatch for overlay regression.

**Acceptance criteria**
- `python eval_ocr.py --help` works with no model running.
- Given `probe/manifest.json` + a model, prints per-page and aggregate CER/WER
  (both modes) and writes `eval_report.json`.
- Handles the empty/missing-GT case without crashing.

**Depends on:** none. **Unblocks:** Tasks 2–4.

---

## Task 2 — Arabic-fine-tuned VLM candidate + A/B capability  [PARTIAL — capability DONE; live run BLOCKED]
**Why:** Arabic-specialized VLMs likely beat generic `qwen2.5vl:7b`; QARI-2B also
cuts GPU heat.

**Deliverables**
- Config option to select an alternative vision backend without code changes
  (env-driven, fail-soft), e.g. a `vision_backend` switch and HF-model id.
- An adapter path for HuggingFace VLMs (QARI-2B first) alongside the Ollama path,
  preserving the existing Ollama default.
- `ab_test.py` extended (or a new script) to A/B two engines on the probe set and
  print CER/WER deltas.
- Docs: which model to pull/download and the exact env vars.

**Acceptance criteria**
- Default behaviour unchanged when the new option is unset.
- A/B run produces a side-by-side CER/WER table.
- If the HF model is absent, pipeline falls back to Ollama with a warning (no crash).

**Depends on:** Task 1. **Note:** actual A/B needs the model downloaded (resources).

---

## Task 3 — Resolution for diacritized pages  [DONE]
**Why:** 300 DPI + 1500 px downscale likely strips diacritics.

**Deliverables**
- Config: `OCR_DPI` guidance/default raise path to 400 for diacritized pages;
  `OLLAMA_MAX_IMAGE_DIM` raise path to ~2500.
- A per-page "diacritized/heavy" heuristic that selects the higher cap while
  keeping the default unchanged when false.
- Config docs updated.

**Acceptance criteria**
- Defaults unchanged; heuristic is opt-in via env.
- Higher-resolution path is exercised and logged (which DPI/cap used per page).

**Depends on:** Task 1 (to measure the effect).

---

## Task 4 — VLM-path preprocessing: no binarize, opt-in sharpen  [DONE]
**Why:** Binarization/denoise/contrast degrade VLMs on Arabic; sharpening helps.

**Deliverables**
- Ensure binarize/denoise/CLAHE are never applied on the VLM OCR call path
  (reserve for a traditional-OCR path only).
- New opt-in `light+`/sharpen mode (unsharp mask) in `preprocess.py`.
- `enhance_contrast` retry re-tested; add a sharpening retry alternative.
- Preprocessing docs updated with the evidence.

**Acceptance criteria**
- With preprocessing "off" (default), VLM path unchanged.
- Sharpen mode is toggleable and preserves dimensions (deskew/overlay mapping intact).
- `ab_test.py` can compare off vs sharpen.

**Depends on:** Task 1.

---

## Task 5 — Arabic text normalization for search/quality  [DONE]
**Why:** Improves search/RAG and enables normalized CER scoring.

**Deliverables**
- `text_cleanup.py`: add `normalize_arabic(text, ...)` implementing NFC, tatweel
  strip, alef/hamza/ya/ta-marbuta normalization, digit normalization,
  presentation-form (FB50–FEFF) → base-letter conversion, optional diacritic strip.
- A config/CLI switch to apply normalization to output (and a pure function reusable
  by the harness).
- Unit tests for the normalization function.

**Acceptance criteria**
- Pure function, no I/O, importable by `eval_ocr.py`.
- Tests cover: presentation forms, Arabic-Indic digits, alef variants, tatweel,
  diacritics, mixed bidi/Arabic-Latin strings.
- Default pipeline output is unchanged unless normalization is explicitly enabled.

**Depends on:** none (but used by Task 1's normalized mode).

---

## Task 6 — LoRA fine-tuning on our typeface  [DEFERRED]
**Why:** Largest accuracy lever if probe results plateau.

**Deliverables (when unblocked):** annotated line/word dataset (≥800–1,000 lines),
training script (Qwen2.5-VL LoRA), eval report vs baseline.

**Blocked on:** hand-annotated ground truth for our font + a training script + GPU
training time (GPU present: GB10, CUDA OK).

### Unblock checklist (all must be true)
- [ ] `probe/ground_truth/*.txt` hand-corrected (≥30–50 pages) and a baseline normalized CER recorded.
- [ ] Best off-the-shelf model (Task 2 A/B) still misses the target CER after Task 3–4 tuning.
- [ ] ≥800–1,000 annotated lines of the target typeface (or an accepted synthetic set).
- [ ] Training deps installed: `peft`/`trl`/`bitsandbytes` (LoRA) or `kraken[pdf]` (Route A).
- [ ] A training entry point authored (task deliverable) + `eval_ocr.py` used for the before/after.

### Runbook (commands)
```bash
# Phase 0 — baseline (needs Ollama running)
ollama pull qwen2.5vl:7b
.venv/bin/python eval_ocr.py --engine ollama
.venv/bin/python eval_ocr.py --export-templates      # then hand-correct probe/ground_truth/*.txt
.venv/bin/python eval_ocr.py --score-only

# Phase 1 — live model A/B (separate caches/reports per engine!)
.venv/bin/pip install -U accelerate qwen_vl_utils
.venv/bin/python eval_ocr.py --engine ollama --cache-dir probe/_ocr_cache/ollama --report probe/eval_report_ollama.json --force
.venv/bin/python eval_ocr.py --engine hf     --cache-dir probe/_ocr_cache/hf     --report probe/eval_report_hf.json --force

# Phase 3a — Route A (recommended, no training script): typeface-specific Kraken model
.venv/bin/pip install "kraken[pdf]"
#   'path' format needs <stem>.gt.txt NEXT TO each image (empty files abort ketos):
for f in annotation/text/*.txt; do cp "$f" "annotation/images/$(basename "${f%.txt}").gt.txt"; done
#   DEVICE: -d/--device is a TOP-LEVEL ketos option, so it goes BEFORE the subcommand.
#   (It is NOT accepted after `train`.) Or set CUDA_VISIBLE_DEVICES=0 instead.
.venv/bin/ketos -d cuda:0 --precision bf16-mixed train -f path -o models/ourbook annotation/images/*.png
#   evaluate a trained model on held-out lines (prints CER/accuracy):
ls models/ourbook
.venv/bin/ketos test -m models/ourbook/<model file> -f path annotation/images/*.png

# Phase 3b — Route B (Task 6 as written): LoRA Qwen2.5-VL — trainer: train_lora.py
.venv/bin/pip install -U peft         # required; bitsandbytes only if you want 4-bit
#   cut pages into single-line crops (Surya; skips eval-manifest pages, rejects slivers/merged lines)
.venv/bin/python make_line_crops.py --pdf input/rabee/majmua/1.pdf --pages 41-60
#   bootstrapping labels: pre-annotate line crops, then CORRECT every .gt.txt by hand
.venv/bin/python prefill_line_gt.py --data annotation/
find annotation/images -name '*.gt.txt' -size +0c | wc -l   # must be > 0
#   lists characters with < 20 instances: annotate lines containing them next
.venv/bin/python train_lora.py --check-data --data annotation/
.venv/bin/python train_lora.py --model Qwen/Qwen2.5-VL-3B-Instruct --data annotation/ \
    --out models/qwen-arabic-lora --epochs 3 --accum 8
#   optional variant for search-only output: targets without harakat (fewer classes)
.venv/bin/python train_lora.py --strip-diacritics --data annotation/ --out models/qwen-arabic-lora-plain
VISION_HF_MODEL=./models/qwen-arabic-lora .venv/bin/python eval_ocr.py --engine hf --cache-dir probe/_ocr_cache/lora --report probe/eval_report_lora.json --force
```
> **Note:** `eval_ocr.py` caches per entry id, so always use a distinct
> `--cache-dir` and `--report` per engine in an A/B. For LoRA inference, save
> **merged** weights (`merge_and_unload`) so `from_pretrained` can load them.


---

## Execution log

| # | Task | Status | Delivered | Notes |
|---|------|--------|-----------|-------|
| 1 | Probe set + harness | **DONE** | `eval_ocr.py`, `probe/manifest.json`, `probe/README.md`, `probe/ground_truth/` | CER/WER verbatim+normalized; optional overlay LineMatch reused from `measure_alignment.py`. Acceptance verified: `--help`, `--score-only` on empty GT, template export. |
| 2 | QARI/HF candidate + A/B | **PARTIAL** | `hf_vlm_engine.py` (`HFVisionEngine` + `build_ocr_engine`), `VisionConfig`, `eval_ocr.py --engine hf`, `ab_test.py` uses factory | Code path + fail-soft fallback verified. **Live A/B BLOCKED:** needs model download (~4 GB) + GPU time + `qwen_vl_utils`/`accelerate` (missing). |
| 3 | Diacritized resolution | **DONE** | `preprocess.score_diacritic_density`, `OCR_DPI_HEAVY`/`OCR_HEAVY_*`/`OLLAMA_MAX_IMAGE_DIM_HEAVY`, per-page switch in `pdf_processor._process_one` | Off by default; base behaviour unchanged. Heuristic + per-call `max_dim` verified. |
| 4 | VLM-path preprocessing | **DONE** | `preprocess._sharpen`/`sharpen_image`, `PREPROCESS_MODE=sharpen`, `PREPROCESS_VLM_SAFE`, `--preprocess-sharpen`, `OLLAMA_RETRY_MODE`, `ab_test.py --mode` | Defaults unchanged; sharpen & vlm-safe paths + dimension preservation verified. |
| 5 | Text normalization | **DONE** | `text_cleanup.normalize_arabic`; wired into ES/Qdrant search copies; `OCR_NORMALIZE_SEARCH`; `tests/` | Canonical `.md` output never normalized. 26 unit tests pass. |
| 6 | LoRA fine-tuning | **DEFERRED** | — | Needs annotated ground truth + GPU training. |

### Post-review fixes (applied)
- `eval_ocr.py --engine` now actually selects the backend, and the report records the real engine used.
- `HFVisionEngine` falls back to Ollama when the weights fail to load (fail-soft implemented).
- `OCR_NORMALIZE_SEARCH` default flipped to `false` (opt-in; requires a reindex) to avoid silent index/embedding changes.
- Shared `prepare_page_image()` used by both backends (no prepare/downscale drift).
- `pdf_processor` checks the per-page cache before rendering (no wasted render/score on cached pages).
- Removed the dead `--overlay-dir` option; added optional C-accelerated (rapidfuzz) edit distance.

### Verification

```bash
# Unit tests (26 passing)
.venv/bin/python -m unittest discover -s tests -t . -v

# Probe harness (no model needed)
.venv/bin/python eval_ocr.py --help
.venv/bin/python eval_ocr.py --score-only

# Preprocessing A/B, including the new VLM-friendly sharpen mode
.venv/bin/python ab_test.py --pdf input/rabee/majmua/1.pdf --pages 3,4,5 --mode sharpen
```

### Recommended next actions (human-in-the-loop)
1. **Fill the probe ground truth** (`eval_ocr.py` → `--export-templates` → hand-correct
   `probe/ground_truth/*.txt`). Every other decision depends on this.
2. **Run the live model A/B** for Task 2: install `qwen_vl_utils accelerate`,
   download `NAMAA-Space/Qari-OCR-v0.2-VL-2B-Instruct`, then
   `eval_ocr.py --engine hf` vs `--engine ollama` and compare `eval_report.json`.
3. **A/B preprocessing** on the probe set: baseline vs `--preprocess-sharpen` vs
   `PREPROCESS_VLM_SAFE=true`.
4. Enable `OCR_HEAVY_HEURISTIC=true` and confirm diacritized pages improve in
   normalized CER before adopting.
