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
#   WORKERS: --workers is ALSO top-level. Use `--workers 0`: with the default
#   (--workers 1), the forked dataloader worker can deadlock on the first
#   augmented batch (hang at `stage 0/N`, 0% CPU; worker stack ends in
#   kraken/lib/dataset/augment.py `_warp` -> torchvision op). Hit 2026-10-14
#   (14 h overnight hang); `--workers 0` validated (58/58 batches in 1:28 on GB10).
#   Fixed split (924/103 lines): train.txt / holdout.txt    (regenerate: make_holdout.py)
.venv/bin/ketos -d cuda:0 -s 42 --precision bf16-mixed --workers 0 train -f path -o models/run_x_3e4 \
  -B 16 -N 90 -q fixed --schedule cosine --cos-max 90 --cos-min-lr 1e-5 --warmup 100 -r 3e-4 --augment \
  -t train.txt -e holdout.txt
#   evaluate on the fixed holdout (prints Character/Word accuracy + confusion table):
.venv/bin/ketos -d cuda:0 --workers 0 test -m models/run_x_3e4/best_0.9690.safetensors -B 16 -f path \
  $(cat holdout.txt)
#   per-line error analysis -> models/run_x_3e4/error_report.json (worst lines, classes):
.venv/bin/python analyze_errors.py
#   honourific ligatures: the SHIPPED model keeps them in GT (annotation_nodiac
#   convention) and expands them AFTER recognition on the search path
#   (text_cleanup.normalize_arabic -> HONOURIFIC_LIGATURES). Training canonical
#   directly is worse (run_x_canon 94.67% vs 96.92% char, 2026-10-14) because the
#   spelled-out phrases are too sparse; that build remains available via
#   canonicalize_gt.py (default mode) for search-side-only datasets.
#   expand the corpus: cut crops -> rank/prefill -> HAND-CORRECT worst-first, then rebuild:
.venv/bin/python make_line_crops.py --pdf input/rabee/majmua/1.pdf --pages 61-80
.venv/bin/python prioritize_lines.py --data annotation/ --prefill --top 400
#   ... correct annotation/images/*.gt.txt by hand first (MACHINE labels are not training data) ...
#   fold hand-corrected lines into the shipped dataset + manifests (idempotent;
#   rebuilds annotation_nodiac in place, holdout stays frozen, new lines -> train):
.venv/bin/python canonicalize_gt.py --src annotation --strip-diacritics \
    --no-expand-ligatures --dst annotation_nodiac \
    --train train.txt --holdout holdout.txt \
    --train-out train.txt --holdout-out holdout.txt --append-new
#   (verified: the rebuild reproduces the 1027 original annotation_nodiac GT files
#    byte-identically; append-new is idempotent and warns until lines are corrected)
#   retrain on the expanded dataset (same recipe as run_x_3e4):
.venv/bin/ketos -d cuda:0 -s 42 --precision bf16-mixed --workers 0 train -f path -o models/run_x_v2 \
  -B 16 -N 90 -q fixed --schedule cosine --cos-max 90 --cos-min-lr 1e-5 --warmup 100 -r 3e-4 --augment \
  -t train.txt -e holdout.txt
#   use the trained model as the pipeline engine (fail-soft to Ollama):
VISION_BACKEND=kraken .venv/bin/python eval_ocr.py --engine kraken \
    --cache-dir probe/_ocr_cache_kraken --report probe/eval_report_kraken.json
VISION_BACKEND=kraken .venv/bin/python prefill_line_gt.py --data annotation/
.venv/bin/python prioritize_lines.py --data annotation/ --prefill --top 400   # active-learning order
.venv/bin/python ab_lines.py --engines kraken,ollama --manifest holdout.txt    # line-level engine A/B

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
| 5 | Text normalization | **DONE** | `text_cleanup.normalize_arabic` (+ `expand_honourifics`, `HONOURIFIC_LIGATURES`, shared with `canonicalize_gt.py`); wired into ES/Qdrant search copies; `OCR_NORMALIZE_SEARCH`; `tests/` | Canonical `.md` output never normalized. Honourific ligatures (ﷺ …) expanded to phrases by default on the search path (2026-10-15). 53 unit tests pass. |
| 6 | LoRA fine-tuning | **DEFERRED** | — | Needs annotated ground truth + GPU training. |

### Post-review fixes (applied)
- `eval_ocr.py --engine` now actually selects the backend, and the report records the real engine used.
- `HFVisionEngine` falls back to Ollama when the weights fail to load (fail-soft implemented).
- `OCR_NORMALIZE_SEARCH` default flipped to `false` (opt-in; requires a reindex) to avoid silent index/embedding changes.
- Shared `prepare_page_image()` used by both backends (no prepare/downscale drift).
- `pdf_processor` checks the per-page cache before rendering (no wasted render/score on cached pages).
- Removed the dead `--overlay-dir` option; added optional C-accelerated (rapidfuzz) edit distance.

### Kraken Route A follow-up (2026-10-14)

- **Training**: `run_x` (`-r 1e-4`, 60 ep) was undertrained — holdout char accuracy 0.7511,
  and a training-slice test showed only 80.3% (underfit, not overfit). Retrained
  `run_x_3e4` (`-r 3e-4`, 90 ep): best val 0.9690 char / 0.860 word
  (`models/run_x_3e4/best_0.9690.safetensors`).
- **Holdout evaluation** (`ketos test`, 103 lines / 6163 chars): 96.92% character /
  86.36% word accuracy (24 insertions / 67 deletions / 99 substitutions).
- **Per-line error analysis** (`analyze_errors.py`, logical order, kraken inference path):
  96.84% char / 86.36% word, 43/103 lines exact. Weak classes: Arabic-Indic **digits 64.3%
  right**, honourific ligatures 83.3%, punctuation 95.5%; worst lines are Quranic quotations
  (`ٱ` wasla) and footnote/number lines. Top confusions: deletions of ل/ن/ب and spaces;
  substitutions ي↔ب، ن↔ل، خ↔ع، ث↔ت، أ↔ا.
- **Honourific canonicalization** (`canonicalize_gt.py`): 11 presentation-form ligatures
  (ﷺ، ﷻ، ﷿، ﵀، ﵁، ﵂، ﵃، ﵄، ﵇، ﵉، ﷾) expanded to spelled-out phrases into a new dataset
  `annotation_canon/` (102/1027 lines changed, +1434 chars) with `train_canon.txt` /
   `holdout_canon.txt`; ornate Quran parentheses kept. Canonical retrain ran 2026-10-14
   (`models/run_x_canon`, alphabet 68 classes, `alphabet mismatch` warning gone).
- **Canonical retrain result** (finished 2026-10-14 ~17:52 local, 90 ep, `--workers 0`):
  best `models/run_x_canon/best_0.9470.safetensors` (val 0.9470 char, reached at epoch 75; no
  later improvement). `ketos test` on `holdout_canon.txt` (103 lines / **6453** chars):
  **94.67% character / 80.78% word** accuracy (23 ins / 200 del / 121 sub; 34/103 lines exact
  vs 43/103) — *worse* than
  `run_x_3e4` (96.92% / 86.36%, 6163 chars). The regression is the canonicalization itself,
  not Arabic recognition: canonicalization adds exactly **+290 chars** to the holdout, all in
  spelled-out honourifics (`صلى الله عليه وسلم` x13, `عليه الصلاة والسلام` x3, `عز وجل` x3,
  `رضي الله عنه` x2, `جل جلاله` x2, `رحمه الله` x1); the holdout error count rises by +154
  (190→344), and **all 10 worst lines** contain those phrases, which the model mangles/drops
  (e.g. `صلى الله عليه وسلم` → `صد ليه ولم`). The `ﷺ`-type ligatures occurred in only
  ~102/1027 training lines — too sparse for a 4 M-param CTC model to learn the long expanded
  form; digits stay weak (59.5% vs 64.3%). Verified the expansion itself is clean
  (byte-identical rebuild). **Implication:** keep canonical GT as the correct/searchable target,
  but the model must see the expanded phrases far more often (oversample the honourific lines),
  or apply the ligature→phrase mapping *after* recognition instead of training on it.
- **Kraken engine** (`kraken_engine.py`, `VISION_BACKEND=kraken`): page path = bundled BLLA +
  recognition with baseline→bbox conversion (see `05_CONFIGURATION.md`); line path =
  `ocr_line` (96.8% char). `eval_ocr.py --engine kraken` and `prefill_line_gt.py` wired in.
- **Tooling**: `ab_lines.py` (line-level engine A/B against the trustworthy holdout),
  `prioritize_lines.py` (confidence-ranked annotation worklist + machine prefill).
- **Line A/B result** (`ab_lines.py`, 103 holdout lines, `probe/ab_lines_report.json`):
  Kraken CER 3.26% verbatim / 3.65% normalized, WER 13.73% / 13.42%;
  Ollama qwen2.5vl:7b (line prompt) CER 23.54% / 6.83%, WER 41.29% / 15.21%.
  The in-repo Kraken model wins in every mode and is ~30x faster (14 s vs ~7 min).
- **Annotation batch (2026-10-14)**: 468 line crops cut from book 1 pages 61-80; ranked with the
  trained model into `annotation/worklist.tsv` and machine-prefilled worst-first (384 lines) for
  hand correction. Once corrected, fold the lines into the shipped dataset with the
  ligature-preserving rebuild + `--append-new` (runbook Phase 3a above):
  the rebuild was verified byte-identical on the 1027 original `annotation_nodiac` GT files, the
  holdout stays frozen, appending is idempotent, and the tool warns that appended lines must be
  human-corrected. Until corrected, the prefills stay out of training (they are not in either
  manifest).
- **Harness guard**: `eval_ocr.py` now flags machine-generated ground truth (`gt_warning` in the
  report plus a log warning) so the probe set cannot silently score against LLM/markdown junk.
- **Probe caveat**: the probe ground truth under `probe/probe/ground_truth/` is LLM/markdown
  junk, so every page-level probe CER number (ollama/hf/kraken reports) is meaningless until
  it is re-corrected (see `probe/README.md`).

### Deployment + follow-ups (2026-10-15)

- **Kraken is the default backend** (`config.py`): `VISION_BACKEND=kraken` →
  `models/run_x_3e4/best_0.9690.safetensors`; verified the factory returns `KrakenEngine`,
  falls back to Ollama with a warning when the weights are missing, and `ollama`/`hf`
  overrides still work.
- **Search-side ligature expansion**: `text_cleanup.normalize_arabic` now expands the 11
  honourific presentation-form ligatures to phrases *before* NFKC (NFKC decomposes only
  U+FDFA/FDFB). The map lives in `text_cleanup.HONOURIFIC_LIGATURES` and
  `canonicalize_gt.py` now imports it (single source of truth, no drift). Raw `.md`/`.txt`
  output keeps the typographic ligature; only normalized/search copies expand. New tests
  cover expansion, idempotency, the disable flag and ornate parens.
- **Dataset builder** (`canonicalize_gt.py`) generalized for fold-in:
  `--no-expand-ligatures` (ligature-preserving build), `--train-out`/`--holdout-out`,
  and `--append-new` (adds dataset files missing from both manifests to the *training*
  manifest; holdout stays frozen; idempotent; warns that appended GT must be
  hand-corrected). Verified: the rebuild reproduces all 1027 original `annotation_nodiac`
  GT files byte-identically; append-new adds 384 and is a no-op on re-run.
- **Capacity probe — DONE, decisive win (2026-10-15)**: `--arch ppocrv6 --variant medium`
  (32.9 M params: 15.6 M backbone + 17.3 M NRTR attention decoder), same split/recipe,
  90 epochs ≈ 3.5 h (`models/run_ppocrv6_m`). Independent `ketos test` on
  `best_0.9948.safetensors` (103 lines / 6163 chars): **99.40% character / 96.83% word**
  (5 ins / 15 del / 17 sub) vs `run_x_3e4`'s 96.92% / 86.36% (24/67/99) — **+2.5 char /
  +10.5 word points, ~5x fewer errors**. `analyze_errors.py`: 99.16% char / 95.69% word,
  **76/103 lines exact** (vs 43/103); per-category: Arabic letters 99.54%, spaces 99.32%,
  punctuation 97.77%, digits 85.7% (was 64.3%), honourifics 94.4%, ornate parens 100%.
  Training-time val (0.9948) matches the independent test, so the NRTR path is not
  optimistic. Verified through our own `KrakenEngine.ocr_line` too (loads + decodes; keeps
  the ligature classes, expanded on the search path by `normalize_arabic`).
  **Decision: the shipped default is now `models/run_ppocrv6_m/best_0.9948.safetensors`**
  (`VISION_KRAKEN_MODEL`), and `run_x_v2` after the annotation fold-in should retrain
  PP-OCRv6-medium, not VGSL. Checkpoints pruned; remaining errors are page-number lines,
  Quranic quotes (`ٱ`/`﴿﴾`) and letter-spacing joins.

### Verification

```bash
# Unit tests (43 passing)
.venv/bin/python -m unittest discover -s tests -t . -v

# Probe harness (no model needed)
.venv/bin/python eval_ocr.py --help
.venv/bin/python eval_ocr.py --score-only

# Preprocessing A/B, including the new VLM-friendly sharpen mode
.venv/bin/python ab_test.py --pdf input/rabee/majmua/1.pdf --pages 3,4,5 --mode sharpen
```

### Recommended next actions (human-in-the-loop)
1. **Deploy** — DONE (2026-10-15): `VISION_BACKEND` defaults to `kraken` with
   `VISION_KRAKEN_MODEL=models/run_ppocrv6_m/best_0.9948.safetensors` (99.40% char / 96.83%
   word; fail-soft to Ollama), and `text_cleanup.normalize_arabic` expands the 11 honourific
   ligatures to phrases (`HONOURIFIC_LIGATURES`, shared with `canonicalize_gt.py`) so
   search/RAG text is canonical without retraining. The canonical *training* question is
   settled: ship the ligature-preserving model + post-OCR expansion (canonical retrain lost
   2.2 pts of char accuracy, see above). Raw .txt output keeps the typographic ligature.
2. **Expand annotations worst-first** (the main accuracy lever left): hand-correct the 384
   prefilled crops in `annotation/worklist.tsv` (worst-first; 84 crops have no prefill),
   then fold them in with the ligature-preserving rebuild + `--append-new` (Phase 3a
   runbook) and retrain `models/run_x_v2` with **PP-OCRv6-medium** (not VGSL — see the
   capacity result above). Remaining weak spots of the new default model: page-number
   lines, Quranic quotations (`ٱ`/`﴿﴾`) and letter-spacing joins — see
   `models/run_ppocrv6_m/error_report.json`; prioritise lines that contain those.
3. **Re-correct the probe ground truth**: `eval_ocr.py --export-templates`, then hand-correct
   `probe/ground_truth/*.txt` (the current fallback files are junk — see probe/README.md).
   Page-level A/B decisions depend on it; line-level A/B already works via `ab_lines.py`.
4. **Capacity A/B** — DONE (2026-10-15): PP-OCRv6-medium beat the 4 M-param VGSL model by
   +2.5 char / +10.5 word points and is now the default (see above). No further
   architecture search needed before the annotation fold-in.
