# 07 — Arabic OCR Research: State of the Art, Open-Source Landscape & Recommendations

> **Provenance.** This document captures a research investigation into current
> solutions and methodologies for improving Arabic OCR, conducted 2026-10-10,
> and synthesized against *this* project's actual pipeline (`qwen2.5vl:7b` via
> Ollama, OpenCV preprocessing, Surya line detection, searchable overlay).
> It is a research/context document, not a spec. Actionable items are tracked in
> [`08_IMPLEMENTATION_PLAN.md`](08_IMPLEMENTATION_PLAN.md).

## Research brief (original request)

Investigate current solutions and methodologies for improving Arabic OCR across
GitHub, Stack Overflow/technical forums, arXiv/Google Scholar, and blogs, and
synthesize: (1) most effective modern architectures, (2) promising open-source
tools/codebases, (3) key challenges and community/research solutions, and
(4) concrete, actionable recommendations for this project (data augmentation,
model selection, preprocessing, post-processing).

> **Note on dates.** The environment date is Oct 2026. Some benchmark/model names
> below (e.g. "Surya 2", "Gemini 3.8 Flash") come from sources on that timeline.
> Vendor-run or single-source figures are flagged. Any published Arabic CER
> should be treated as non-predictive of our corpus until validated locally.

---

## 0. Where the project stands today (baseline)

| Component | Current implementation | Assessment |
|---|---|---|
| Vision OCR | `qwen2.5vl:7b` primary, `qwen3-vl` fallback (`config.py:85-86`, `ocr_engine.py:212`) | Solid general VLM, not Arabic-specialized. KITAB-Bench: Qwen2.5-VL-7B **CER 1.20** vs AIN-7B **0.20** on image-to-text. |
| Preprocessing | OpenCV deskew/binarize/denoise/CLAHE, **default `off`** (`config.py:115`, `preprocess.py:124`) | Defaulting off is well-aligned with latest VLM evidence (§3). |
| Line detection | Surya (mAP@50 **79.67**, best in KITAB-Bench) | Correct choice; strongest open line detector. |
| Post-processing | Qwen2.5-Coder text model, **disabled** (`ocr_engine.py:405`) | Correctly gated; LLM post-correction can hurt (§3). |
| Overlay | Custom invisible text layer, line-aligned | Could be validated against OCRmyPDF's RTL renderer (§4.5). |
| DPI / image cap | 300 DPI, downscale to 1500 px (`config.py:102,141`) | **Likely too aggressive for diacritized text** — see §4.3. |

---

## 1. Most effective modern architectures and techniques

Three paradigms, ranked by current effectiveness on printed Arabic.

### 1.1 Vision-Language Models (VLMs) — dominant for printed Arabic

KITAB-Bench (ACL 2025, MBZUAI; [arXiv 2502.14949](https://arxiv.org/abs/2502.14949))
found closed VLMs beat traditional frameworks by **~60% relative CER**:

| Group | Model | CER ↓ | WER ↓ |
|---|---|---|---|
| Closed | Gemini-2.0-Flash | **0.13** | 0.32 |
| Closed | GPT-4o | 0.31 | 0.55 |
| Open | **AIN-7B** | 0.20 | 0.28 |
| Open | Qwen2.5-VL-7B | 1.20 | 1.41 |
| Framework | Tesseract | 0.54 | 0.84 |
| Framework | EasyOCR | 0.58 | 0.89 |
| Framework | PaddleOCR | 0.79 | 1.02 |

Independent 2026 testing ([arelion.dev](https://arelion.dev/arabic-ocr/), single
author, code public) on 44 real Arabic scans/photos: **Surya 3.9% CER**, Apple
Live Text 5.0%, EasyOCR 18.7%, **Tesseract 19.7%**, PaddleOCR 33.5%; best overall
Gemini Flash (~1.6–1.9%). Key insight: **traditional engines fall apart on real
scans; VLMs and Surya hold up.**

### 1.2 Arabic-specialized VLMs (fine-tuned Qwen2-VL / Qwen2.5-VL)

The dominant open recipe is **LoRA fine-tuning of Qwen2-VL/Qwen2.5-VL on Arabic**:

- **QARI-OCR** ([arXiv 2506.02295](https://arxiv.org/abs/2506.02295)) — Qwen2-VL-2B
  fine-tune. v0.2 claims **CER 0.061**; v0.3 is layout-aware (HTML/Markdown),
  trained on 10k synthetic pages in **11 h on one A6000**. 2B ⇒ low GPU heat.
- **DIMI-Arabic-OCR v2** — Qwen2.5-VL-7B LoRA, **CER 0.112 / WER 0.305**, strong diacritics.
- **AIN-7B** (MBZUAI) — Arabic-inclusive LMM, CER 0.20 on KITAB.
- **Fanar-2-Oryx-IVU** (QCRI, **Apache-2.0**) — Qwen2.5-VL-7B based, 62M bilingual
  examples, Arabic calligraphy recognition.

### 1.3 Hybrid CNN+RNN/Transformer and TrOCR-style seq2seq

Still SOTA for **handwriting and typeface-specific print**:

- **TrOCR / ALNASIKH** (IEEE MIUCC 2023) — Arabic-tailored TrOCR; **Qalam-Net V2**
  is an open TrOCR-based Arabic model (HF).
- **Kraken** (via eScriptorium) — segmentation-free CNN+BiLSTM, gold standard for
  **historical/typeface-specific print**. Typeface-specific models reach
  **>97.5% char accuracy with ~800–1,000 annotated lines**; generalized Arabic
  model **97.41%** char / **98.46%** Arabic-script-only
  ([arXiv 2402.10943](https://arxiv.org/pdf/2402.10943v1.pdf)).
- **Arabic-Nougat** ([arXiv 2411.17835](https://huggingface.co/papers/2411.17835)) —
  page→Markdown, custom `Aranizer-PBE-86k` tokenizer, released a 1.1B-token corpus.
- **HATFormer** and CNN-BiLSTM-CTC+attention hybrids for manuscripts (CER ~4–8% KHATT-class).

**Takeaway:** a low-parameter Arabic-fine-tuned VLM (QARI-2B / DIMI-7B) beats both
generic `qwen2.5vl:7b` and every classical engine; a **typeface-specific Kraken
model** remains unbeatable if a few hundred book-font lines can be annotated.

---

## 2. Promising open-source tools and codebases

### Highest leverage

| Project | What it is | Why it matters |
|---|---|---|
| **[datalab-to/surya](https://github.com/datalab-to/surya)** (21.5k★) | 650M VLM: OCR + layout + tables + RTL reading order; separate EfficientViT line detector | Already used for line boxes. **Surya 2** adds full-page OCR (83.3% olmOCR-bench, top <3B). ⚠ Weights OpenRAIL-M, free only under $5M. |
| **[KITAB-Bench](https://github.com/mbzuai-oryx/KITAB-Bench)** | 8,809-sample Arabic OCR/doc benchmark (9 domains, 36 subdomains) | De-facto Arabic OCR yardstick; use subsets for model selection. |
| **[h9-tec/Manazir-OCR](https://github.com/h9-tec/Manazir-OCR)** | Arabic-first multi-model framework with **pluggable backends** (Qari, DIMI, OCR-RL2, TrOCR, Qwen2/VL, PaddleOCR-Arabic, Surya) | Models the "try several engines, pick best per page" pattern. |
| **[misraj-ai/qalam](https://github.com/misraj-ai/qalam)** | Rust+Python: logical word order, **base letters not presentation forms**, ligatures, per-page OCR verdict | Solves Unicode/ordering bugs plaguing Arabic extraction. |
| **[ocrmypdf/OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF)** | Standard searchable-PDF tool; v17 fpdf2 renderer supports **RTL (Arabic/Hebrew/Persian)** bbox-aligned text layers | Battle-tested reference for our overlay. |
| **[MohamedAliRashad/arabic-nougat](https://github.com/MohamedAliRashad/arabic-nougat)** | Page→Markdown Arabic OCR suite + large corpus | Relevant to Markdown-first output. |
| **[CAMeL-Lab/arafix_ocr](https://github.com/CAMeL-Lab/arafix_ocr)** | n-gram post-correction (OpenITI/Islamic corpora models exist) | Domain-matched post-correction idea (note: `predict` currently disabled). |

### Classical engines / training frameworks

- **Kraken + [eScriptorium](https://escriptorium.eu/)** — per-typeface historical/print
  training; GUI or `ketos train`. Best accuracy ceiling with modest annotation.
- **Tesseract `ara`** — usable on *clean* print; fine-tuning helps
  ([ClearCypher](https://github.com/ClearCypher/enhancing-tesseract-arabic-text-recognition)
  1038-font `ara.traineddata`; [OmarSamirz](https://github.com/OmarSamirz/Fine-Tuning-an-Arabic-OCR-Model-using-Tesseract-5.0) v5).
- **PaddleOCR** — Arabic `arabic_PP-OCRv5_mobile_rec` at **81.27%**; **not in v6**
  ([issue #16449](https://github.com/PaddlePaddle/PaddleOCR/discussions/16449)). Not competitive here.
- **EasyOCR** — convenient, 18–20% CER on real Arabic scans.

### Dataset resources

KHATT, IFN/ENIT (handwriting); **Muharaf**, RASAM, HistoryAr, ADAB (manuscripts);
**SARD** (843.6K synthetic book pages, 689.7M words, 10 fonts); **APTI** (45M
synthetic word images, 10 fonts × 10 sizes × 4 styles); PATS-A01; BCE-Arabic-v1,
DocLayNet (layout); EvAREST (scene); **HAFITH combined benchmark**
(MUHARAF+KHATT+RASAM); `arabic-img2md` (13.7K page↔Markdown pairs).

---

## 3. Key challenges and community/research solutions

| Challenge | Why it breaks OCR | Best-known solution |
|---|---|---|
| **Cursive connectivity / contextual letterforms** | ع has 4 shapes; no character segmentation | Segmentation-free seq2seq (CTC/attention/VLM). Never force per-character segmentation. |
| **Diacritics (tashkeel)** | Small marks overlap glyphs; inflate CER; engines silently drop them | Diacritic-aware models (QARI, DIMI); **normalize diacritics before scoring**. |
| **Ligatures (esp. لا)** | One glyph → two code points | Explicit ligature mapping; convert presentation forms (FB50–FEFF) to base letters. |
| **RTL + bidi with Latin/numbers** | Positionally scrambled even when chars correct | Unicode BiDi (UAX #9); layout-aware models (Surya reading order, QARI v0.3). |
| **Skew / rotation** | Arabic connectivity very skew-sensitive; 2–3° hurts | Deskew before traditional OCR; test for VLMs (§4.3). |
| **Resolution & small/diacritized text** | <200 DPI fatal; <8pt + diacritics fails | 300 DPI min; **400–600 DPI for diacritized text**. |
| **Historical typefaces / manuscripts** | Train/test typeface mismatch | Typeface-specific Kraken fine-tune (~800–1,000 lines); VLMs help more at page than line level. |
| **Post-OCR errors** (spacing, ك/ي/ح, tanwīn, hamza) | Most common residual errors | n-gram LM; confusion-matrix + regression (**−35% WER**); AraSpell (**CER 1.11%**); ByT5 for noisy text. |
| **No native VLM confidence** | Can't target correction | Grounding/box confidence; Surya error-detection; heuristic ratios (`quality_score`, `ocr_engine.py:357`). |

### Cross-cutting: *"When Do VLMs Help Arabic Manuscript OCR?"* (2026)

[arXiv 2608.22366](https://arxiv.org/pdf/2608.22366.pdf) — most directly actionable:

1. **Preprocessing is not uniformly helpful for VLMs.** Sharpening was the *only*
   operation that improved Qwen (CER 70.4→67.2). **Binarization, denoising,
   contrast enhancement, and combined pipelines degraded performance** — aggressive
   preprocessing destroys small Arabic-script cues.
2. **OCR-conditioned LLM correction helps only when the OCR prior is recoverable**
   (Tesseract→Qwen cut CER 69.2→60.6 on one set, but *hurt* on others). ⇒ Keep
   post-processing gated.
3. **Adopt a "probe-and-adapt" workflow:** annotate 20–50 pages of *our* corpus,
   compare engines, inspect CER/length-ratio/repetition, then commit.

### Benchmark caveat

Surya scores worst-in-class (CER 4.95) in KITAB's *line-recognition* table but
best-in-class (3.9%) in page-level testing. The discrepancy is task/metric/output-format
normalization — **do not trust any published Arabic CER as predictive of our
documents.** Validate locally.

---

## 4. Concrete, actionable recommendations for *this* project

### 4.1 Model selection — run a probe, then likely switch
1. **Build a 30–50 page ground-truth probe set** from our corpus (pages 31–35 of
   the sample book, plus diacritized and footnote-heavy pages). Score CER + WER
   (verbatim **and** diacritic-normalized) + existing LineMatch.
2. **Benchmark these locally-runnable Arabic VLMs against `qwen2.5vl:7b`:** QARI-OCR
   v0.2 (2B) and v0.3 (layout-aware); DIMI-Arabic-OCR v2 (7B); AIN-7B;
   Fanar-2-Oryx-IVU (Apache-2.0); optionally Surya 2 full-page OCR (check license).
3. Swap `OLLAMA_VISION_MODEL` to the winner; keep second as `OLLAMA_VISION_FALLBACK`.
   QARI-2B is far smaller than 7B ⇒ meaningfully reduces `THERMAL_*` pressure.

### 4.2 Fine-tuning — single biggest accuracy lever
If probe results plateau, **LoRA fine-tune Qwen2.5-VL-7B (or 3B) on our font/book**:
- **Kraken-style:** ~800–1,000 annotated *lines* of our typeface ⇒ >97.5% char accuracy.
- **QARI-style:** ~10k synthetic pages in ~11 h on one A6000; generate with
  **SARD / SynthOCR-Gen** (both handle RTL + diacritics; export CRNN/TrOCR/PaddleOCR/Tesseract/HF).
- Augment with geometric/photometric degradations (rotation, blur, noise, JPEG).

### 4.3 Preprocessing — stop binarizing the VLM path; add sharpening
- **Keep `PREPROCESS_MODE=off` default for the VLM** (§3). Do **not** enable
  `--preprocess` (binarize/denoise/CLAHE) for the LLM OCR call — reserve
  binarization for a Tesseract/EasyOCR path only.
- **Add a mild unsharp-mask step** as an opt-in "light+" mode; the one operation
  shown to help VLMs on Arabic.
- **Keep deskew**, conditional (>~0.5° detected), on the **raw** image before
  Surya/LLM (already done for Surya — correct).
- **Raise resolution for diacritized pages:** 300 → **400 DPI**, and raise
  `OLLAMA_MAX_IMAGE_DIM` **1500 → ~2500** (`config.py:102`). At 1500 px we likely
  downscale away diacritics. Gate the higher cap behind a per-page
  "diacritized/heavy" heuristic to control thermal cost.
- **Re-test the contrast-retry logic:** `enhance_contrast` (`preprocess.py:166`)
  is current retry for refused pages, but contrast enhancement can *degrade* VLM
  output. Prefer **sharpening** as the retry transform.

### 4.4 Post-processing — normalize first, correct cautiously, target low-confidence words
- **Always normalize for indexing/search:** Unicode NFC, strip tatweel/kashida,
  normalize alef/hamza/ya/ta-marbuta, normalize Arabic-Indic (٠-٩/۰-۹) and Latin
  digits, convert presentation forms to base letters. Orthogonal to accuracy;
  fixes search/RAG quality (relevant to `text_cleanup.py`, DB sync).
- **Do not run the LLM corrector unconditionally** — keep
  `OLLAMA_POSTPROCESS_ENABLED=false` default. If enabling, only correct
  low-confidence words and **measure before/after CER**.
- Prefer a **ByT5 / AraSpell-style seq2seq** or an **Arabic n-gram LM** over a
  general coder model; purpose-built and cheaper.

### 4.5 Searchable overlay — adopt OCRmyPDF's RTL approach
- Validate our custom overlay against **OCRmyPDF v17's fpdf2 renderer** (explicit
  RTL Arabic support, bbox-aligned). Adopt or mirror its positioning/baseline
  handling. Keep line-level boxes; add **word-level boxes** where the detector is
  confident (Surya supplies polygons + confidence).

### 4.6 Layout, structure, and evaluation
- **Upgrade Surya usage from line-detection to layout + reading-order** to classify
  headers, footnotes, tables in classical texts (footnotes are pervasive). Open
  table leaders: Surya 2 / RT-DETR.
- **Stand up an evaluation harness** computing CER/WER (normalized + verbatim)
  plus LineMatch on the probe set; run against a KITAB-Bench subset. Use
  `ocrevalUAtion`/`ocreval`. Without a probe metric, switching models or enabling
  preprocessing is unsafe.

### 4.7 Quick-win checklist (ordered by ROI)
1. Build the 30–50 page probe set + CER/WER harness. **(enables everything else)**
2. Add QARI v0.2/2B as a candidate; A/B against `qwen2.5vl:7b`. **(accuracy + thermal win)**
3. Raise DPI to 400 and `max_image_dim` to ~2500 for diacritized pages. **(recover lost diacritics)**
4. Disable binarization on the VLM path; add opt-in sharpening; re-test `enhance_contrast` retry.
5. Implement Unicode/base-letter/digit normalization in `text_cleanup.py` for search quality.
6. If accuracy target unmet: LoRA fine-tune on 800–1,000 annotated lines of our typeface.

> §4.7 is tracked as concrete tasks in
> [`08_IMPLEMENTATION_PLAN.md`](08_IMPLEMENTATION_PLAN.md).

---

## References (representative)

**Benchmarks & surveys:** KITAB-Bench [arXiv 2502.14949](https://arxiv.org/abs/2502.14949);
Arabic OCR survey [arXiv 2312.11812](https://arxiv.org/html/2312.11812v1);
"When Do VLMs Help Arabic Manuscript OCR?" [arXiv 2608.22366](https://arxiv.org/pdf/2608.22366.pdf);
Kiessling, open-source Arabic-script OCR [arXiv 2402.10943](https://arxiv.org/pdf/2402.10943v1.pdf).

**Models:** QARI-OCR [arXiv 2506.02295](https://arxiv.org/abs/2506.02295) ·
[HF](https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct);
DIMI v2 [HF](https://huggingface.co/AhmedZaky1/DIMI-Arabic-OCR-V2);
Fanar-2-Oryx-IVU [HF](https://huggingface.co/QCRI/Fanar-2-Oryx-IVU);
Arabic-Nougat [arXiv 2411.17835](https://huggingface.co/papers/2411.17835);
Surya [GitHub](https://github.com/datalab-to/surya) · [Surya 2 blog](https://www.datalab.to/blog/surya-2);
PaddleOCR Arabic [HF 81.27%](https://huggingface.co/PaddlePaddle/arabic_PP-OCRv5_mobile_rec).

**Techniques & tools:** SARD [arXiv 2505.24600](https://arxiv.org/html/2505.24600);
SynthOCR-Gen [arXiv 2601.16113](https://arxiv.org/html/2601.16113v1);
AraSpell [arXiv 2405.06981](https://arxiv.org/html/2405.06981);
OCRmyPDF [docs](https://ocrmypdf.readthedocs.io/en/latest/advanced.html);
Kraken [training docs](https://kraken.re/4.0/training.html);
arafix [GitHub](https://github.com/CAMeL-Lab/arafix_ocr).

**Practical comparisons:** arelion 14-engine Arabic benchmark [arelion.dev](https://arelion.dev/arabic-ocr/);
AnMoon guide [anmoon.org](https://anmoon.org/blog/arabic-ocr-guide.html);
Tesseract quality guidance [tessdoc](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html);
Tesseract Arabic fine-tuning [ClearCypher](https://github.com/ClearCypher/enhancing-tesseract-arabic-text-recognition).

---

## Bottom line

Our architecture is already reasonable (VLM + Surya + optional OpenCV). The
highest-impact changes are: **(1)** measure on a local probe set, **(2)** switch to
an Arabic-fine-tuned VLM (QARI-2B or DIMI-7B), and **(3)** stop VLM-path
binarization while raising resolution for diacritics — three changes likely to
yield large accuracy gains, with the 2B model also reducing thermal risk.
