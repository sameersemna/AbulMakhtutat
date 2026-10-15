"""Ollama vision OCR engine with exponential backoff and batching.

Queries the local Ollama server (Dell Pro Max GB10) to OCR Arabic page images
using a vision model, then optionally post-processes the raw text with a
text model to enforce RTL logical order and clean Markdown syntax.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from typing import Any

import httpx

from config import Config, get_config
from preprocess import enhance_contrast, prepare_line_image, prepare_page_image, sharpen_image

logger = logging.getLogger(__name__)

# Model-tag prefix used by the fallback paths (kraken/HF engine -> Ollama) so
# callers can tell a degraded fallback result from the configured engine's own
# output. `pdf_processor` refuses to cache fallback pages (they are retried on a
# later run instead of being silently persisted at much lower quality).
FALLBACK_MODEL_PREFIX = "ollama-fallback:"

# Phrases the vision model emits when it *refuses* to transcribe (instead of
# returning Arabic text). These are English "I can't read this" responses that
# must never be stored as valid OCR output.
_REFUSAL_PHRASES = (
    "overexposed",
    "washed out",
    "nearly illegible",
    "no clear or readable",
    "no transcribable",
    "cannot be discerned",
    "unreadable",
    "not legible",
    "illegible",
    "no readable text",
    "no text",
    "blank page",
    "empty page",
    "no arabic",
    "no visible text",
    "unable to read",
    "cannot read",
    "can't read",
    "cannot transcribe",
    "unable to transcribe",
    "no content",
    "nothing to transcribe",
    "if you have a clearer",
    "provide it for accurate",
)

# Arabic refusal phrases — the model sometimes refuses in Arabic instead of
# English (e.g. "لا يوجد نص قابل للقراءة في هذه الصفحة").
_ARABIC_REFUSAL_PHRASES = (
    "لا يوجد نص",
    "لا يوجد نص قابل للقراءة",
    "غير قابل للقراءة",
    "لا يمكن قراءة",
    "لا يمكن التعرف",
    "نص فارغ",
    "صفحة فارغة",
    "لا توجد كتابة",
    "لا يوجد محتوى",
    "لا يمكن نسخ",
    "لا يمكن استخراج",
    "الصورة غير واضحة",
    "النص غير مقروء",
    "لا يوجد نص عربي",
)

_VISION_PROMPT = """You are an expert Arabic OCR engine. Transcribe the Arabic text in this page image EXACTLY as it appears.

Requirements:
- Preserve all diacritical marks (Harakat: fatha, damma, kasra, sukun, shadda, tanwin).
- Preserve classical Arabic typography, footnotes, and structural layout.
- Output clean Markdown: use `#`, `##`, `###` for headings, `-` for lists, and `>` for quotes.
- Keep the original right-to-left reading order.
- Do NOT translate, summarize, or add commentary. Output ONLY the transcribed text.
- Do NOT repeat any watermark, stamp, or library logo text more than once.
- If a region is unreadable, skip it silently rather than guessing.
"""

# Prompt for a single cropped text line (annotation prefill). The page prompt asks
# for Markdown, which makes the model invent headings/quotes on a bare line crop.
# Mirrored by train_lora.py so the fine-tuned model sees the same instruction.
_LINE_PROMPT = (
    "Transcribe the Arabic text in this line image exactly as it appears, "
    "preserving all diacritical marks (harakat). Output only the transcribed text."
)
# Token cap for one line: a printed line is ~15-25 words, so anything longer is a
# repetition loop, not text.
_LINE_MAX_TOKENS = 256

_POSTPROCESS_PROMPT = """You are an expert Arabic OCR post-corrector. The following text was extracted from a scanned Arabic document and may contain OCR errors.

Correct it carefully:
- Fix obvious OCR artifacts (broken words, misread letters, stray Latin characters) while preserving the original meaning.
- Do NOT add, remove, or reorder content — only correct what is clearly wrong.
- Preserve ALL diacritical marks (Harakat) exactly as they appear.
- Preserve Markdown structure (headings, lists, quotes) and right-to-left order.
- If a word is ambiguous, leave it unchanged rather than guessing.
- Output ONLY the corrected text. Do not add commentary or explanations.

OCR TEXT:
{text}
"""

# Prompt for word-level OCR with bounding boxes. Qwen2.5-VL supports visual
# grounding and returns `bbox_2d` coordinates in the image's pixel space.
# Word-level boxes (like hOCR `ocrx_word`) give the tightest alignment for
# invisible text layers — the same approach Google Books / Internet Archive use.
_BBOX_PROMPT = """Detect all Arabic WORDS in this image. For each word, return its text and bounding box coordinates.

Output a JSON array of objects, each with:
- "bbox_2d": [x1, y1, x2, y2] — pixel coordinates of the word (top-left and bottom-right corners).
- "text": the Arabic text of that word.

Requirements:
- Preserve all diacritical marks (Harakat).
- Keep the original right-to-left reading order (top-to-bottom, right-to-left within a line).
- Do NOT translate, summarize, or add commentary. Output ONLY the JSON array.
"""

# Prompt for LINE-level OCR with bounding boxes. Line boxes are more accurate
# than word boxes for Arabic (the model detects line boundaries reliably, while
# word boundaries are unreliable due to cursive script + diacritics).
_LINE_BBOX_PROMPT = """Detect all Arabic text LINES in this image. For each line, return its full text and bounding box coordinates.

Output a JSON array of objects, each with:
- "bbox_2d": [x1, y1, x2, y2] — pixel coordinates of the line (top-left and bottom-right corners).
- "text": the full Arabic text of that line.

Requirements:
- Preserve all diacritical marks (Harakat).
- Keep the original top-to-bottom reading order.
- Do NOT translate, summarize, or add commentary. Output ONLY the JSON array.
"""


def is_refusal_text(text: str) -> bool:
    """Return True if OCR output is a refusal/failure message, not Arabic text.

    Shared by all vision backends (Ollama, HuggingFace) so refusal handling is
    identical regardless of engine.
    """
    if not text or not text.strip():
        return True
    lowered = text.lower()
    # English refusal phrases.
    if any(phrase in lowered for phrase in _REFUSAL_PHRASES):
        return True
    # Arabic refusal phrases.
    if any(phrase in text for phrase in _ARABIC_REFUSAL_PHRASES):
        return True
    # Essentially no Arabic content => failure.
    arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    return arabic_chars < 3


def quality_score_text(text: str) -> float:
    """Heuristic confidence score for OCR output (0.0 = bad, 1.0 = good).

    Vision-LLMs don't emit native confidence, so approximate it with signal-based
    heuristics (grounded in arXiv 2409.04117): Arabic character ratio (vs. Latin
    garbage / refusal prose), refusal phrases, and a Latin-content penalty.
    """
    if not text or not text.strip():
        return 0.0
    lowered = text.lower()
    if any(phrase in lowered for phrase in _REFUSAL_PHRASES):
        return 0.0

    total = len(text)
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    arabic_ratio = arabic / total if total else 0.0
    latin_ratio = latin / total if total else 0.0

    score = arabic_ratio * 0.7  # dominant signal
    score -= latin_ratio * 0.3  # penalty for Latin garbage
    return round(max(0.0, min(1.0, score)), 3)


class OCREngine:
    """Async client for Ollama vision + text post-processing.

    Serializes all inference calls through a global queue (asyncio lock) so
    only one runs at a time, preventing GPU overload/overheating. Models stay
    resident in GPU memory (default keep-alive) to avoid reload latency.
    """

    # Global queue (lock) shared across all OCREngine instances in the process.
    _global_lock: asyncio.Lock | None = None

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._client = httpx.AsyncClient(
            base_url=self.config.ollama.base_url,
            timeout=httpx.Timeout(self.config.ollama.timeout),
        )
        if OCREngine._global_lock is None:
            OCREngine._global_lock = asyncio.Lock()

    async def close(self) -> None:
        await self._client.aclose()

    async def _generate(
        self,
        model: str,
        prompt: str,
        images: list[str] | None = None,
        *,
        format_json: bool = False,
        num_predict: int | None = None,
    ) -> str:
        """Call Ollama /api/generate with retry + exponential backoff.

        Serialized through a global queue so only one inference runs at a time.
        Models remain resident (keep-alive) to avoid reload latency.
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.config.ollama.keep_alive,
            "options": {
                "temperature": self.config.ollama.temperature,
                "num_ctx": self.config.ollama.num_ctx,
            },
        }
        if images:
            payload["images"] = images
        if num_predict:
            payload["options"]["num_predict"] = num_predict
        if format_json:
            payload["format"] = "json"

        lock = OCREngine._global_lock if self.config.ollama.queue_inference else None

        async def _call() -> str:
            last_err: Exception | None = None
            for attempt in range(self.config.ollama.max_retries):
                try:
                    resp = await self._client.post("/api/generate", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("response", "")
                except (httpx.HTTPError, ValueError) as exc:
                    last_err = exc
                    wait = self.config.ollama.backoff_base ** attempt
                    logger.warning(
                        "Ollama call failed (attempt %d/%d) for model %s: %s — retrying in %.1fs",
                        attempt + 1,
                        self.config.ollama.max_retries,
                        model,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
            raise RuntimeError(f"Ollama call failed after {self.config.ollama.max_retries} attempts: {last_err}")

        if lock is not None:
            async with lock:
                return await _call()
        return await _call()

    async def ocr_page(self, image_bytes: bytes, max_dim: int | None = None) -> tuple[str, str]:
        """OCR a single page image. Returns (text, model_used).

        `max_dim` overrides the vision model's long-edge downscale for this call
        (e.g. a larger cap for diacritized "heavy" pages); None uses the config
        default so existing callers are unaffected.

        Detects refusal/failure responses (e.g. "image is overexposed") and
        retries with contrast enhancement, then falls back to the secondary
        model. Returns ("", "") if all attempts fail, so the failure prose is
        never stored as valid OCR output.
        """
        b64 = self._prepare_image(image_bytes, max_dim)

        # Attempt 1: primary model, original image.
        for model in (self.config.ollama.vision_model, self.config.ollama.vision_fallback):
            try:
                text = await self._generate(model, _VISION_PROMPT, images=[b64])
                if not self._is_refusal(text):
                    return text, model
                logger.warning("Model %s refused page (unreadable/overexposed); retrying with contrast enhancement", model)
                break  # move to enhanced-image retry
            except RuntimeError:
                if model == self.config.ollama.vision_model:
                    logger.warning("Primary vision model failed, trying fallback %s", self.config.ollama.vision_fallback)
                    continue
                raise

        # Attempt 2: retry with an image transform (contrast and/or sharpening).
        # Sharpening is preferred for vision LLMs (arXiv 2608.22366). The default
        # mode "contrast" preserves the original single-transform behaviour.
        mode = self.config.ollama.retry_mode.lower()
        transforms: list[tuple[str, object]] = []
        if mode in ("contrast", "both"):
            transforms.append(("contrast", enhance_contrast))
        if mode in ("sharpen", "both"):
            transforms.append(("sharpen", sharpen_image))
        if not transforms:
            transforms.append(("contrast", enhance_contrast))

        for label, transform in transforms:
            retry_b64 = self._prepare_image(transform(image_bytes), max_dim)  # type: ignore[operator]
            for model in (self.config.ollama.vision_model, self.config.ollama.vision_fallback):
                try:
                    text = await self._generate(model, _VISION_PROMPT, images=[retry_b64])
                    if not self._is_refusal(text):
                        logger.info("Recovered page via %s enhancement using %s", label, model)
                        return text, model
                except RuntimeError:
                    continue

        # All attempts failed — return empty so the caller marks the page as failed.
        logger.error("All OCR attempts failed for page (refusal persisted)")
        return "", ""

    async def ocr_line(self, image_bytes: bytes) -> tuple[str, str]:
        """OCR one cropped text line. Returns (text, model_used); ("", "") on failure.

        Uses the line prompt and a token cap, and sends the crop at full
        resolution on a white margin (`prepare_line_image`).
        """
        buf = io.BytesIO()
        prepare_line_image(image_bytes).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        model = self.config.ollama.vision_model
        try:
            text = await self._generate(model, _LINE_PROMPT, images=[b64], num_predict=_LINE_MAX_TOKENS)
        except RuntimeError as exc:
            logger.warning("Line OCR failed: %s", exc)
            return "", ""
        if self._is_refusal(text):
            return "", ""
        return text.strip(), model

    async def ocr_page_with_boxes(self, image_bytes: bytes) -> list[dict]:
        """OCR a page and return text lines with bounding boxes.

        Uses Qwen2.5-VL's visual grounding to return a list of
        {"bbox_2d": [x1,y1,x2,y2], "text": "..."} dicts, where coordinates are
        in the (downscaled) image's pixel space. Returns [] on failure.

        This enables word/line-aligned invisible text layers in the overlay PDF.
        """
        import json as _json
        import re as _re

        b64 = self._prepare_image(image_bytes)
        for model in (self.config.ollama.vision_model, self.config.ollama.vision_fallback):
            try:
                raw = await self._generate(model, _BBOX_PROMPT, images=[b64])
                parsed = self._parse_bbox_response(raw)
                if parsed:
                    return parsed
            except RuntimeError:
                continue
        return []

    async def ocr_page_with_line_boxes(self, image_bytes: bytes) -> list[dict]:
        """OCR a page and return text LINES with bounding boxes.

        Line boxes are more accurate than word boxes for Arabic (the model
        detects line boundaries reliably, while word boundaries are unreliable
        due to cursive script + diacritics). Returns a list of
        {"bbox_2d": [x1,y1,x2,y2], "text": "..."} in downscaled pixel space.
        """
        b64 = self._prepare_image(image_bytes)
        for model in (self.config.ollama.vision_model, self.config.ollama.vision_fallback):
            try:
                raw = await self._generate(model, _LINE_BBOX_PROMPT, images=[b64])
                parsed = self._parse_bbox_response(raw)
                if parsed:
                    return parsed
            except RuntimeError:
                continue
        return []

    @staticmethod
    def _parse_bbox_response(raw: str) -> list[dict]:
        """Parse the model's bbox response into a list of {bbox_2d, text} dicts.

        The model may wrap the JSON in markdown fences or prefix it with prose.
        We extract the first JSON array of objects containing bbox_2d + text.
        """
        import json as _json
        import re as _re

        if not raw or not raw.strip():
            return []

        # Strip markdown code fences if present.
        cleaned = _re.sub(r"```(?:json)?", "", raw).strip()

        # Try to find a JSON array in the text.
        # First, try parsing the whole thing as JSON.
        try:
            data = _json.loads(cleaned)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict) and "bbox_2d" in d and "text" in d]
        except (ValueError, TypeError):
            pass

        # Fallback: extract individual {"bbox_2d": [...], "text": "..."} objects.
        results = []
        # Match objects with bbox_2d and text keys (order may vary).
        pattern = _re.compile(
            r'\{\s*"bbox_2d"\s*:\s*\[([^\]]+)\]\s*,\s*"text"\s*:\s*"([^"]*)"\s*\}'
            r'|\{\s*"text"\s*:\s*"([^"]*)"\s*,\s*"bbox_2d"\s*:\s*\[([^\]]+)\]\s*\}'
        )
        for m in pattern.finditer(raw):
            if m.group(1) is not None:  # bbox first
                coords = [int(float(x)) for x in m.group(1).split(",")]
                text = m.group(2)
            else:  # text first
                coords = [int(float(x)) for x in m.group(4).split(",")]
                text = m.group(3)
            if len(coords) == 4:
                results.append({"bbox_2d": coords, "text": text})
        return results

    def _is_refusal(self, text: str) -> bool:
        """Return True if the OCR output is a refusal/failure message, not Arabic text."""
        return is_refusal_text(text)

    def is_refusal(self, text: str) -> bool:
        """Public wrapper for refusal detection (used by the processor)."""
        return is_refusal_text(text)

    def quality_score(self, text: str) -> float:
        """Heuristic confidence score for OCR output (0.0 = bad, 1.0 = good)."""
        return quality_score_text(text)

    def _prepare_image(self, image_bytes: bytes, max_dim: int | None = None) -> str:
        """Preprocess (optional), downscale, and base64-encode the page image.

        `max_dim` overrides `config.ollama.max_image_dim` when provided; None uses
        the config default. A value of 0 disables downscaling for the call. Uses the
        shared `prepare_page_image` so every backend sees an identical image.
        """
        try:
            im = prepare_page_image(image_bytes, self.config, max_dim)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            logger.warning("Image prepare failed, using original bytes: %s", exc)
            return base64.b64encode(image_bytes).decode("ascii")

    async def postprocess(self, text: str) -> str:
        """Clean OCR text with the post-processing model (optional)."""
        if not self.config.ollama.postprocess_enabled or not text.strip():
            return text
        try:
            prompt = _POSTPROCESS_PROMPT.format(text=text)
            return await self._generate(self.config.ollama.postprocess_model, prompt)
        except RuntimeError as exc:
            logger.warning("Post-processing failed, returning raw text: %s", exc)
            return text

    async def embed(self, text: str) -> list[float]:
        """Generate a vector embedding for a text snippet (for Qdrant)."""
        payload = {
            "model": self.config.ollama.embed_model,
            "prompt": text,
            "keep_alive": self.config.ollama.keep_alive,
        }
        lock = OCREngine._global_lock if self.config.ollama.queue_inference else None

        async def _call() -> list[float]:
            resp = await self._client.post("/api/embeddings", json=payload)
            resp.raise_for_status()
            return resp.json().get("embedding", [])

        if lock is not None:
            async with lock:
                return await _call()
        return await _call()
