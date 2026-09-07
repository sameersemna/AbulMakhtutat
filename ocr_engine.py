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
from PIL import Image

from config import Config, get_config
from preprocess import enhance_contrast, preprocess

logger = logging.getLogger(__name__)

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

    async def ocr_page(self, image_bytes: bytes) -> tuple[str, str]:
        """OCR a single page image. Returns (text, model_used).

        Detects refusal/failure responses (e.g. "image is overexposed") and
        retries with contrast enhancement, then falls back to the secondary
        model. Returns ("", "") if all attempts fail, so the failure prose is
        never stored as valid OCR output.
        """
        b64 = self._prepare_image(image_bytes)

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

        # Attempt 2: contrast-enhanced image (fixes "washed out" pages).
        enhanced_b64 = self._prepare_image(enhance_contrast(image_bytes))
        for model in (self.config.ollama.vision_model, self.config.ollama.vision_fallback):
            try:
                text = await self._generate(model, _VISION_PROMPT, images=[enhanced_b64])
                if not self._is_refusal(text):
                    logger.info("Recovered page via contrast enhancement using %s", model)
                    return text, model
            except RuntimeError:
                continue

        # All attempts failed — return empty so the caller marks the page as failed.
        logger.error("All OCR attempts failed for page (refusal persisted)")
        return "", ""

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
        if not text or not text.strip():
            return True
        lowered = text.lower()
        # If the response contains any English refusal phrase, treat it as a failure.
        if any(phrase in lowered for phrase in _REFUSAL_PHRASES):
            return True
        # If the response contains any Arabic refusal phrase, treat it as a failure.
        if any(phrase in text for phrase in _ARABIC_REFUSAL_PHRASES):
            return True
        # If there is essentially no Arabic content, treat as failure.
        arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
        if arabic_chars < 3:
            return True
        return False

    def is_refusal(self, text: str) -> bool:
        """Public wrapper for refusal detection (used by the processor)."""
        return self._is_refusal(text)

    def quality_score(self, text: str) -> float:
        """Heuristic confidence score for OCR output (0.0 = bad, 1.0 = good).

        Vision-LLMs don't emit native confidence, so we approximate it with
        signal-based heuristics (grounded in arXiv 2409.04117, which shows
        confidence signals dramatically improve OCR error detection):
          - Arabic character ratio (vs. Latin garbage / refusal prose)
          - Presence of refusal phrases
          - Length vs. expected page density
        """
        if not text or not text.strip():
            return 0.0
        lowered = text.lower()
        if any(phrase in lowered for phrase in _REFUSAL_PHRASES):
            return 0.0

        total = len(text)
        arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
        # Latin letters (potential OCR garbage / mixed-script noise).
        latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")

        # Arabic ratio: high = good. Penalize heavy Latin content.
        arabic_ratio = arabic / total if total else 0.0
        latin_ratio = latin / total if total else 0.0

        score = arabic_ratio * 0.7  # dominant signal
        score -= latin_ratio * 0.3  # penalty for Latin garbage
        score = max(0.0, min(1.0, score))
        return round(score, 3)

    def _prepare_image(self, image_bytes: bytes) -> str:
        """Preprocess (optional) and downscale the page image, then base64-encode."""
        # Optional OpenCV preprocessing (deskew, binarize, denoise, contrast).
        image_bytes = preprocess(image_bytes, self.config)

        max_dim = self.config.ollama.max_image_dim
        if max_dim and max_dim > 0:
            try:
                im = Image.open(io.BytesIO(image_bytes))
                if max(im.size) > max_dim:
                    im.thumbnail((max_dim, max_dim))
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    image_bytes = buf.getvalue()
            except Exception as exc:
                logger.warning("Image downscale failed, using original: %s", exc)
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
