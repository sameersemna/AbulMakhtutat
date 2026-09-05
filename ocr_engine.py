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

_POSTPROCESS_PROMPT = """You are an Arabic text post-processor. Clean up the following OCR output.

Requirements:
- Fix obvious OCR artifacts and broken words while preserving meaning.
- Ensure right-to-left logical reading order.
- Normalize Markdown syntax (headings, lists, quotes) without changing content.
- Preserve all diacritical marks (Harakat).
- Output ONLY the cleaned Markdown text. Do not add commentary.

OCR TEXT:
{text}
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

    def _is_refusal(self, text: str) -> bool:
        """Return True if the OCR output is a refusal/failure message, not Arabic text."""
        if not text or not text.strip():
            return True
        lowered = text.lower()
        # If the response contains any refusal phrase, treat it as a failure.
        if any(phrase in lowered for phrase in _REFUSAL_PHRASES):
            return True
        # If there is essentially no Arabic content, treat as failure.
        arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
        if arabic_chars < 3:
            return True
        return False

    def is_refusal(self, text: str) -> bool:
        """Public wrapper for refusal detection (used by the processor)."""
        return self._is_refusal(text)

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
