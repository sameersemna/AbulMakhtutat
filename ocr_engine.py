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
from preprocess import preprocess

logger = logging.getLogger(__name__)

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
    """Async client for Ollama vision + text post-processing."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._client = httpx.AsyncClient(
            base_url=self.config.ollama.base_url,
            timeout=httpx.Timeout(self.config.ollama.timeout),
        )

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
        """Call Ollama /api/generate with retry + exponential backoff."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.ollama.temperature,
                "num_ctx": self.config.ollama.num_ctx,
            },
        }
        if images:
            payload["images"] = images
        if format_json:
            payload["format"] = "json"

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

    async def ocr_page(self, image_bytes: bytes) -> tuple[str, str]:
        """OCR a single page image. Returns (text, model_used)."""
        b64 = self._prepare_image(image_bytes)
        # Try primary vision model, fall back to secondary on failure.
        for model in (self.config.ollama.vision_model, self.config.ollama.vision_fallback):
            try:
                text = await self._generate(model, _VISION_PROMPT, images=[b64])
                return text, model
            except RuntimeError:
                if model == self.config.ollama.vision_model:
                    logger.warning("Primary vision model failed, trying fallback %s", self.config.ollama.vision_fallback)
                    continue
                raise
        raise RuntimeError("No vision model available")

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
        resp = await self._client.post(
            "/api/embeddings",
            json={"model": self.config.ollama.embed_model, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json().get("embedding", [])
