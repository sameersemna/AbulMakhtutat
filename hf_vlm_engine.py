"""HuggingFace vision-language OCR backend (opt-in alternative to Ollama).

Loads an Arabic-fine-tuned VLM (e.g. QARI-OCR, DIMI) in-process via `transformers`
and exposes the same `ocr_page` interface as `OCREngine`, so the pipeline and the
evaluation harness (`eval_ocr.py`) can use it interchangeably.

Enable with `VISION_BACKEND=hf` and `VISION_HF_MODEL=<hf-id>`. Fail-soft: if the
required packages or weights are unavailable at construction, `build_ocr_engine`
falls back to the Ollama engine and the pipeline keeps running. The heavy model
is loaded lazily on first use.

Thermal note: inference is serialized through the same global lock as the Ollama
engine (when `OLLAMA_QUEUE_INFERENCE` is on) to avoid GPU overload.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from PIL import Image

from config import Config, get_config
from ocr_engine import _LINE_MAX_TOKENS, _LINE_PROMPT, _VISION_PROMPT, is_refusal_text, quality_score_text
from preprocess import prepare_line_image, prepare_page_image

logger = logging.getLogger(__name__)


class HFVisionEngine:
    """In-process HuggingFace vision-language OCR engine (lazy-loaded)."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._model: Any = None
        self._processor: Any = None
        self._device: Any = None
        self._fallback: Any = None  # Ollama engine used if the HF model can't load
        self._load_failed = False
        self._check_deps()

    @staticmethod
    def _check_deps() -> None:
        """Raise ImportError if the HF stack is unavailable (factory falls back)."""
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(f"HuggingFace backend requires torch+transformers: {exc}") from exc

    def _load(self) -> None:
        """Load the model, marking the engine permanently failed on error."""
        try:
            self._load_impl()
            self._load_failed = False
        except Exception:
            self._load_failed = True
            raise

    def _load_impl(self) -> None:
        """Load the model + processor (called once, on the first OCR request)."""
        from transformers import AutoProcessor

        model_id = self.config.vision.hf_model
        logger.info("Loading HF vision model %s (this can take a while)...", model_id)

        load_kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
        if self.config.vision.hf_load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            load_kwargs.pop("torch_dtype", None)

        model = None
        try:
            from transformers import AutoModelForVision2Seq

            model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs)
        except Exception:
            from transformers import Qwen2VLForConditionalGeneration

            model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = model.eval()
        self._device = getattr(model, "device", None) or self.config.vision.hf_device
        logger.info("HF vision model %s ready on %s", model_id, self._device)

    def _prepare_pil(self, image_bytes: bytes, max_dim: int | None = None) -> Image.Image:
        """Preprocess (optional) and downscale to a PIL image for the processor.

        Uses the shared `prepare_page_image` so the Ollama and HF backends send
        byte-identical images (required for a fair A/B in eval_ocr.py).
        """
        return prepare_page_image(image_bytes, self.config, max_dim)

    def _generate_sync(
        self,
        image_bytes: bytes,
        max_dim: int | None,
        prompt: str = _VISION_PROMPT,
        max_new_tokens: int | None = None,
        line: bool = False,
    ) -> str:
        """Blocking generation (run in a worker thread). `line` = single line crop."""
        import torch

        if self._model is None:
            self._load()
        im = prepare_line_image(image_bytes) if line else self._prepare_pil(image_bytes, max_dim)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[im], padding=True, return_tensors="pt").to(self._device)
        with torch.no_grad():
            generated = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.config.vision.hf_max_new_tokens
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _get_lock(self) -> asyncio.Lock | None:
        """Share the Ollama global inference lock when queueing is enabled."""
        from ocr_engine import OCREngine

        if not self.config.ollama.queue_inference:
            return None
        if OCREngine._global_lock is None:
            OCREngine._global_lock = asyncio.Lock()
        return OCREngine._global_lock

    async def ocr_page(self, image_bytes: bytes, max_dim: int | None = None) -> tuple[str, str]:
        """OCR a page; returns (text, model_id). Empty text on refusal/failure.

        Fail-soft: if the model could not be loaded (missing / unavailable
        weights), permanently delegate to the Ollama engine so the pipeline keeps
        producing output instead of emitting empty pages for every page.
        """
        if self._load_failed:
            return await self._fallback_ocr(image_bytes, max_dim)
        lock = self._get_lock()
        try:
            if lock is not None:
                async with lock:
                    text = await asyncio.to_thread(self._generate_sync, image_bytes, max_dim)
            else:
                text = await asyncio.to_thread(self._generate_sync, image_bytes, max_dim)
        except Exception as exc:
            logger.error("HF OCR failed: %s", exc)
            if self._load_failed:
                return await self._fallback_ocr(image_bytes, max_dim)
            return "", ""
        if is_refusal_text(text):
            logger.warning("HF model returned a refusal/empty result")
            return "", ""
        return text, self.config.vision.hf_model

    async def ocr_line(self, image_bytes: bytes) -> tuple[str, str]:
        """OCR one cropped text line (line prompt, token cap, full resolution)."""
        if self._load_failed:
            if self._fallback is None:
                from ocr_engine import OCREngine

                self._fallback = OCREngine(self.config)
            return await self._fallback.ocr_line(image_bytes)
        lock = self._get_lock()
        call = (self._generate_sync, image_bytes, None, _LINE_PROMPT, _LINE_MAX_TOKENS, True)
        try:
            if lock is not None:
                async with lock:
                    text = await asyncio.to_thread(*call)
            else:
                text = await asyncio.to_thread(*call)
        except Exception as exc:
            logger.error("HF line OCR failed: %s", exc)
            return "", ""
        if is_refusal_text(text):
            return "", ""
        return text.strip(), self.config.vision.hf_model

    async def _fallback_ocr(self, image_bytes: bytes, max_dim: int | None) -> tuple[str, str]:
        """Delegate to the Ollama engine after the HF model failed to load."""
        if self._fallback is None:
            from ocr_engine import OCREngine

            logger.warning("HF weights unavailable; falling back to the Ollama engine for this run")
            self._fallback = OCREngine(self.config)
        return await self._fallback.ocr_page(image_bytes, max_dim=max_dim)

    def is_refusal(self, text: str) -> bool:
        return is_refusal_text(text)

    def quality_score(self, text: str) -> float:
        return quality_score_text(text)

    async def postprocess(self, text: str) -> str:
        """No-op: the HF backend does not run the Ollama post-process model."""
        return text

    async def ocr_page_with_boxes(self, image_bytes: bytes) -> list[dict]:
        logger.debug("HF backend does not support word boxes; returning []")
        return []

    async def ocr_page_with_line_boxes(self, image_bytes: bytes) -> list[dict]:
        logger.debug("HF backend does not support line boxes; returning []")
        return []

    async def embed(self, text: str) -> list[float]:
        raise RuntimeError("Embeddings are not supported by the HF vision backend; use Ollama for Qdrant.")

    async def close(self) -> None:
        self._model = None
        self._processor = None
        if self._fallback is not None:
            await self._fallback.close()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def build_ocr_engine(config: Config | None = None):
    """Return the configured vision OCR engine, falling back to Ollama.

    Default (`VISION_BACKEND=kraken`) returns `KrakenEngine` when kraken and
    `VISION_KRAKEN_MODEL` are available. `VISION_BACKEND=hf` returns
    `HFVisionEngine` when the HF stack is importable; `VISION_BACKEND=ollama`
    returns `OCREngine`. If a non-Ollama backend is requested but unavailable,
    the factory logs a warning and falls back to Ollama so the pipeline keeps
    working. If a heavy backend imports but its weights fail to load at first
    use, that engine itself falls back to Ollama (see `_fallback_ocr`).
    """
    from ocr_engine import OCREngine

    config = config or get_config()
    if config.vision.backend == "hf":
        try:
            return HFVisionEngine(config)
        except ImportError as exc:
            logger.warning("VISION_BACKEND=hf unavailable (%s); falling back to Ollama", exc)
    if config.vision.backend == "kraken":
        try:
            from kraken_engine import KrakenEngine

            return KrakenEngine(config)
        except ImportError as exc:
            logger.warning("VISION_BACKEND=kraken unavailable (%s); falling back to Ollama", exc)
    return OCREngine(config)
