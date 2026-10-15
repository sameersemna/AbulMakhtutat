"""Kraken recognition backend (in-process, GPU) for the project's own OCR models.

Loads a kraken recognition model (the in-repo "Route A" models trained with
`ketos train`) via kraken's Python API and exposes the same `ocr_page` /
`ocr_line` interface as `OCREngine` and `HFVisionEngine`, so the pipeline and
`eval_ocr.py` can use it interchangeably.

The default backend (`VISION_BACKEND=kraken`); override with `VISION_BACKEND=ollama`
to use the Ollama VLM server, or set `VISION_KRAKEN_MODEL=<path>` (defaults to
`models/run_x_3e4/best_0.9690.safetensors`). Full-page OCR runs kraken's bundled
BLLA line segmenter first (overridable with `VISION_KRAKEN_SEG_MODEL`), then
recognizes every line. `ocr_line` recognizes pre-cropped line images directly
(used by `prefill_line_gt.py`).

Fail-soft: if kraken is not importable or the model file is missing, the engine
factory falls back to the Ollama engine so the pipeline keeps running.

Note: kraken is a line recognizer -- pages are segmented first; there is no
language model, so output is raw CTC decoding reordered to logical order.
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from config import Config, get_config
from ocr_engine import FALLBACK_MODEL_PREFIX, is_refusal_text, quality_score_text

logger = logging.getLogger(__name__)


def line_segmentation(im: Image.Image):
    """Whole-image bbox segmentation (mirrors kraken's `-s/--no-segmentation` mode)."""
    from kraken.containers import BBoxLine, Segmentation

    return Segmentation(type="bbox",
                        imagename="",
                        text_direction="horizontal-tb",
                        script_detection=False,
                        lines=[BBoxLine(id="line-0", bbox=(0, 0, im.width, im.height))])


class KrakenEngine:
    """In-process Kraken recognition engine (lazy-loaded)."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._model: Any = None
        self._seg_model: Any = None
        self._fallback: Any = None
        self._load_failed = False
        self._check_deps()
        model_path = Path(self.config.vision.kraken_model).expanduser()
        if not model_path.is_file():
            raise ImportError(f"kraken model not found: {model_path}")

    @staticmethod
    def _check_deps() -> None:
        """Raise ImportError if kraken is unavailable (factory falls back to Ollama)."""
        try:
            import torch  # noqa: F401
            from kraken.configs import RecognitionInferenceConfig  # noqa: F401
            from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(f"Kraken backend requires kraken+torch: {exc}") from exc

    def _inference_config(self):
        from kraken.configs import RecognitionInferenceConfig

        return RecognitionInferenceConfig(batch_size=self.config.vision.kraken_batch_size,
                                          num_line_workers=0,  # in-process extraction (fork-safe)
                                          padding=16,
                                          bidi_reordering=True,
                                          text_direction="horizontal-tb",
                                          device=self.config.vision.kraken_device)

    def _segmentation_config(self):
        from kraken.configs import SegmentationInferenceConfig

        return SegmentationInferenceConfig(text_direction=self.config.vision.kraken_text_direction,
                                           device=self.config.vision.kraken_device)

    def _load(self) -> None:
        """Load the recognition model; mark the engine permanently failed on error."""
        try:
            from kraken.tasks import RecognitionTaskModel

            path = Path(self.config.vision.kraken_model).expanduser()
            self._model = RecognitionTaskModel.load_model(str(path))
            self._load_failed = False
        except Exception:
            self._load_failed = True
            raise

    def _segment_page(self, im: Image.Image):
        from kraken.tasks import SegmentationTaskModel

        if self._seg_model is None:
            path = self.config.vision.kraken_seg_model.strip() or None
            self._seg_model = SegmentationTaskModel.load_model(path)
        segmentation = self._seg_model.predict(im, self._segmentation_config())
        if (self._model is not None and getattr(self._model, "seg_type", None) == "bbox"
                and segmentation.type != "bbox"):
            segmentation = self._to_bbox_segmentation(segmentation)
        return segmentation

    @staticmethod
    def _to_bbox_segmentation(segmentation):
        """Convert baseline lines to bounding-box lines.

        The in-repo models are trained on bbox line crops (`ketos -f path`).
        Feeding blla *baseline* polygons to a bbox-trained model degrades
        recognition badly (measured 2026-10-14: 1/7 vs 7/7 target words on a
        book-9 page); collapsing each baseline's boundary polygon to its
        axis-aligned bounding box restores line-level accuracy.
        """
        import numpy as np
        from kraken.containers import BBoxLine, Segmentation

        lines = []
        for line in segmentation.lines:
            poly = getattr(line, "boundary", None)
            if poly is None:
                poly = getattr(line, "bbox", None)
            box = np.asarray(poly)
            lines.append(BBoxLine(id=line.id,
                                  bbox=(int(box[:, 0].min()), int(box[:, 1].min()),
                                        int(box[:, 0].max()), int(box[:, 1].max())),
                                  base_dir=getattr(line, "base_dir", None)))
        return Segmentation(type="bbox",
                             imagename=segmentation.imagename,
                             text_direction=segmentation.text_direction,
                             script_detection=False,
                             lines=lines,
                             regions=segmentation.regions,
                             line_orders=segmentation.line_orders)

    @staticmethod
    def _to_pil(image_bytes: bytes) -> Image.Image:
        im = Image.open(io.BytesIO(image_bytes))
        return im if im.mode in ("RGB", "L") else im.convert("RGB")

    def _recognize_sync(self, im: Image.Image, page: bool) -> tuple[str, float | None]:
        """Blocking recognition; returns the text and a mean confidence (or None)."""
        if self._model is None:
            self._load()
        segmentation = self._segment_page(im) if page else line_segmentation(im)
        records = list(self._model.predict(im=im, segmentation=segmentation, config=self._inference_config()))
        text = "\n".join(r.prediction for r in records)
        confidence = None
        try:
            values = [c for r in records for c in (getattr(r, "confidences", None) or [])]
            if values:
                confidence = float(sum(values) / len(values))
        except Exception:
            confidence = None
        return text, confidence

    def _get_lock(self) -> asyncio.Lock | None:
        """Share the Ollama global inference lock when queueing is enabled."""
        from ocr_engine import OCREngine

        if not self.config.ollama.queue_inference:
            return None
        if OCREngine._global_lock is None:
            OCREngine._global_lock = asyncio.Lock()
        return OCREngine._global_lock

    async def _run(self, image_bytes: bytes, page: bool) -> tuple[str, float | None]:
        if self._load_failed:
            raise RuntimeError("kraken model failed to load")
        lock = self._get_lock()
        im = self._to_pil(image_bytes)
        if lock is not None:
            async with lock:
                return await asyncio.to_thread(self._recognize_sync, im, page)
        return await asyncio.to_thread(self._recognize_sync, im, page)

    async def ocr_page(self, image_bytes: bytes, max_dim: int | None = None) -> tuple[str, str]:
        """OCR a full page (BLLA segmentation + recognition). Empty text on failure.

        `max_dim` is accepted for interface compatibility and ignored: kraken
        normalizes line height internally and segments at full render resolution.
        """
        try:
            text, _ = await self._run(image_bytes, page=True)
        except Exception as exc:
            logger.error("Kraken OCR failed: %s", exc)
            return await self._fallback_ocr(image_bytes, max_dim)
        if is_refusal_text(text):
            logger.warning("Kraken returned an empty/refusal result")
            return "", ""
        return text, self.config.vision.kraken_model

    async def ocr_line(self, image_bytes: bytes) -> tuple[str, str]:
        """OCR one cropped text line (no segmentation)."""
        text, _ = await self.ocr_line_scored(image_bytes)
        return text, self.config.vision.kraken_model

    async def ocr_line_scored(self, image_bytes: bytes) -> tuple[str, float | None]:
        """Like `ocr_line`, but also returns the mean per-character confidence."""
        try:
            text, confidence = await self._run(image_bytes, page=False)
        except Exception as exc:
            logger.error("Kraken line OCR failed: %s", exc)
            text, _ = await self._fallback_line(image_bytes)
            return text, None
        return ("" if is_refusal_text(text) else text), confidence

    async def _fallback_ocr(self, image_bytes: bytes, max_dim: int | None) -> tuple[str, str]:
        """Delegate to the Ollama engine after the kraken model failed.

        The returned model tag is prefixed with `FALLBACK_MODEL_PREFIX` so the
        pipeline can refuse to cache the (lower-quality) fallback text.
        """
        if self._fallback is None:
            from ocr_engine import OCREngine

            logger.warning("Kraken model unavailable; falling back to the Ollama engine for this run")
            self._fallback = OCREngine(self.config)
        text, model = await self._fallback.ocr_page(image_bytes, max_dim=max_dim)
        return text, f"{FALLBACK_MODEL_PREFIX}{model}"

    async def _fallback_line(self, image_bytes: bytes) -> tuple[str, str]:
        if self._fallback is None:
            from ocr_engine import OCREngine

            logger.warning("Kraken model unavailable; falling back to the Ollama engine for this run")
            self._fallback = OCREngine(self.config)
        return await self._fallback.ocr_line(image_bytes)

    def is_refusal(self, text: str) -> bool:
        return is_refusal_text(text)

    def quality_score(self, text: str) -> float:
        return quality_score_text(text)

    async def postprocess(self, text: str) -> str:
        """No-op: no Ollama post-process model on this backend."""
        return text

    async def ocr_page_with_boxes(self, image_bytes: bytes) -> list[dict]:
        logger.debug("Kraken backend does not support word boxes; returning []")
        return []

    async def ocr_page_with_line_boxes(self, image_bytes: bytes) -> list[dict]:
        logger.debug("Kraken backend does not support line boxes; returning []")
        return []

    async def embed(self, text: str) -> list[float]:
        raise RuntimeError("Embeddings are not supported by the Kraken backend; use Ollama for Qdrant.")

    async def close(self) -> None:
        self._model = None
        self._seg_model = None
        if self._fallback is not None:
            await self._fallback.close()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
