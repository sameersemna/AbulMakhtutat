"""Surya-based text-line detection for pixel-accurate overlay alignment.

Surya (datalab-to/surya) has the best Arabic text-line detection (mAP@0.50
79.67% on KITAB-Bench) but its *recognition* is poor (CER 4.95). We only use
its DETECTION boxes and pair them with the LLM's accurate Arabic text.

Uses the GPU by default (device='cuda') for speed; the thermal guard gates
inference so the GPU can work hard while throttling safely. Set
OVERLAY_DEVICE=cpu to force CPU if needed.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_predictor(device: str):
    """Load the Surya detection predictor once (lazy, cached per device)."""
    from surya.detection import DetectionPredictor

    predictor = DetectionPredictor.local(device=device)
    logger.info("Surya detection model loaded on %s", device)
    return predictor


def detect_line_boxes_surya(image, device: str = "cuda") -> list[tuple[int, int, int, int]]:
    """Detect text-line bounding boxes using Surya.

    Args:
        image: PIL.Image to detect lines in.
        device: "cuda" (default, fast) or "cpu".

    Returns:
        List of (x0, y0, x1, y1) line boxes in top-to-bottom reading order.
        Returns [] if Surya is unavailable or fails.
    """
    try:
        predictor = _get_predictor(device)
        results = predictor([image])
        lines: list[tuple[int, int, int, int]] = []
        for res in results:
            for b in res.bboxes:
                # b.bbox is [x0, y0, x1, y1]
                lines.append((int(b.bbox[0]), int(b.bbox[1]), int(b.bbox[2]), int(b.bbox[3])))
        lines.sort(key=lambda b: b[1])
        return lines
    except Exception as exc:
        logger.warning("Surya line detection failed on %s: %s", device, exc)
        return []
