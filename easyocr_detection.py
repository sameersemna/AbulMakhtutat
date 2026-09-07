"""EasyOCR-based text-line detection for overlay alignment.

EasyOCR has the best Arabic *recognition* (CER 0.20 on KITAB-Bench) and good
detection. We use its DETECTION boxes (grouped into lines) paired with the
LLM's accurate Arabic text. Kept as a selectable alternative to Surya for
A/B comparison and enhancement iterations.

Uses the GPU by default (device='cuda'); the thermal guard gates inference.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_reader(device: str):
    """Load the EasyOCR reader once (lazy, cached per device)."""
    import easyocr

    reader = easyocr.Reader(["ar"], gpu=(device == "cuda"), verbose=False)
    logger.info("EasyOCR reader loaded on %s", device)
    return reader


def _group_words_to_lines(word_boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Group word boxes into lines by vertical overlap (top-to-bottom)."""
    boxes = sorted(word_boxes, key=lambda b: b[1])
    lines: list[list[tuple[int, int, int, int]]] = []
    for box in boxes:
        x0, y0, x1, y1 = box
        bh = y1 - y0
        placed = False
        for line in lines:
            ly0 = min(b[1] for b in line)
            ly1 = max(b[3] for b in line)
            overlap = min(y1, ly1) - max(y0, ly0)
            if overlap > 0 and overlap >= 0.5 * min(bh, ly1 - ly0):
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])
    result = []
    for line in lines:
        x0 = min(b[0] for b in line)
        y0 = min(b[1] for b in line)
        x1 = max(b[2] for b in line)
        y1 = max(b[3] for b in line)
        result.append((int(x0), int(y0), int(x1), int(y1)))
    result.sort(key=lambda b: b[1])
    return result


def detect_line_boxes_easyocr(image, device: str = "cuda") -> list[tuple[int, int, int, int]]:
    """Detect text-line bounding boxes using EasyOCR.

    Args:
        image: PIL.Image to detect lines in.
        device: "cuda" (default, fast) or "cpu".

    Returns:
        List of (x0, y0, x1, y1) line boxes in top-to-bottom reading order.
        Returns [] if EasyOCR is unavailable or fails.
    """
    try:
        import numpy as np

        reader = _get_reader(device)
        arr = np.array(image)
        results = reader.readtext(arr, detail=1, paragraph=False)
        word_boxes = []
        for bbox, _text, _conf in results:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            word_boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
        return _group_words_to_lines(word_boxes)
    except Exception as exc:
        logger.warning("EasyOCR line detection failed on %s: %s", device, exc)
        return []
